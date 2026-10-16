import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from ..base_module import BaseModule
from ...config import OrcheConfig

class OrcheConv2d(BaseModule):
    def __init__(self, module: nn.Conv2d, orche_config: OrcheConfig, is_first_layer: bool = False):
        super(OrcheConv2d, self).__init__(module, orche_config)
        self.is_first_layer = is_first_layer

    def naive_forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.module(x)
        return output
    
    def sliced_forward(self, x: torch.Tensor) -> torch.Tensor:
        config = self.orche_config
        b, c, h, w = x.shape 

        stride = self.module.stride[0]
        padding = self.module.padding[0]

        idx = config.split_idx()
        portion_assigned = self.coordinator.get_portion_assigned_for_batch()
        portion_sum = self.coordinator.get_portion_sum_for_batch()
        h_unit = h // stride // portion_sum
        h_begin = h_unit * sum(portion_assigned[:idx]) * stride - padding
        h_end = h_unit * sum(portion_assigned[:idx+1]) * stride + padding

        final_padding = [padding, padding, 0, 0] 
        if h_begin < 0:
            h_begin = 0
            final_padding[2] = padding
        if h_end > h:
            h_end = h
            final_padding[3] = padding
        sliced_input = x[:, :, h_begin:h_end, :]
        padded_input = F.pad(sliced_input, final_padding, mode="constant")
        final_output = F.conv2d(padded_input, self.module.weight, self.module.bias, stride=stride, padding="valid")
        return final_output

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        orche_config = self.orche_config
        b, c, h, w = x.shape
        idx = orche_config.split_idx()
        if (self.counter<= orche_config.warmup_steps or \
            (self.counter - orche_config.warmup_steps) % self.comm_interval == 0) and\
            self.comm_manager is not None and \
            self.comm_manager.handles is not None and \
            self.idx is not None:
            if self.comm_manager.handles[self.idx] is not None:
                self.comm_manager.handles[self.idx].wait()
                self.comm_manager.handles[self.idx] = None

        if self.is_first_layer:
            full_x = x
            output = self.sliced_forward(full_x)
        else:
            boundary_size = self.module.padding[0] 
            if self.buffer_list is None: 
                if self.comm_manager.buffer_list is None: 
                    self.idx = self.comm_manager.register_tensor(
                        shape=[2, b, c, boundary_size, w],
                        torch_dtype=x.dtype,
                        layer_type="conv2d",
                    )
                else:
                    self.buffer_list = self.comm_manager.get_buffer_list(self.idx)

            if self.buffer_list is None: 
                output = self.naive_forward(x)

            else: 
                def create_padded_x():
                    if idx == 0: 
                        concat_x = torch.cat([x, self.buffer_list[idx + 1][0]], dim=2)
                        padded_x = F.pad(concat_x, [0, 0, boundary_size, 0], mode="constant")
                    elif idx == orche_config.n_device_per_batch - 1:
                        concat_x = torch.cat([self.buffer_list[idx - 1][1], x], dim=2)
                        padded_x = F.pad(concat_x, [0, 0, 0, boundary_size], mode="constant")
                    else:
                        padded_x = torch.cat(
                            [
                                self.buffer_list[idx - 1][1],
                                x,
                                self.buffer_list[idx + 1][0],
                            ],
                            dim=2,
                        )
                    return padded_x
                boundary = torch.stack([x[:, :, :boundary_size, :], x[:, :, -boundary_size:, :]], dim=0)

                if orche_config.mode == "full_sync" or self.counter <= orche_config.warmup_steps:
                    dist.all_gather(self.buffer_list, boundary, group=orche_config.batch_group, async_op=False)
                    padded_x = create_padded_x()
                    output = F.conv2d(
                        padded_x,
                        self.module.weight,
                        self.module.bias,
                        stride=self.module.stride[0],
                        padding=(0, self.module.padding[1]),
                    )
                else:
                    padded_x = create_padded_x()
                    output = F.conv2d(
                        padded_x,
                        self.module.weight,
                        self.module.bias,
                        stride=self.module.stride[0],
                        padding=(0, self.module.padding[1]),
                    ) 
                    if orche_config.mode != "no_sync" and \
                        (self.counter - orche_config.warmup_steps) % self.comm_interval == 0: 
                        self.comm_manager.enqueue(self.idx, boundary)
        self.counter += 1
        return output

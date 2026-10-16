import torch
import time
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from torch import distributed as dist, nn

from .base_model import BaseModel
from ..modules.stpp.attn import OrcheSelfAttention, OrcheCrossAttention
from ..modules.base_module import BaseModule
from ..modules.stpp.conv2d import OrcheConv2d
from ..modules.stpp.groupnorm import OrcheGroupNorm
from ..config import OrcheConfig
from ..coordinator import OrcheCoordinator

class OrcheUNetSTPP(BaseModel): 
    def __init__(self, model: UNet2DConditionModel, orche_config: OrcheConfig, coordinator: OrcheCoordinator):
        assert isinstance(model, UNet2DConditionModel)
        super(OrcheUNetSTPP, self).__init__(model, orche_config)
        self.idx = orche_config.split_idx()

        if orche_config.world_size > 1:
            if orche_config.split_batch and orche_config.do_classifier_free_guidance:
                raise NotImplementedError("We have not yet adapted it for batch splitting technology.")
            if orche_config.n_device_per_batch > 1:
                self.set_coordinator(coordinator)
            if orche_config.n_device_per_batch > 1:
                for _, module in model.named_modules():
                    if isinstance(module, BaseModule):
                        continue
                    for subname, submodule in module.named_children():
                        if isinstance(submodule, nn.Conv2d):
                            kernel_size = submodule.kernel_size
                            if kernel_size == (1, 1) or kernel_size == 1:
                                continue
                            wrapped_submodule = OrcheConv2d(
                                submodule, orche_config, is_first_layer=subname == "conv_in"
                            )
                            setattr(module, subname, wrapped_submodule)
                        elif isinstance(submodule, Attention):
                            if subname == "attn1": 
                                wrapped_submodule = OrcheSelfAttention(submodule, orche_config)
                            else:
                                assert subname == "attn2"
                                wrapped_submodule = OrcheCrossAttention(submodule, orche_config)
                            setattr(module, subname, wrapped_submodule)
                        elif isinstance(submodule, nn.GroupNorm):
                            wrapped_submodule = OrcheGroupNorm(submodule, orche_config)
                            setattr(module, subname, wrapped_submodule)

    def _should_communicate(self, comm_interval: int | None = None) -> bool:
        if comm_interval is None:
            comm_interval = self.comm_interval
        
        if self.counter <= self.orche_config.warmup_steps:
            return True
        elif (self.counter - self.orche_config.warmup_steps) % comm_interval == 0:
            return True
        else:
            return False

    def _should_barrier(self):
        if self.counter <= self.orche_config.warmup_steps:
            dist.barrier()
        elif (self.counter - self.orche_config.warmup_steps) % self.comm_interval == 0:
            dist.barrier()
        else:
            pass


    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor | float | int,
        encoder_hidden_states: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        timestep_cond: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, any] | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        down_intrablock_additional_residuals: tuple[torch.Tensor] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        record: bool = False,
    ):
        orche_config = self.orche_config
        b, c, h, w = sample.shape
        assert (
            class_labels is None
            and timestep_cond is None
            and attention_mask is None
            and cross_attention_kwargs is None
            and down_block_additional_residuals is None
            and mid_block_additional_residual is None
            and down_intrablock_additional_residuals is None
            and encoder_attention_mask is None
        )

        if orche_config.use_cuda_graph and not record:
            raise NotImplementedError("We have not yet adapted it for CUDA graph technology.")
        else:
            if orche_config.world_size == 1:
                output = self.model(
                    sample,
                    timestep,
                    encoder_hidden_states,
                    class_labels=class_labels,
                    timestep_cond=timestep_cond,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_additional_residuals,
                    mid_block_additional_residual=mid_block_additional_residual,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            elif orche_config.do_classifier_free_guidance and orche_config.split_batch:
                raise NotImplementedError("We have not yet adapted it for batch splitting technology.")
            else:
                output = self.model(
                    sample,
                    timestep,
                    encoder_hidden_states,
                    class_labels=class_labels,
                    timestep_cond=timestep_cond,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_additional_residuals,
                    mid_block_additional_residual=mid_block_additional_residual,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                if self.output_buffer is None: 
                    self.output_buffer = torch.empty((b, c, h, w), device=output.device, dtype=output.dtype) 
                if self.buffer_list is None:
                    self.buffer_list = [torch.empty((b, c, h//self.coordinator.portion_sum*portion, w
                                                    ), device=output.device, 
                                                    dtype=output.dtype) for portion in self.coordinator.portion_assigned
                                        ] 
                output = output.contiguous()

                if self._should_communicate():
                    dist.all_gather(self.buffer_list, output, async_op=False) 
                torch.cat(self.buffer_list, dim=2, out=self.output_buffer)
                assert self.output_buffer.shape[2] == h
                output = self.output_buffer
                
            if record:
                if self.static_inputs is None:
                    self.static_inputs = {
                        "sample": sample,
                        "timestep": timestep,
                        "encoder_hidden_states": encoder_hidden_states,
                        "added_cond_kwargs": added_cond_kwargs
                    }
                self.synchronize()

        if return_dict:
            output = UNet2DConditionOutput(sample=output)
        else:
            output = (output,)

        self.counter += 1
        return output

    @property
    def add_embedding(self):
        return self.model.add_embedding

import time
import torch
from torch import distributed as dist

from .config import OrcheConfig
from .utils import print_by_rank


class OrcheCommManager:
    
    def __init__(self, orche_config: OrcheConfig):
        self.orche_config = orche_config

        self.torch_dtype = None
        self.numel = 0
        self.numel_dict = {}  

        self.buffer_list = None

        self.starts = []
        self.ends = []
        self.shapes = []
        self.types = []

        self.idx_queue = []
        self.handles = None

        self.buffer_numel_list = []
        self.starts_list = []
        self.ends_list = []
        self.shapes_list = []

        self.comm_checkpoint = orche_config.comm_checkpoint

        self.static_send_tensors = []
        self.static_recv_lists = []
        self.comm_stage = 0
        self.comm_block_num = 0


    def register_tensor(
        self, 
        shape: tuple[int, ...] | list[int], 
        torch_dtype: torch.dtype, 
        layer_type: str | None = None
    ) -> int:
        if self.torch_dtype is None:
            self.torch_dtype = torch_dtype
        else:
            assert self.torch_dtype == torch_dtype

        self.starts.append(self.numel)

        # 计算张量元素数量
        numel = 1
        if isinstance(shape, tuple):
            shape = list(shape)
        for dim in shape:
            numel *= dim
        self.numel += numel

        # 统计各层类型的参数量
        if layer_type is not None:
            if layer_type not in self.numel_dict:
                self.numel_dict[layer_type] = 0
            self.numel_dict[layer_type] += numel

        self.ends.append(self.numel)
        self.shapes.append(shape)
        self.types.append(layer_type)

        return len(self.starts) - 1

    def keep_comm_consist(self):
        def shapes_list_to_tensor():
            max_len = max(len(shape) for shape in self.shapes)
            padded_shapes = [shape + [0] * (max_len - len(shape)) for shape in self.shapes]
            return torch.tensor(padded_shapes).to(self.orche_config.device)

        numel_tensor = torch.tensor(self.numel).to(self.orche_config.device)
        starts_tensor = torch.tensor(self.starts).to(self.orche_config.device)
        ends_tensor = torch.tensor(self.ends).to(self.orche_config.device)
        shapes_tensor = shapes_list_to_tensor()

        self.buffer_numel_list = [torch.empty_like(numel_tensor) for _ in range(self.orche_config.n_device_per_batch)]
        self.starts_list = [torch.empty_like(starts_tensor) for _ in range(self.orche_config.n_device_per_batch)]
        self.ends_list = [torch.empty_like(ends_tensor) for _ in range(self.orche_config.n_device_per_batch)]
        self.shapes_list = [torch.empty_like(shapes_tensor) for _ in range(self.orche_config.n_device_per_batch)]

        dist.all_gather(self.buffer_numel_list, numel_tensor, group=self.orche_config.batch_group, async_op=False)
        dist.all_gather(self.starts_list, starts_tensor, group=self.orche_config.batch_group, async_op=False)
        dist.all_gather(self.ends_list, ends_tensor, group=self.orche_config.batch_group, async_op=False)
        dist.all_gather(self.shapes_list, shapes_tensor, group=self.orche_config.batch_group, async_op=False)

        print_by_rank(self.shapes_list)
        shapes_list = []
        for i, shapes in enumerate(self.shapes_list):
            shapes_tuple = [tuple(row[row != 0].tolist()) for row in shapes]
            shapes_list.append(shapes_tuple)

        self.shapes_list = shapes_list

    def create_buffer(self):
        orche_config = self.orche_config
        checkpoint_len = self.comm_checkpoint

        print(
            f"Create buffer with {self.numel / 1e6:.3f}M ({self.numel}) parameters "
            f"(Type:{self.torch_dtype}) for {len(self.starts)} tensors "
            f"on device:{dist.get_rank()}."
        )
        for layer_type, numel in self.numel_dict.items():
            print(f"  {layer_type}: {numel / 1e6:.3f}M ({numel}) parameters")

        self.buffer_list = [
            torch.empty(numel, dtype=self.torch_dtype, device=orche_config.device)
            for numel in self.buffer_numel_list
        ]

        self.static_send_tensors = []
        self.static_recv_lists = []
        self.comm_stage = 0
        self.comm_block_num = len(self.starts) // checkpoint_len

        idx = orche_config.split_idx()
        print_by_rank(0, "=" * 50)
        print_by_rank(0, " " * 18, "Create Buffer")
        print_by_rank(0, "=" * 50)

        for block in range(self.comm_block_num + 1):
            idx_start = block * checkpoint_len
            idx_end = (block + 1) * checkpoint_len - 1 if block < self.comm_block_num else len(self.starts) - 1
            send_view = self.buffer_list[idx][self.starts[idx_start]:self.ends[idx_end]]
            self.static_send_tensors.append(send_view)

            print_by_rank(0, f"Block{block}: {(idx_start, idx_end)}")
            print_by_rank(0, f"Block{block}: need to send {send_view.shape}")
            print_by_rank(0, f"Block{block}: need to recv:")
            recv_list = []
            for i, buffer in enumerate(self.buffer_list):
                start = self.starts_list[i][idx_start]
                end = self.ends_list[i][idx_end]
                recv_list.append(buffer[start:end])
                print_by_rank(0, f"        recv_list[{i}]: {buffer[start:end].shape}")
            self.static_recv_lists.append(recv_list)

        print_by_rank(0, "=" * 50)
        print_by_rank(0, " " * 19, "Complete")
        print_by_rank(0, "=" * 50)

        self.handles = [None] * len(self.starts)

    def get_buffer_list(self, idx: int) -> list[torch.Tensor]:
        buffer_list = []
        for i, buffer in enumerate(self.buffer_list):
            shape = self.shapes_list[i][idx]
            start = self.starts_list[i][idx]
            end = self.ends_list[i][idx]
            buffer_element = buffer[start:end].view(shape)
            buffer_list.append(buffer_element)
        return buffer_list

    def enqueue(self, idx: int, tensor: torch.Tensor):
        orche_config = self.orche_config
        selfrk = orche_config.split_idx()

        if idx == 0 and len(self.idx_queue) > 0:
            t1 = time.time()
            self.communicate_static()
            self.comm_stage = 0

        assert len(self.idx_queue) == 0 or self.idx_queue[-1] == idx - 1

        self.idx_queue.append(idx)
        self.buffer_list[selfrk][self.starts[idx]:self.ends[idx]].copy_(tensor.flatten())

        if len(self.idx_queue) == self.comm_checkpoint:
            self.communicate_static()
            self.comm_stage += 1

    def communicate(self):
        orche_config = self.orche_config
        idx_start = self.idx_queue[0]
        idx_end = self.idx_queue[-1]
        start = self.starts[idx_start]
        end = self.ends[idx_end]
        tensor = self.buffer_list[orche_config.split_idx()][start:end]

        buffer_list = []
        for i, buffer in enumerate(self.buffer_list):
            start = self.starts_list[i][idx_start]
            end = self.ends_list[i][idx_end]
            buffer_element = buffer[start:end]
            buffer_list.append(buffer_element)

        handle = dist.all_gather(buffer_list, tensor, self.orche_config.batch_group, async_op=True)

        for idx in self.idx_queue:
            self.handles[idx] = handle
        self.idx_queue = []

    def communicate_static(self):
        idx_block = self.comm_stage

        tensor = self.static_send_tensors[idx_block]
        buffer_list = self.static_recv_lists[idx_block]

        handle = dist.all_gather(buffer_list, tensor, self.orche_config.batch_group, async_op=True)
        for idx in self.idx_queue:
            self.handles[idx] = handle
        self.idx_queue = []

    def clear(self):
        if len(self.idx_queue) > 0:
            self.communicate_static()
            self.comm_stage = 0

        if self.handles is not None:
            for i in range(len(self.handles)):
                if self.handles[i] is not None:
                    self.handles[i].wait()
                    self.handles[i] = None

class DistriCommManager:
    def __init__(self, distri_config: OrcheConfig):
        self.distri_config = distri_config

        self.torch_dtype = None
        self.numel = 0
        self.numel_dict = {}

        self.buffer_list = None

        self.starts = []
        self.ends = []
        self.shapes = []

        self.idx_queue = []

        self.handles = None

    def register_tensor(
        self, shape: tuple[int, ...] | list[int], torch_dtype: torch.dtype, layer_type: str = None
    ) -> int:
        if self.torch_dtype is None:
            self.torch_dtype = torch_dtype
        else:
            assert self.torch_dtype == torch_dtype
        self.starts.append(self.numel)
        numel = 1
        for dim in shape:
            numel *= dim
        self.numel += numel
        if layer_type is not None:
            if layer_type not in self.numel_dict:
                self.numel_dict[layer_type] = 0
            self.numel_dict[layer_type] += numel

        self.ends.append(self.numel)
        self.shapes.append(shape)
        return len(self.starts) - 1

    def create_buffer(self):
        distri_config = self.distri_config
        if distri_config.rank == 0:
            print(
                f"Create buffer with {self.numel / 1e6:.3f}M parameters for {len(self.starts)} tensors on each device."
            )
            for layer_type, numel in self.numel_dict.items():
                print(f"  {layer_type}: {numel / 1e6:.3f}M parameters")

        self.buffer_list = [
            torch.empty(self.numel, dtype=self.torch_dtype, device=self.distri_config.device)
            for _ in range(self.distri_config.n_device_per_batch)
        ]
        self.handles = [None for _ in range(len(self.starts))]

    def get_buffer_list(self, idx: int) -> list[torch.Tensor]:
        buffer_list = [t[self.starts[idx] : self.ends[idx]].view(self.shapes[idx]) for t in self.buffer_list]
        return buffer_list

    def communicate(self):
        distri_config = self.distri_config
        start = self.starts[self.idx_queue[0]]
        end = self.ends[self.idx_queue[-1]]
        tensor = self.buffer_list[distri_config.split_idx()][start:end]
        
        buffer_list = [t[start:end] for t in self.buffer_list]
        handle = dist.all_gather(buffer_list, tensor, group=self.distri_config.batch_group, async_op=True)
        for i in self.idx_queue:
            self.handles[i] = handle
        self.idx_queue = []

    def enqueue(self, idx: int, tensor: torch.Tensor):
        distri_config = self.distri_config
        if idx == 0 and len(self.idx_queue) > 0:
            self.communicate()
        assert len(self.idx_queue) == 0 or self.idx_queue[-1] == idx - 1
        self.idx_queue.append(idx)
        self.buffer_list[distri_config.split_idx()][self.starts[idx] : self.ends[idx]].copy_(tensor.flatten())

        if len(self.idx_queue) == distri_config.comm_checkpoint:
            self.communicate()

    def clear(self):
        if len(self.idx_queue) > 0:
            self.communicate()
        if self.handles is not None:
            for i in range(len(self.handles)):
                if self.handles[i] is not None:
                    self.handles[i].wait()
                    self.handles[i] = None


def print_by_rank(rank: int, *args):
    if not dist.is_initialized() or \
        (dist.is_initialized() and rank==dist.get_rank()):
        print(f"Rank[{rank}]:", end=" ")
        print(*args)

def print_with_rank(*args):
    if dist.is_initialized():
        print(f"Rank[{dist.get_rank()}]:", end=" ")
        print(*args)
    else:
        print(f"Rank[0]:", end=" ")
import json
import torch
from torch import distributed as dist


class OrcheConfig:

    def __init__(
        self,
        height: int = 1024,
        width: int = 1024,
        do_classifier_free_guidance: bool = True,
        split_batch: bool = False,
        warmup_steps: int = 4,
        comm_checkpoint: int = 60,
        mode: str = "corrected_async_gn",
        parallelism: str = "orche",
        num_inference_steps: int = None,
        use_temporal_adaptation: bool = True,
        use_spatial_adaptation: bool = True,
        use_cuda_graph: bool = True,
        use_deeper_reduce: bool = False,
    ):

        try:
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                dist.init_process_group("nccl")
                rank = dist.get_rank()
                world_size = dist.get_world_size()
        except Exception as e:
            rank = 0
            world_size = 1
            print(f"Failed to initialize process group: {e}, falling back to single GPU")
        
        if not (parallelism == "orche" and not split_batch):
            assert (world_size & (world_size - 1) == 0) and world_size != 0
        self.world_size = world_size
        self.rank = rank
        self.height = height
        self.width = width
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.split_batch = split_batch
        self.warmup_steps = warmup_steps
        self.comm_checkpoint = comm_checkpoint
        self.mode = mode
        # Orchestra
        self.use_temporal_adaptation = use_temporal_adaptation
        self.use_spatial_adaptation = use_spatial_adaptation
        self.use_cuda_graph = use_cuda_graph
        self.use_deeper_reduce = use_deeper_reduce
        self.parallelism = parallelism
        self.num_inference_steps = num_inference_steps

        if do_classifier_free_guidance and split_batch:
            n_device_per_batch = world_size // 2
            if n_device_per_batch == 0:
                n_device_per_batch = 1
        else:
            n_device_per_batch = world_size

        self.n_device_per_batch = n_device_per_batch

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        self.device = device

        batch_group = None
        split_group = None
        if do_classifier_free_guidance and split_batch and world_size >= 2:
            batch_groups = []
            for i in range(2):
                batch_groups.append(dist.new_group(list(range(i * (world_size // 2), (i + 1) * (world_size // 2)))))
            batch_group = batch_groups[self.batch_idx()]
            split_groups = []
            for i in range(world_size // 2):
                split_groups.append(dist.new_group([i, i + world_size // 2]))
            split_group = split_groups[self.split_idx()]
        self.batch_group = batch_group
        self.split_group = split_group


    def batch_idx(self, rank: int or None = None) -> int:
        if rank is None:
            rank = self.rank
        if self.do_classifier_free_guidance and self.split_batch:
            return 1 - int(rank < (self.world_size // 2))
        else:
            return 0 

    def split_idx(self, rank: int | None = None) -> int:
        if rank is None:
            rank = self.rank
        return rank % self.n_device_per_batch


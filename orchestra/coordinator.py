import math
from functools import reduce
import torch
import numpy as np

from .utils import print_by_rank

from .config import OrcheConfig


class OrcheCoordinator:
    def __init__(self, orche_config: OrcheConfig):
        self.orche_config = orche_config
        self.base_num_inference_steps = orche_config.num_inference_steps
        self.use_temporal_adaptation = orche_config.use_temporal_adaptation
        self.use_spatial_adaptation = orche_config.use_spatial_adaptation
        self.use_deeper_reduce = orche_config.use_deeper_reduce

        self.gpu_utils = None
        self.gpu_speed = None 

        self.comm_interval = None  
        self.portion_assigned = None 
        self.portion_sum = None  
        self.all_time_steps = None 
        self.comm_steps_list = None
        
        self.comm_interval_batch0 = None 
        self.comm_interval_batch1 = None
        self.portion_assigned_batch0 = None
        self.portion_assigned_batch1 = None 
        self.portion_sum_batch0 = None  
        self.portion_sum_batch1 = None  
        self.all_time_steps_batch0 = None  
        self.all_time_steps_batch1 = None  

    def set_gpu_speed(self, gpu_speed: torch.Tensor | list):
        if isinstance(gpu_speed, torch.Tensor):
            self.gpu_speed = gpu_speed.tolist()
        else:
            self.gpu_speed = gpu_speed

    def set_preinference_gpu_utilization(self, gpu_utils: list):
        self.gpu_utils = gpu_utils
        print(f"GPU utilization: {gpu_utils}")
        assert all(0 <= rho <= 100 for rho in gpu_utils)

        gpu_speed = []
        for rho in self.gpu_utils:
            if rho == 0:
                gpu_speed.append(1000)
            elif 0 < rho <= 70:
                gpu_speed.append(1000 - rho * 10 * 5 / 7)
            else:
                gpu_speed.append(500)
        self.gpu_speed = gpu_speed

    def _temporal_adaptation_for_group(self, gpu_speed_group: list, reduce_list_group: list | None = None):
        max_v = max(gpu_speed_group)
        larger_comm_interval = 4 if self.use_deeper_reduce else 2
        comm_interval = []
        all_time_steps = []

        if self.use_temporal_adaptation:
            if reduce_list_group is not None:
                assert len(reduce_list_group) == len(gpu_speed_group)
                has_slow_gpu = False
                for i in range(len(gpu_speed_group)):
                    if reduce_list_group[i] == 0:
                        all_time_steps.append(self.base_num_inference_steps)
                        comm_interval.append(larger_comm_interval)
                    else:
                        has_slow_gpu = True
                        all_time_steps.append(self.base_num_inference_steps // larger_comm_interval)
                        comm_interval.append(1)
                if not has_slow_gpu:
                    comm_interval = [1] * len(gpu_speed_group)
                return comm_interval, all_time_steps

            has_slow_gpu = False
            for v in gpu_speed_group:
                if v >= 0.75 * max_v:
                    all_time_steps.append(self.base_num_inference_steps)
                    comm_interval.append(larger_comm_interval)
                else:
                    has_slow_gpu = True
                    all_time_steps.append(self.base_num_inference_steps // larger_comm_interval)
                    comm_interval.append(1)
            if not has_slow_gpu:
                comm_interval = [1] * len(gpu_speed_group)
        else:
            all_time_steps = [self.base_num_inference_steps] * len(gpu_speed_group)
            comm_interval = [1] * len(gpu_speed_group)

        return comm_interval, all_time_steps

    def temporal_adaptation(self, reduce_list=None):
        gpu_speed = self.gpu_speed
        assert gpu_speed is not None

        if (self.orche_config.split_batch and 
            self.orche_config.do_classifier_free_guidance and 
            self.orche_config.world_size >= 2):
            raise NotImplementedError("We have not yet adapted it for batch splitting technology.")
        else:
            comm_interval, all_time_steps = self._temporal_adaptation_for_group(gpu_speed, reduce_list)
            self.all_time_steps = all_time_steps
            self.comm_interval = comm_interval

    def _spatial_adaptation_for_group(self, gpu_speed_group: list, 
                                      all_time_steps_group: list, 
                                      patches_group: list | None = None):

        patch_space = 32
        if self.use_spatial_adaptation:
            if patches_group is not None:
                assert len(patches_group) == len(gpu_speed_group)
                gcd = reduce(math.gcd, patches_group)
                patches_group = [P // gcd for P in patches_group]
                portion_sum = sum(patches_group)
                return patches_group, portion_sum

            weighted_sum = sum(v / M for v, M in zip(gpu_speed_group, all_time_steps_group))
            patch_size_list = []
            for i in range(len(gpu_speed_group)):
                patch_size = int(((gpu_speed_group[i] / all_time_steps_group[i]) / weighted_sum) * patch_space)
                patch_size_list.append(patch_size)
        else:
            patch_size_list = [patch_space // len(gpu_speed_group)] * len(gpu_speed_group)

        while patch_space - sum(patch_size_list) != 0:
            rest = patch_space - sum(patch_size_list)
            rd_id = rest % len(patch_size_list)
            patch_size_list[rd_id] += 1

        gcd = reduce(math.gcd, patch_size_list)
        patch_size_list = [P // gcd for P in patch_size_list]
        patch_space = patch_space // gcd

        return patch_size_list, patch_space

    def spatial_adaptation(self, patches=None):
        gpu_speed = self.gpu_speed
        all_time_steps = self.all_time_steps
        assert gpu_speed is not None
        assert all_time_steps is not None


        if (self.orche_config.split_batch and 
            self.orche_config.do_classifier_free_guidance and 
            self.orche_config.world_size >= 2):
            raise NotImplementedError("We have not yet adapted it for batch splitting technology.")
        else:
            portion_assigned, portion_sum = self._spatial_adaptation_for_group(
                gpu_speed, all_time_steps, patches
            )
            self.portion_sum = portion_sum
            self.portion_assigned = portion_assigned

    def get_comm_interval_for_batch(self, batch_idx: int | None = None):
        if (self.orche_config.split_batch and 
            self.orche_config.do_classifier_free_guidance and 
            self.orche_config.world_size >= 2):
            raise NotImplementedError("We have not yet adapted it for batch splitting technology.")
        else:
            return self.comm_interval

    def get_portion_assigned_for_batch(self, batch_idx: int | None = None):
        if (self.orche_config.split_batch and 
            self.orche_config.do_classifier_free_guidance and 
            self.orche_config.world_size >= 2):
            raise NotImplementedError("We have not yet adapted it for batch splitting technology.")
        else:
            return self.portion_assigned

    def get_portion_sum_for_batch(self, batch_idx: int | None = None):
        if (self.orche_config.split_batch and 
            self.orche_config.do_classifier_free_guidance and 
            self.orche_config.world_size >= 2):
            raise NotImplementedError("We have not yet adapted it for batch splitting technology.")
        else:
            return self.portion_sum

    def get_sync_steps(self):
        if self.comm_steps_list == None:
            warmup_steps = self.orche_config.warmup_steps
            base_steps = self.base_num_inference_steps
            warmup_offset = (base_steps - warmup_steps - 1) % 2
            comm_steps_list = []
            for count, interval in enumerate(self.comm_interval):
                comm_steps = []
                if interval == 1:
                    compressed_steps_num = (base_steps - warmup_steps - 1 - warmup_offset)/2
                    
                    warmup_timesteps = np.arange(0, warmup_steps + 1)
                    compressed_timesteps = np.arange(warmup_steps + 1, warmup_steps + 1 + compressed_steps_num)
                    offset_timesteps = np.arange(warmup_steps + 1 + compressed_steps_num, warmup_steps + 1 + compressed_steps_num + warmup_offset)

                if interval == 2: 
                    compressed_steps_num = (base_steps - warmup_steps - 1 - warmup_offset)
                    warmup_timesteps = np.arange(0, warmup_steps + 1)
                    compressed_timesteps = np.arange(warmup_steps + 2, warmup_steps + 1 + compressed_steps_num, 2)
                    offset_timesteps = np.arange(warmup_steps + 1 + compressed_steps_num, warmup_steps + 1 + compressed_steps_num + warmup_offset)
                
                comm_steps.extend([int(s) for s in list(warmup_timesteps)])
                comm_steps.extend([int(s) for s in list(compressed_timesteps)])
                comm_steps.extend([int(s) for s in list(offset_timesteps)])
                comm_steps_list.append(comm_steps)
            self.comm_steps_list = comm_steps_list
            print_by_rank(0, comm_steps_list)
            return comm_steps_list
        else:
            return self.comm_steps_list
    
    def print_info(self):
        print("*" * 10, "Coordinator Info", "*" * 10)
        print("*" * 2, f"gpu_utils: \t{self.gpu_utils}")
        print("*" * 2, f"gpu_speed: \t{self.gpu_speed}")
        
        if (self.orche_config.split_batch and 
            self.orche_config.do_classifier_free_guidance and 
            self.orche_config.world_size >= 2):
            raise NotImplementedError("We have not yet adapted it for batch splitting technology.")
        else:
            print("*" * 2, f"Steps_num: \t{self.all_time_steps}")
            print("*" * 2, f"Patch_size: \t{self.portion_assigned}")
            print("*" * 2, f"Interval: \t{self.comm_interval}")
            if self.portion_sum is not None:
                print("*" * 2, f"Portion_sum: \t{self.portion_sum}")
        print("*" * 38)


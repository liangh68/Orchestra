from torch import nn

from ..config import OrcheConfig
from ..communication import OrcheCommManager
from ..coordinator import OrcheCoordinator

class BaseModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        orche_config: OrcheConfig,
    ):
        super(BaseModule, self).__init__()
        self.module = module
        self.orche_config = orche_config
        self.comm_manager = None
        self.buffer_list = None
        self.idx = None
        self.rank = orche_config.split_idx()

        self.coordinator = None
        self.counter = 0
        self.comm_interval = None


    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_counter(self, counter: int = 0):
        self.counter = counter

    def set_comm_manager(self, comm_manager: OrcheCommManager):
        self.comm_manager = comm_manager
        
    def set_coordinator(self, coordinator: OrcheCoordinator):
        self.coordinator = coordinator
        if (coordinator.orche_config.split_batch and 
            coordinator.orche_config.do_classifier_free_guidance and 
            coordinator.orche_config.world_size >= 2):
            batch_idx = coordinator.orche_config.batch_idx()
            comm_interval_list = coordinator.get_comm_interval_for_batch(batch_idx)
            batch_group_rank = coordinator.orche_config.split_idx()
            self.comm_interval = comm_interval_list[batch_group_rank]
        else:
            self.comm_interval = coordinator.comm_interval[self.rank]

class DistriBaseModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        distri_config: OrcheConfig,
    ):
        super(DistriBaseModule, self).__init__()
        self.module = module
        self.distri_config = distri_config
        self.comm_manager = None

        self.counter = 0

        self.buffer_list = None
        self.idx = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_counter(self, counter: int = 0):
        self.counter = counter

    def set_comm_manager(self, comm_manager):
        self.comm_manager = comm_manager
from diffusers import ConfigMixin, ModelMixin
from torch import nn

from ..modules.base_module import BaseModule, DistriBaseModule
from ..config import OrcheConfig
from ..communication import OrcheCommManager, DistriCommManager
from ..coordinator import OrcheCoordinator

class BaseModel(ModelMixin, ConfigMixin):
    def __init__(
            self, 
            model: nn.Module, 
            orche_config: OrcheConfig):
        super(BaseModel, self).__init__()
        self.model = model
        self.orche_config = orche_config
        self.comm_manager = None

        self.buffer_list = None
        self.output_buffer = None

        # about coordinator
        self.counter = 0
        self.coordinator = None
        self.comm_interval = None

        # for cuda graph
        self.static_inputs = None
        self.static_outputs = None
        self.rank = orche_config.split_idx()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_counter(self, counter: int = 0):
        self.counter = counter
        for module in self.model.modules():
            if isinstance(module, BaseModule):
                module.set_counter(counter)

    def set_comm_manager(self, comm_manager: OrcheCommManager):
        self.comm_manager = comm_manager
        for module in self.model.modules():
            if isinstance(module, BaseModule):
                module.set_comm_manager(comm_manager)
        
    def set_coordinator(self, coordinator: OrcheCoordinator):
        self.coordinator = coordinator
        self.comm_interval = coordinator.comm_interval[self.rank]
        for module in self.model.modules():
            if isinstance(module, BaseModule):
                module.set_coordinator(coordinator)

    def setup_cuda_graph(self, static_outputs, cuda_graphs):
        self.static_outputs = static_outputs
        self.cuda_graphs = cuda_graphs
        
    @property
    def config(self):
        return self.model.config

    def synchronize(self):
        if self.comm_manager is not None and self.comm_manager.handles is not None:
            for i in range(len(self.comm_manager.handles)):
                if self.comm_manager.handles[i] is not None:
                    self.comm_manager.handles[i].wait()
                    self.comm_manager.handles[i] = None

    # for debug
    def print_all_basemodel(self):
        base_list = []
        for name, module in self.model.named_modules():
            if isinstance(module, BaseModule):
                base_list.append(name)

    
class DistriBaseModel(ModelMixin, ConfigMixin):
    def __init__(self, model: nn.Module, distri_config: OrcheConfig):
        super(DistriBaseModel, self).__init__()
        self.model = model
        self.distri_config = distri_config
        self.comm_manager = None

        self.buffer_list = None
        self.output_buffer = None
        self.counter = 0

        # for cuda graph
        self.static_inputs = None
        self.static_outputs = None
        self.cuda_graphs = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_counter(self, counter: int = 0):
        self.counter = counter
        for module in self.model.modules():
            if isinstance(module, DistriBaseModule):
                module.set_counter(counter)

    def set_comm_manager(self, comm_manager: DistriCommManager):
        self.comm_manager = comm_manager
        for module in self.model.modules():
            if isinstance(module, DistriBaseModule):
                module.set_comm_manager(comm_manager)

    def setup_cuda_graph(self, static_outputs, cuda_graphs):
        self.static_outputs = static_outputs
        self.cuda_graphs = cuda_graphs

    @property
    def config(self):
        return self.model.config

    def synchronize(self):
        if self.comm_manager is not None and self.comm_manager.handles is not None:
            for i in range(len(self.comm_manager.handles)):
                if self.comm_manager.handles[i] is not None:
                    self.comm_manager.handles[i].wait()
                    self.comm_manager.handles[i] = None

        # for debug
    def print_all_basemodel(self):
        base_list = []
        for name, module in self.model.named_modules():
            if isinstance(module, BaseModule):
                base_list.append(name)

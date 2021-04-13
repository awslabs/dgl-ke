from collections import OrderedDict
from torch import nn

class Module(nn.Module):
    def __init__(self,
                 module_name):
        super(Module, self).__init__()
        self.module_name = module_name

    def set_training_params(self, args):
        raise NotImplementedError

    def set_test_params(self, args):
        raise NotImplementedError

    @property
    def name(self):
        return self.module_name

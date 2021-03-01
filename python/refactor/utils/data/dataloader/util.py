# we use aggregation to aggregate sample data into DataWrapper. So that encoder&decoder can
# have different behavior with these args defined.
# one benifit of using DataWrapper instead of passing additional arguments directly to encoder&decoder
# is that, we do not need to provide encoder(), decode() with other args that is different from case to case
# so that the encode, decode method is eazier to extend.
# we use adapter Design Pattern here.
from collections import defaultdict
from copy import deepcopy

import torch as th

class SamplerCombiner(object):
    def __init__(self, dataloaders, training_args:dict):
        self.dataloaders = dataloaders
        self.args = training_args

    def __iter__(self):
        self.iter = [iter(dataloader) for dataloader in self.dataloaders]
        return self

    def __next__(self):
        data = {}
        data.update(self.args)
        for iterator in self.iter:
            data.update(next(iterator))
        return data


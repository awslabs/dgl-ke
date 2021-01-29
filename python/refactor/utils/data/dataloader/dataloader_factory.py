from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler
from .dataloader_wrapper import DataLoaderWrapper
from functools import partial

class DataLoaderFactory:
    def __init__(self,
                 batch_size=64,
                 num_workers=0,
                 pin_memory=True,
                 drop_last=False,
                 shuffle=False):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.datasets = []
        self.samplers = []
        self.wrappers = [DataLoaderWrapper()]

    def append_dataset_sampler(self, dataset, sampler: Sampler = None):
        # dataset should be be attached together with graph
        self.datasets += [dataset]
        self.samplers += [sampler]

    def append_dataset_wrapper(self, wrapper: DataLoaderWrapper):
        self.wrappers += [wrapper]

    def create_dataloader(self, rank=0, world_size=1):
        curr_wrapper = []
        for dataset, sampler, wrapper in zip(self.datasets, self.samplers, self.wrappers):
            subdataset = dataset.partition(rank, world_size)
            sub_sampler = sampler(subdataset) if sampler is not None else None
            dataloader = DataLoader(dataset=subdataset,
                                    batch_size=self.batch_size,
                                    sampler=sub_sampler,
                                    shuffle=False if sub_sampler is not None else self.shuffle,
                                    num_workers=self.num_workers,
                                    pin_memory=self.pin_memory,
                                    drop_last=self.drop_last)
            curr_wrapper += [dataloader]

        for wrapper in self.wrappers:
            wrapper.set_dataloader(curr_wrapper)
            curr_wrapper = wrapper
        return curr_wrapper




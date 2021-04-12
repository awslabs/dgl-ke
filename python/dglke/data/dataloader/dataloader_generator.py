import torch as th
from torch.utils.data.dataloader import DataLoader

class CustomizeDataLoaderGenerator:
    def __init__(self,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 pin_memory=True,
                 drop_last=True,
                 timeout=0,
                 worker_init_fn=None,
                 multiprocessing_context=None,
                 generator=None,
                 prefetch_factor=2,
                 persistent_workers=False,
                 num_workers=0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.generator = generator
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers

    def generate_dataloader(self, dataset):
        raise NotImplementedError


class KGETrainDataLoaderGenerator(CustomizeDataLoaderGenerator):
    def __init__(self,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 pin_memory=True,
                 drop_last=True,
                 timeout=0,
                 worker_init_fn=None,
                 multiprocessing_context=None,
                 generator=None,
                 prefetch_factor=2,
                 persistent_workers=False,
                 num_workers=0,
                 metadata=dict()):
        super(KGETrainDataLoaderGenerator, self).__init__(batch_size,
                                                shuffle,
                                                sampler,
                                                batch_sampler,
                                                pin_memory,
                                                drop_last,
                                                timeout,
                                                worker_init_fn,
                                                multiprocessing_context,
                                                generator,
                                                prefetch_factor,
                                                persistent_workers,
                                                num_workers)
        self.metadata = metadata

    def generate_dataloader(self, dataset):
        batch_size = 1 if isinstance(dataset, th.utils.data.IterableDataset) else self.batch_size
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=self.shuffle,
                                sampler=self.sampler,
                                batch_sampler=self.batch_sampler,
                                pin_memory=self.pin_memory,
                                drop_last=self.drop_last,
                                timeout=self.timeout,
                                worker_init_fn=self.worker_init_fn,
                                multiprocessing_context=self.multiprocessing_context,
                                generator=self.generator,
                                prefetch_factor=self.prefetch_factor,
                                persistent_workers=self.persistent_workers,
                                num_workers=self.num_workers)
        return TrainDataLoader(dataloader, self.metadata)

class KGEEvalDataLoaderGenerator(CustomizeDataLoaderGenerator):
    def __init__(self,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 pin_memory=True,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,
                 multiprocessing_context=None,
                 generator=None,
                 prefetch_factor=2,
                 persistent_workers=False,
                 num_workers=0,
                 metadata=dict()):
        super(KGEEvalDataLoaderGenerator, self).__init__(batch_size,
                                                shuffle,
                                                sampler,
                                                batch_sampler,
                                                pin_memory,
                                                drop_last,
                                                timeout,
                                                worker_init_fn,
                                                multiprocessing_context,
                                                generator,
                                                prefetch_factor,
                                                persistent_workers,
                                                num_workers)
        self.metadata = metadata

    def generate_dataloader(self, dataset):
        batch_size = 1 if isinstance(dataset, th.utils.data.IterableDataset) else self.batch_size
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=self.shuffle,
                                sampler=self.sampler,
                                batch_sampler=self.batch_sampler,
                                pin_memory=self.pin_memory,
                                drop_last=self.drop_last,
                                timeout=self.timeout,
                                worker_init_fn=self.worker_init_fn,
                                multiprocessing_context=self.multiprocessing_context,
                                generator=self.generator,
                                prefetch_factor=self.prefetch_factor,
                                persistent_workers=self.persistent_workers,
                                num_workers=self.num_workers)
        return EvalDataLoader(dataloader, self.metadata)

# wrapper for dataloader to pass metadata into arguments
class TrainDataLoader:
    def __init__(self,
                 dataloader,
                 train_args=dict()):
        self.dataloader = dataloader
        self.data = train_args

    def __iter__(self):
        self.iter_data = iter(self.dataloader)
        return self

    def __next__(self):
        self.data['neg_type'] = 'head' if ('neg_type' not in self.data or self.data['neg_type'] == 'tail') else 'tail'
        head, rel, tail, edge_impt, neg = next(self.iter_data)
        self.data.update({'head': head,
                          'rel': rel,
                          'tail': tail,
                          'neg': neg,
                          'edge_impt': edge_impt})
        return self.data

class EvalDataLoader:
    def __init__(self, dataloader, eval_args):
        self.data = eval_args
        self.data['neg'] = th.arange(eval_args['num_nodes'])
        self.dataloader = dataloader

    def __iter__(self):
        self.iter_data = iter(self.dataloader)
        return self

    def __next__(self):
        self.data['neg_type'] = 'head_tail'
        self.data.update(next(self.iter_data))
        return self.data

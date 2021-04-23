import copy
import torch as th
from torch.utils.data.dataloader import DataLoader

class CustomizeDataLoaderGenerator:
    """ dataloader generator base class that generates dataloader on the fly during fit&test.

    This dataloader generator wraps all the necessary args for dataloader. As we might have multi-process
    training/evaluation, the CustomizeDataLoaderGenerator automatically handle this situation and generate
    partition dataloader for each subprocess.
    """
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
        """ Generate dataloader on the fly for each subprocess and wrap metadata with KGETrainDataLoader

        Parameters
        ----------
        dataset: torch.utils.data.dataset
            partitioned dataset

        Returns
        -------
        KGETrainDataLoader
            Iterable Dataloader that contains training samples and metadata.

        """
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
        return KGETrainDataLoader(dataloader, self.metadata)

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
        """ Generate dataloader on the fly for each subprocess and wrap metadata with KGEEvalDataLoader
        Parameters
        ----------
        dataset: torch.utils.data.dataset
            partitioned dataset

        Returns
        -------
        KGETrainDataLoader
            Iterable Dataloader that contains evaluation samples and metadata.

        """
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
        return KGEEvalDataLoader(dataloader, self.metadata)

class KGETrainDataLoader:
    """ Train Dataloader that wraps torch.utils.data.dataloader and metadata

    TrainDataloader is an iterable that outputs samples just as torch.utils.data.dataloader. Besides,
    it provide additional metadata of the training dataset like neg_type, chunk_size, etc.
    """
    def __init__(self,
                 dataloader,
                 train_args=dict()):
        self.dataloader = dataloader
        self.args = train_args

    def __iter__(self):
        self.iter_data = iter(self.dataloader)
        self.data = dict()
        self.data.update(self.args)
        return self

    def __next__(self):
        self.data['neg_type'] = 'head' if ('neg_type' not in self.data or self.data['neg_type'] == 'tail') else 'tail'
        head, rel, tail, edge_impt, neg = next(self.iter_data)
        head, rel, tail, edge_impt, neg = head.view(-1), rel.view(-1), tail.view(-1), edge_impt.view(-1), neg.view(-1)
        self.data.update({'head': head,
                          'rel': rel,
                          'tail': tail,
                          'neg': neg,
                          'edge_impt': edge_impt})
        return self.data

class KGEEvalDataLoader:
    """ Eval Dataloader that wraps torch.utils.data.dataloader and metadata

    KGEEValDataLoader is an iterable that outputs samples just as torch.utils.data.dataloader. Besides,
    it provide additional metadata of the eval dataset like neg_type, chunk_size, etc. For evaluation, the negative sample
    are all entities of graph. It's stored as attributes to prevent relentless resample.
    """
    def __init__(self, dataloader, eval_args):
        self.args = eval_args
        self.neg = th.arange(eval_args['num_nodes'])
        self.dataloader = dataloader

    def __iter__(self):
        self.iter_data = iter(self.dataloader)
        self.data = dict()
        self.data.update(self.args)
        self.data['neg'] = self.neg
        return self

    def __next__(self):
        self.data['neg_type'] = 'head_tail'
        self.data.update(next(self.iter_data))
        return self.data

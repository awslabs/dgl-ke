# we use aggregation to aggregate sample data into DataWrapper. So that encoder&decoder can
# have different behavior with these args defined.
# one benifit of using DataWrapper instead of passing additional arguments directly to encoder&decoder
# is that, we do not need to provide encoder(), decode() with other args that is different from case to case
# so that the encode, decode method is eazier to extend.
# we use adapter Design Pattern here.
from collections import defaultdict
import torch as th

class DataLoaderWrapper:
    def __init__(self):
        pass

    def set_dataloader(self, dataloader):
        self._dataloader = dataloader

    def __next__(self):
        ret = {}
        for iter_data in self._iter_data:
            ret.update(next(iter_data))
        return ret

    def __iter__(self):
        self._iter_data = [iter(dataloader) for dataloader in self._dataloader]
        return self


class LCWADataLoaderWrapper(DataLoaderWrapper):
    def __init__(self,
                 num_of_nodes,
                 lcwa=True):
        self.__defaults = defaultdict(lcwa=lcwa, tails=th.arange(num_of_nodes))
        super(LCWADataLoaderWrapper, self).__init__()

    def __next__(self):
        ret = next(self._iter_data)
        ret.update(self.__defaults)
        return ret

    def __iter__(self):
        self._iter_data = iter(self._dataloader)
        return self



class NegChunkDataLoaderWrapper(DataLoaderWrapper):
    def __init__(self,
                 chunk_size=1,
                 neg_sample_size=1,
                 self_neg=False):
        self.__defaults = defaultdict(chunk_size=chunk_size,
                        neg_sample_size=neg_sample_size,
                        self_neg=self_neg)
        super(NegChunkDataLoaderWrapper, self).__init__()

    def __next__(self):
        ret = next(self._iter_data)
        ret.update(self.__defaults)
        return ret

    def __iter__(self):
        self._iter_data = iter(self._dataloader)
        return self


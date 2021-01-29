import torch as th
from .module import Module

class Decoder(Module):
    def __init__(self, decoder_name):
        super(Decoder, self).__init__(decoder_name)

    def decode(self, encoded_data, data):
        pass

class DecoderList(Decoder):
    def __init__(self):
        super(DecoderList, self).__init__('decoder_list')
        self.decoder = {}
        self.decoder_index = {}

    def add_module(self, name, module):
        self.decoder[name] = module
        self.decoder_index[name] = len(self.idx)

    def save(self, save_path: str):
        for decoder in self.decoder.items():
            decoder.save(save_path)

    def load(self, load_path: str):
        for decoder in self.decoder.items():
            decoder.load(load_path)

    def share_memory(self):
        for decoder in self.decoder.items():
            decoder.share_memory()

    def dense_parameters(self) -> list:
        dense_parameters = []
        for decoder in self.decoder.items():
            dense_parameters += [decoder.dense_parameters()]
        return dense_parameters

    def sparse_parameters(self) -> list:
        sparse_parameters = []
        for decoder in self.decoder.items():
            sparse_parameters += [decoder.sparse_parameters()]
        return sparse_parameters

    def prepare_distributed_training(self, gpu_id, rank=0, world_size=-1):
        for decoder in self.decoder.items():
            decoder.prepare_distributed_training(gpu_id, rank, world_size)

    def get_loss(self, results: list):
        loss = {}
        for decoder in self.decoder.items():
            loss.update(decoder.get_loss(results))
        return loss

    def decode(self, encoded_data, data, idx=None):
        ret_val = []
        for name, decoder in self.decoder.items():
            idx = self.decoder_index[name]
            ret_val += [decoder.decode(encoded_data, data, idx)]
        return ret_val

    def evaluate(self, results: list):
        for result, decoder in zip(results, self.decoder.values()):
            decoder.decode(result)

class KGEDecoder(Decoder):
    def __init__(self, decoder_name):
        super(KGEDecoder, self).__init__(decoder_name)

    def attach_score_func(self, score_func):
        self._score_func = score_func

    def prepare_distributed_training(self, gpu_id, rank=0, world_size=-1):
        pass

    def decode(self, encoded_data, data):
        if 'lcwa' in data.keys() and data['lcwa'] is True:
            # head: b x h, rel: b x h, tail: num_nodes x h
            head, rel, tail = encoded_data
            score = self._score_func.score_hr(head, rel, tail)
            return [score]
        elif 'chunk_size' in data.keys():
            head, rel, tail, neg = encoded_data
            chunk_size, neg_sample_size = data['chunk_size'], data['neg_sample_size']
            pos_score = self._score_func.predict(head, rel, tail)
            if data['neg_type'] == 'head':
                neg_func = self._score_func.create_neg(True)
                neg_score = neg_func(neg, rel, tail, chunk_size, neg_sample_size)
            else:
                neg_func = self._score_func.create_neg(False)
                neg_score = neg_func(head, rel, neg, chunk_size, neg_sample_size)
            return [pos_score, neg_score]



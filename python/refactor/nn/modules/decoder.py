import torch as th
from .module import Module


class Decoder(Module):
    def __init__(self, decoder_name):
        super(Decoder, self).__init__(decoder_name)

    def forward(self, encoded_data, data, gpu_id):
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

    def prepare_model(self, gpu_id, rank=0, world_size=-1):
        for decoder in self.decoder.items():
            decoder.prepare_model(gpu_id, rank, world_size)

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

    def prepare_model(self, gpu_id, rank=0, world_size=-1):
        pass

    def save(self, save_path: str):
        pass

    def load(self, load_path: str):
        pass

    def share_memory(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def update(self, gpu_id):
        pass

    def set_training_params(self, args):
        pass

    def set_test_params(self, args):
        pass

    def sync_model(self, gpu_id, rank, world_size):
        pass

    def postprocess_model(self, gpu_id, rank, world_size):
        pass

    def forward(self, encoded_data, data, gpu_id):
        head, rel, tail, neg = encoded_data['head'], encoded_data['rel'], encoded_data['tail'], encoded_data['neg']
        chunk_size, neg_sample_size = data['chunk_size'], data['neg_sample_size']
        pos_score = self._score_func.predict(head, rel, tail)
        if data['neg_type'] == 'head':
            neg_func = self._score_func.create_neg(True)
            neg_score = neg_func(neg, rel, tail, chunk_size, neg_sample_size)
            return {'pos_score': pos_score,
                    'neg_score': neg_score}
        elif data['neg_type'] == 'tail':
            neg_func = self._score_func.create_neg(False)
            neg_score = neg_func(head, rel, neg, chunk_size, neg_sample_size)
            return {'pos_score': pos_score,
                    'neg_score': neg_score}
        else:
            neg_func_head = self._score_func.create_neg(True)
            neg_score_head = neg_func_head(neg, rel, tail, chunk_size, neg_sample_size)
            neg_func_tail = self._score_func.create_neg(False)
            neg_score_tail = neg_func_tail(head, rel, neg, chunk_size, neg_sample_size)
            return {'pos_score': pos_score,
                    'neg_score_head': neg_score_head,
                    'neg_score_tail': neg_score_tail}



    def evaluate(self, results, data, graph):
        return self.metrics_evaluator.evaluate(results, data, graph)

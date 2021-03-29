import torch as th
from .module import Module

class KGEDecoder(Module):
    def __init__(self, decoder_name):
        super(KGEDecoder, self).__init__(decoder_name)

    def attach_score_func(self, score_func):
        self._score_func = score_func

    def set_training_params(self, args):
        pass

    def set_test_params(self, args):
        pass

    def forward(self, encoded_data, data, gpu_id):
        head, rel, tail, neg = encoded_data['head'], encoded_data['rel'], encoded_data['tail'], encoded_data['neg']
        pos_score = self._score_func.predict(head, rel, tail)
        chunk_size, neg_sample_size = data['chunk_size'], data['neg_sample_size']
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

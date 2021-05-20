import torch as th
from .module import Module
from .score_fun import ATTHScore
import torch.nn as nn
from dglke.nn.loss.loss_generator import LossGenerator
from dglke.nn.metrics import MetricsEvaluator
from dglke.nn.modules import TransEScore, TransRScore, DistMultScore, ComplExScore, RESCALScore, RotatEScore, SimplEScore

class BaseDecoder(Module):
    def __init__(self,
                 decoder_name,
                 metrics_evaluator=MetricsEvaluator(),
                 loss_generator=LossGenerator()):
        super(BaseDecoder, self).__init__(decoder_name)
        self.metrics_evaluator = metrics_evaluator
        self.loss_generator = loss_generator

    def set_training_params(self, args):
        raise NotImplementedError

    def set_test_params(self, args):
        raise NotImplementedError

    def evaluate(self, results: list, data, graph):
        raise NotImplementedError

    def forward(self, encoded_data, data, gpu_id):
        raise NotImplementedError

    def infer(self, encoded_data, data, gpu_id):
        raise NotImplementedError

    def get_loss(self, results):
        return self.loss_generator.get_total_loss(results)


class KGEDecoder(BaseDecoder):
    """ General KGE Decoder

    The KGE Decoder supports different type of embedding methods including TransE_l1, TransE_l2, TransR, RESCAL,
    DistMult, ComplEx, RotatE, SimplE. By subsituting score_func, user can define their own embedding method.
    """
    def __init__(self,
                 decoder_name,
                 score_func=None,
                 loss_gen=LossGenerator(),
                 metrics_evaluator=MetricsEvaluator(),
                 ):
        super(KGEDecoder, self).__init__(decoder_name,
                                         metrics_evaluator,
                                         loss_gen)
        self._score_func = score_func

    def set_training_params(self, args):
        pass

    def set_test_params(self, args):
        pass

    def forward(self, encoded_data, data, gpu_id):
        """ calculate positive scores and/or negative scores for KGE

        Parameters
        ----------
        encoded_data: dict
            dict of embeddings sliced by sample indicies of training data
        data: dict
            dict of training data containing head, rel, tail, neg indicies and metadata
        gpu_id: int
            which gpu to perform this calculation

        Returns
        -------
        dict of tensors
            positive_score,  (neg_score/neg_score_head, neg_score_tail)
        """
        assert 'head' in encoded_data.keys() and \
               'rel' in encoded_data.keys() and \
               'tail' in encoded_data.keys(), \
               "encoded data should contain keys 'head', 'rel', 'tail' and 'neg'."

        head, rel, tail = encoded_data['head'], encoded_data['rel'], encoded_data['tail']

        pos_score = self._score_func.predict(head, rel, tail)

        if 'neg_type' not in data.keys():
            return {'pos_score': pos_score}
        else:
            chunk_size, neg_sample_size = data['chunk_size'], data['neg_sample_size']
            neg = encoded_data['neg']
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
            elif data['neg_type'] == 'head_tail' :
                neg_func_head = self._score_func.create_neg(True)
                neg_score_head = neg_func_head(neg, rel, tail, chunk_size, neg_sample_size)
                neg_func_tail = self._score_func.create_neg(False)
                neg_score_tail = neg_func_tail(head, rel, neg, chunk_size, neg_sample_size)
                return {'pos_score': pos_score,
                        'neg_score_head': neg_score_head,
                        'neg_score_tail': neg_score_tail}
            else:
                raise ValueError("{} is not correct, choose from head, tail, both.".format(data['neg_type']))

    def infer(self, head_emb, rel_emb, tail_emb):
        pass

    def evaluate(self, results, data, graph):
        return self.metrics_evaluator.evaluate(results, data, graph)

class TransRDecoder(BaseDecoder):
    """TransR Decoder
    """
    def __init__(self,
                 score_func=None,
                 loss_gen=LossGenerator(),
                 metrics_evaluator=MetricsEvaluator(),
                 ):

        decoder_name = 'TransRDecoder'
        super(TransRDecoder, self).__init__(decoder_name,
                                         metrics_evaluator,
                                         loss_gen)
        self._score_func = score_func

    def set_training_params(self, args):
        pass

    def set_test_params(self, args):
        pass

    def forward(self, encoded_data, data, gpu_id):
        """ calculate positive scores and/or negative scores for KGE

        Parameters
        ----------
        encoded_data: dict
            dict of embeddings sliced by sample indicies of training data
        data: dict
            dict of training data containing head, rel, tail, neg indicies and metadata
        gpu_id: int
            which gpu to perform this calculation

        Returns
        -------
        dict of tensors
            positive_score,  (neg_score/neg_score_head, neg_score_tail)
        """
        assert 'head' in encoded_data.keys() and \
               'rel' in encoded_data.keys() and \
               'tail' in encoded_data.keys() and \
               'rel_id' in encoded_data.keys(), \
               "encoded data should contain keys 'head', 'rel', 'tail' and 'rel_id'."

        head, rel, tail, rel_id = encoded_data['head'], encoded_data['rel'], encoded_data['tail'], encoded_data['rel_id']
        
        pos_score = self._score_func.predict(head, rel, tail, rel_id)

        if 'neg_type' not in data.keys():
            return {'pos_score': pos_score}
        else:
            chunk_size, neg_sample_size = data['chunk_size'], data['neg_sample_size']
            neg = encoded_data['neg']
            if data['neg_type'] == 'head':
                neg_func = self._score_func.create_neg(True)
                neg_score = neg_func(neg, rel, tail, rel_id, chunk_size, neg_sample_size)
                return {'pos_score': pos_score,
                        'neg_score': neg_score}
            elif data['neg_type'] == 'tail':
                neg_func = self._score_func.create_neg(False)
                neg_score = neg_func(neg, rel, tail, rel_id, chunk_size, neg_sample_size)
                return {'pos_score': pos_score,
                        'neg_score': neg_score}
            elif data['neg_type'] == 'head_tail' :
                neg_func_head = self._score_func.create_neg(True)
                neg_score_head = neg_func_head(neg, rel, tail, rel_id, chunk_size, neg_sample_size)
                neg_func_tail = self._score_func.create_neg(False)
                neg_score_tail = neg_func_tail(head, rel, neg, rel_id, chunk_size, neg_sample_size)
                return {'pos_score': pos_score,
                        'neg_score_head': neg_score_head,
                        'neg_score_tail': neg_score_tail}
            else:
                raise ValueError("{} is not correct, choose from head, tail, both.".format(data['neg_type']))

    def infer(self, head_emb, rel_emb, tail_emb):
        pass

    def evaluate(self, results, data, graph):
        return self.metrics_evaluator.evaluate(results, data, graph)

class AttHDecoder(BaseDecoder):
    """ Low-Dimensional Hyperbolic Knowledge Graph Embeddings decoder

    paper: https://arxiv.org/pdf/2005.00545.pdf
    """
    def __init__(self,
                 decoder_name='AttH',
                 metrics_evaluator=MetricsEvaluator(),
                 loss_gen=LossGenerator()):
        super(AttHDecoder, self).__init__(decoder_name,
                                          metrics_evaluator,
                                          loss_gen)
        self._score_func = ATTHScore()

    def forward(self, encoded_data, data, gpu_id):
        """ calculate positive scores and/or negative scores for hyperbolic embedding

        Parameters
        ----------
        encoded_data: dict
            dict of embeddings sliced by sample indicies of training data
        data: dict
            dict of training data containing head, rel, tail, neg indicies and metadata
        gpu_id: int
            which gpu to perform this calculation

        Returns
        -------
        dict of tensors
            positive_score,  (neg_score/neg_score_head, neg_score_tail)
        """
        head, head_bias, rel, rel_diag, curvature, context, scale, tail, tail_bias = encoded_data['head'], encoded_data['head_bias'], encoded_data['rel'], encoded_data['rel_diag'], encoded_data['curvature'], encoded_data['context'], encoded_data['scale'], encoded_data['tail'], encoded_data['tail_bias']
        pos_score = self._score_func.predict(head, head_bias, rel, rel_diag, curvature, context, scale, tail, tail_bias)
        if 'neg_type' not in data.keys():
            return {'pos_score': pos_score}
        else:
            chunk_size, neg_sample_size = data['chunk_size'], data['neg_sample_size']
            neg = encoded_data['neg']
            if data['neg_type'] == 'head':
                neg_head_bias = encoded_data['neg_head_bias']
                neg_func = self._score_func.create_neg(True)
                neg_score = neg_func(neg, neg_head_bias, rel, rel_diag, curvature, context, scale, tail, tail_bias, chunk_size, neg_sample_size)
                return {'pos_score': pos_score,
                        'neg_score': neg_score}
            elif data['neg_type'] == 'tail':
                neg_tail_bias = encoded_data['neg_tail_bias']
                neg_func = self._score_func.create_neg(False)
                neg_score = neg_func(head, head_bias, rel, rel_diag, curvature, context, scale, neg, neg_tail_bias, chunk_size, neg_sample_size)
                return {'pos_score': pos_score,
                        'neg_score': neg_score}
            elif data['neg_type'] == 'head_tail':
                neg_head_bias, neg_tail_bias = encoded_data['neg_head_bias'], encoded_data['neg_tail_bias']
                neg_func_head = self._score_func.create_neg(True)
                neg_score_head = neg_func_head(neg, neg_head_bias, rel, rel_diag, curvature, context, scale, tail, tail_bias, chunk_size, neg_sample_size)
                neg_func_tail = self._score_func.create_neg(False)
                neg_score_tail = neg_func_tail(head, head_bias, rel, rel_diag, curvature, context, scale, neg, neg_tail_bias, chunk_size, neg_sample_size)
                return {'pos_score': pos_score,
                        'neg_score_head': neg_score_head,
                        'neg_score_tail': neg_score_tail}
            else:
                raise ValueError(f"{data['neg_type']} is not correct, choose from head, tail, both.")

    def set_training_params(self, args):
        pass

    def set_test_params(self, args):
        pass

    def infer(self, encoded_data, data, gpu_id):
        pass

    def evaluate(self, results, data, graph):
        return self.metrics_evaluator.evaluate(results, data, graph)




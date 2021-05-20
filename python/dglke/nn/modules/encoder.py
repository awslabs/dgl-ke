from .module import Module
import torch as th
from torch import nn
import numpy as np

class KGEEncoder(Module):
    def __init__(self,
                 hidden_dim,
                 n_entity,
                 n_relation,
                 init_func,
                 score_func='TransE'):
        encoder_name = 'KGEEncoder'
        super(KGEEncoder, self).__init__(encoder_name)
        self.score_func = score_func 
        double_ent = (score_func == 'RotatE') or (score_func == 'SimplE')
        double_rel = score_func == 'SimplE'
        self.entity_emb = nn.Embedding(n_entity, 2 * hidden_dim if double_ent else hidden_dim, sparse=True)
        if score_func == "RESCAL":
            self.relation_emb = nn.Embedding(n_relation, hidden_dim * hidden_dim, sparse=True)
        else:
            self.relation_emb = nn.Embedding(n_relation, 2 * hidden_dim if double_rel else hidden_dim, sparse=True)
        # use init func to initialize parameters
        init_func[0](self.entity_emb.weight.data)
        init_func[1](self.relation_emb.weight.data)

    def forward(self, data, gpu_id):
        fwd_data = {k: v.to(f'cuda:{gpu_id}') if type(v) == th.Tensor else v for k, v in data.items()} if gpu_id != -1 else data
        head = self.entity_emb(fwd_data['head'])
        tail = self.entity_emb(fwd_data['tail'])
        rel = self.relation_emb(fwd_data['rel'])
        neg = self.entity_emb(fwd_data['neg'])

        return {'head': head,
                'rel': rel,
                'tail': tail,
                'neg': neg}

    def set_training_params(self, args):
        pass

    def set_test_params(self, args):
        pass

    def evaluate(self, results: list, data, graph):
        pass

class TransREncoder(Module):
    def __init__(self,
                 hidden_dim,
                 n_entity,
                 n_relation,
                 init_func):
        encoder_name = 'TransREncoder'
        super(TransREncoder, self).__init__(encoder_name)
        self.entity_emb = nn.Embedding(n_entity, hidden_dim, sparse=True)
        self.relation_emb = nn.Embedding(n_relation, hidden_dim, sparse=True)
        # use init func to initialize parameters
        init_func[0](self.entity_emb.weight.data)
        init_func[1](self.relation_emb.weight.data)

    def forward(self, data, gpu_id):
        fwd_data = {k: v.to(f'cuda:{gpu_id}') if type(v) == th.Tensor else v for k, v in data.items()} if gpu_id != -1 else data
        head = self.entity_emb(fwd_data['head'])
        tail = self.entity_emb(fwd_data['tail'])
        rel = self.relation_emb(fwd_data['rel'])
        neg = self.entity_emb(fwd_data['neg'])

        rel_id = fwd_data['rel']
        return {'head': head,
                'rel': rel,
                'tail': tail,
                'neg': neg,
                'rel_id': rel_id}

    def set_training_params(self, args):
        pass

    def set_test_params(self, args):
        pass

    def evaluate(self, results: list, data, graph):
        pass

class AttHEncoder(Module):
    """ Attention Hyperbolic Embedding Encoder
    paper: https://arxiv.org/abs/2005.00545
    """
    def __init__(self,
                 hidden_dim,
                 n_entity,
                 n_relation,
                 encoder_name='AttHEncoder'):
        super(AttHEncoder, self).__init__(encoder_name)
        init_size = 0.001
        dtype = th.double
        self.scale = th.tensor([1. / np.sqrt(hidden_dim)], dtype=dtype)
        self.entity_emb = nn.Embedding(n_entity, hidden_dim, sparse=True)
        self.entity_emb.weight.data = init_size * th.randn(n_entity, hidden_dim, dtype=dtype)
        self.relation_emb = nn.Embedding(n_relation, hidden_dim, sparse=True)
        self.relation_emb.weight.data = init_size * th.randn(n_relation, hidden_dim, dtype=dtype)
        self.relation_diag = nn.Embedding(n_relation, hidden_dim * 2, sparse=True)
        self.relation_diag.weight.data = 2 * th.rand(n_relation, hidden_dim * 2, dtype=dtype) - 1.0
        self.curvature = th.nn.Parameter(th.ones((n_relation, 1), dtype=dtype), requires_grad=True)
        self.context = nn.Embedding(n_relation, hidden_dim, sparse=True)
        self.context.weight.data = init_size * th.randn((n_relation, hidden_dim), dtype=dtype)
        self.head_bias = nn.Embedding(n_entity, 1, sparse=True)
        self.head_bias.weight.data = th.zeros((n_entity, 1), dtype=dtype)
        self.tail_bias = nn.Embedding(n_entity, 1, sparse=True)
        self.tail_bias.weight.data = th.zeros((n_entity, 1), dtype=dtype)

    def set_training_params(self, args):
        pass

    def set_test_params(self, args):
        pass

    def forward(self, data, gpu_id):
        fwd_data = {k: v.to(f'cuda:{gpu_id}') if type(v) == th.Tensor else v for k, v in data.items()} if gpu_id != -1 else data
        neg_type = data['neg_type']
        encoded_data = {}
        encoded_data['scale'] = self.scale.to(f'cuda:{gpu_id}' if gpu_id != -1 else 'cpu')
        encoded_data['head'] = self.entity_emb(fwd_data['head'])
        encoded_data['tail'] = self.entity_emb(fwd_data['tail'])
        encoded_data['rel'] = self.relation_emb(fwd_data['rel'])
        encoded_data['neg'] = self.entity_emb(fwd_data['neg'])
        encoded_data['curvature'] = self.curvature[fwd_data['rel']]
        encoded_data['rel_diag'] = self.relation_diag(fwd_data['rel'])
        encoded_data['context'] = self.context(fwd_data['rel'])
        encoded_data['head_bias'] = self.head_bias(fwd_data['head'])
        encoded_data['tail_bias'] = self.tail_bias(fwd_data['tail'])
        if neg_type == 'head':
            encoded_data['neg_head_bias'] = self.head_bias(fwd_data['neg'])
        elif neg_type == 'tail':
            encoded_data['neg_tail_bias'] = self.tail_bias(fwd_data['neg'])
        elif neg_type == 'head_tail':
            encoded_data['neg_head_bias'] = self.head_bias(fwd_data['neg'])
            encoded_data['neg_tail_bias'] = self.tail_bias(fwd_data['neg'])
        else:
            raise ValueError(f"{data['neg_type']} is not supported, choose from head, tail, both.")
        return encoded_data
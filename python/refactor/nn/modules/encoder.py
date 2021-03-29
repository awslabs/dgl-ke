from .module import Module
import torch as th
from torch import nn
import itertools

class KGEEncoder(Module):
    def __init__(self,
                 hidden_dim,
                 n_entity,
                 n_relation,
                 init_func,
                 encoder_name='KGEEncoder',):
        super(KGEEncoder, self).__init__(encoder_name)
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
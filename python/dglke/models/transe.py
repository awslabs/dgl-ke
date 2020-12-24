from .ke_model import GEModel, EMB_INIT_EPS
from .pytorch.score_fun import TransEScore
from dglke.util import get_device
import torch as th

class TransEModel(GEModel):
    def __init__(self, args, device, model_name):
        score_func = TransEScore(args.gamma, dist_func='l1')
        super(TransEModel, self).__init__(args, device, model_name, score_func)

    def categorize_embedding(self):
        self._entity_related_emb.update({'entity_emb': self._entity_emb})
        self._relation_related_emb.update({'relation_emb': self._relation_emb})

    def initialize(self, n_entities, n_relations, init_strat='uniform'):
        args = self.args
        eps = EMB_INIT_EPS
        emb_init = (args.gamma + eps) / args.hidden_dim
        self._entity_emb.init(emb_init=emb_init, lr=self.lr, async_threads=args.num_thread, num=n_entities, dim=self.hidden_dim,
                              init_strat=init_strat, optimizer=args.optimizer, device=self.entity_related_device)
        self._relation_emb.init(emb_init=emb_init, lr=self.lr, async_threads=args.num_thread, num=n_relations, dim=self.hidden_dim,
                                init_strat=init_strat, optimizer=args.optimizer, device=self.relation_related_device)

    def acquire_embedding(self, data, gpu_id=-1, pos=True, train=True, neg_type='head'):
        if pos and train:
            emb = {'head': self._entity_related_emb['entity_emb'](data['head'], gpu_id=gpu_id, trace=True),
                   'tail': self._entity_related_emb['entity_emb'](data['tail'], gpu_id=gpu_id, trace=True),
                   'rel': self._relation_related_emb['relation_emb'](data['rel'], gpu_id=gpu_id, trace=True), }
        elif not pos and train:
            if neg_type == 'head':
                emb = {'head': self._entity_related_emb['entity_emb'](data['neg'], gpu_id=gpu_id, trace=True), }
            else:
                emb = {'tail': self._entity_related_emb['entity_emb'](data['neg'], gpu_id=gpu_id, trace=True), }
        elif pos and not train:
            emb = {'head': self._entity_related_emb['entity_emb'](data['head'], gpu_id=gpu_id, trace=False),
                   'tail': self._entity_related_emb['entity_emb'](data['tail'], gpu_id=gpu_id, trace=False),
                   'rel': self._relation_related_emb['relation_emb'](data['rel'], gpu_id=gpu_id, trace=False), }
        else:
            emb = {'head': self._entity_related_emb['entity_emb'](data['neg'], gpu_id=gpu_id, trace=False),
                   'tail': self._entity_related_emb['entity_emb'](data['neg'], gpu_id=gpu_id, trace=False), }
        return emb

    def pos_forward(self, pos_emb):
        return self._score_func.predict(pos_emb)


    def neg_forward(self, pos_emb, neg_emb, neg_type, chunk_size, neg_sample_size, train=True):
        pos_emb, neg_emb = self.prepare_embedding(pos_emb, neg_emb, neg_type, chunk_size, neg_sample_size, train)
        heads, relations, tails, neg = pos_emb['head'], pos_emb['rel'], pos_emb['tail'], neg_emb[neg_type]
        num_chunk = len(heads) // chunk_size
        if neg_type == 'head':
            func = self._score_func.create_neg(True)
            return func(neg, relations, tails, num_chunk, chunk_size, neg_sample_size)
        else:
            func = self._score_func.create_neg(False)
            return func(heads, relations, neg, num_chunk, chunk_size, neg_sample_size)

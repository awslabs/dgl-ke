from .pytorch.score_fun import ATTHScore
from .pytorch.ke_tensor import KGEmbedding
from .ke_model import GEModel
import torch as th
from dglke.util import *

class AttHModel(GEModel):
    def __init__(self, args, device, model_name):
        score_func = ATTHScore()
        super(AttHModel, self).__init__(args, device, model_name, score_func)
        self._rel_diag = KGEmbedding(device)
        self._c = KGEmbedding(device)
        self._context = KGEmbedding(device)
        self._head_bias = KGEmbedding(device)
        self._tail_bias = KGEmbedding(device)
        # self.multi_c = args.multi_c
        self.gamma = args.gamma
        self._scale = (1. / np.sqrt(args.hidden_dim))

    def categorize_embedding(self):
        self._entity_related_emb.update({'head_bias': self._head_bias,
                                         'tail_bias': self._head_bias,
                                         'entity_emb': self._entity_emb,
                                         })
        self._relation_related_emb.update({'curvature': self._c,
                                           'context': self._context,
                                           'rel_diag': self._rel_diag,
                                           'relation_emb': self._relation_emb,
                                           })

    def initialize(self, n_entities, n_relations, init_strat='uniform'):
        args = self.args
        init_scale = args.init_scale
        self._entity_emb.init(init_scale, lr=self.lr, async_threads=args.num_thread, num=n_entities, dim=self.hidden_dim,
                              init_strat='normal', optimizer=args.optimizer, device=self.entity_related_device)
        self._relation_emb.init(init_scale, lr=self.lr, async_threads=args.num_thread, num=n_relations, dim=self.hidden_dim,
                                init_strat='normal', optimizer=args.optimizer, device=self.relation_related_device)
        self._rel_diag.init(emb_init=(2, -1), lr=self.lr, async_threads=args.num_thread, num=n_relations, dim=self.hidden_dim * 2,
                            init_strat='random', optimizer=args.optimizer, device=self.relation_related_device)
        self._c.init(emb_init=1, lr=self.lr, async_threads=args.num_thread, num=n_relations, dim=1,
                     init_strat='constant', optimizer=args.optimizer, device=self.relation_related_device)
        self._context.init(emb_init=init_scale, lr=self.lr, async_threads=args.num_thread, num=n_relations, dim=self.hidden_dim,
                           init_strat='normal', optimizer=args.optimizer, device=self.relation_related_device)
        self._head_bias.init(emb_init=0, lr=self.lr, async_threads=args.num_thread, num=n_entities, dim=1,
                             init_strat='constant', optimizer=args.optimizer, device=self.entity_related_device)
        self._tail_bias.init(emb_init=0, lr=self.lr, async_threads=args.num_thread, num=n_entities, dim=1,
                             init_strat='constant', optimizer=args.optimizer, device=self.entity_related_device)


    def prepare_embedding(self, pos_emb, neg_emb, neg_type, chunk_size, neg_sample_size, train=True):
        pos_batch_size = pos_emb['head'].shape[0]
        neg_batch_size = neg_emb[neg_type].shape[0]
        pos_emb_reshape = {}
        neg_emb_reshape = {}
        if train and neg_type == 'head':
            for k in pos_emb.keys():
                pos_emb_reshape[k] = pos_emb[k].view(pos_batch_size // chunk_size, chunk_size, -1)
            for k in neg_emb.keys():
                neg_emb_reshape[k] = neg_emb[k].view(neg_batch_size // neg_sample_size, neg_sample_size, -1)
        elif train and neg_type == 'tail':
            pos_emb_reshape.update(pos_emb)
            pos_emb_reshape['head_bias'] = pos_emb_reshape['head_bias'].view(pos_batch_size // chunk_size,
                                                                             chunk_size, -1)
            for k in neg_emb.keys():
                neg_emb_reshape[k] = neg_emb[k].view(neg_batch_size // neg_sample_size, neg_sample_size, -1)
        elif not train and neg_type == 'head':
            for k in pos_emb.keys():
                pos_emb_reshape[k] = pos_emb[k].view(1, pos_batch_size, -1)
            for k in neg_emb.keys():
                neg_emb_reshape[k] = neg_emb[k].view(1, neg_batch_size, -1)
        elif not train and neg_type == 'tail':
            pos_emb_reshape.update(pos_emb)
            pos_emb_reshape['head_bias'] = pos_emb_reshape['head_bias'].view(1, pos_batch_size, -1)
            for k in neg_emb.keys():
                neg_emb_reshape[k] = neg_emb[k].view(1, neg_batch_size, -1)
        return pos_emb_reshape, neg_emb_reshape

    def get_score(self, lhs_e, head_bias, rhs_e, tail_bias, c, comp='batch'):
        score = self._score_func(lhs_e, rhs_e, c, comp)
        if comp == 'batch':
            return self.gamma + head_bias + tail_bias + score
        else:
            return self.gamma + head_bias.unsqueeze(2) + tail_bias.unsqueeze(1) + score.unsqueeze(-1)

    def pos_forward(self, pos_emb):
        # get lhs
        rel_c = pos_emb['curvature']
        head = pos_emb['head']
        rel_diag = pos_emb['rel_diag']
        context_vec = pos_emb['context_vec']
        rel = pos_emb['rel']
        head_bias = pos_emb['head_bias']

        c = th.nn.functional.softplus(rel_c)
        rot_mat, ref_mat = th.chunk(rel_diag, 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view(-1, 1, self.hidden_dim)
        ref_q = givens_reflection(ref_mat, head).view(-1, 1, self.hidden_dim)
        cands = th.cat([ref_q, rot_q], dim=1)
        context_vec = context_vec.view(-1, 1, self.hidden_dim)
        att_weights = th.sum(context_vec * cands * self._scale, dim=-1, keepdim=True)
        att_weights = th.nn.functional.softmax(att_weights, dim=1)
        att_q = th.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel = expmap0(rel, c)
        lhs = project(mobius_add(lhs, rel, c), c)

        # get rhs
        rhs = pos_emb['tail']
        tail_bias = pos_emb['tail_bias']

        score = self.get_score(lhs, head_bias, rhs, tail_bias, c, comp='batch')
        return score

    def neg_forward(self, pos_emb, neg_emb, neg_type, chunk_size, neg_sample_size, train=True):
        pos_emb, neg_emb = self.prepare_embedding(pos_emb, neg_emb, neg_type, chunk_size, neg_sample_size, train=train)

        if neg_type == 'head':
            #  for head
            head = neg_emb['head']
            head_bias = neg_emb['head_bias'].unsqueeze(1)

            #  for relation
            rel_diag = pos_emb['rel_diag']
            rel_c = pos_emb['curvature']
            rel = pos_emb['rel']
            context_vec = pos_emb['context_vec']

            # for tail
            rhs = pos_emb['tail'].unsqueeze(2)
            tail_bias = pos_emb['tail_bias'].unsqueeze(2)

            c = th.nn.functional.softplus(rel_c)
            rot_mat, ref_mat = th.chunk(rel_diag, 2, dim=-1)
            # batch, chunk, neg, hidden
            rot_q = givens_rotations(rot_mat, head, comp='mm').unsqueeze(-2)
            ref_q = givens_reflection(ref_mat, head, comp='mm').unsqueeze(-2)
            cands = th.cat([ref_q, rot_q], dim=-2)
            context_vec = context_vec.unsqueeze(2).unsqueeze(3)
            attn_weights = th.sum(context_vec * cands * self._scale, dim=-1, keepdim=True)
            attn_weights = th.nn.functional.softmax(attn_weights, dim=-2)
            att_q = th.sum(attn_weights * cands, dim=-2)
            expand_c = c.unsqueeze(2)
            lhs = expmap0(att_q, expand_c)
            rel = expmap0(rel, c).unsqueeze(2)
            lhs = project(mobius_add(lhs, rel, expand_c), expand_c)
            score = self.get_score(lhs, head_bias, rhs, tail_bias, expand_c, comp='batch')
            return score

        elif neg_type == 'tail':
            #  for head
            head = pos_emb['head']
            head_bias = pos_emb['head_bias']

            #  for relation
            rel_c = pos_emb['curvature']
            rel = pos_emb['rel']
            rel_diag = pos_emb['rel_diag']
            context_vec = pos_emb['context_vec']

            # for tail
            rhs = neg_emb['tail']
            tail_bias = neg_emb['tail_bias']

            c = th.nn.functional.softplus(rel_c)
            rot_mat, ref_mat = th.chunk(rel_diag, 2, dim=-1)
            rot_q = givens_rotations(rot_mat, head).view(-1, 1, self.hidden_dim)
            ref_q = givens_reflection(ref_mat, head).view(-1, 1, self.hidden_dim)
            cands = th.cat([ref_q, rot_q], dim=1)
            context_vec = context_vec.view(-1, 1, self.hidden_dim)
            att_weights = th.sum(context_vec * cands * self._scale, dim=-1, keepdim=True)
            att_weights = th.nn.functional.softmax(att_weights, dim=1)
            att_q = th.sum(att_weights * cands, dim=1)
            lhs = expmap0(att_q, c)
            rel = expmap0(rel, c)
            lhs = project(mobius_add(lhs, rel, c), c)
            c = c.view(c.shape[0] // chunk_size, chunk_size, -1)
            score = self.get_score(lhs.view(lhs.shape[0] // chunk_size, chunk_size, -1), head_bias, rhs, tail_bias, c,
                                   comp='mm')
            return score

    def acquire_embedding(self, data, gpu_id=-1, pos=True, train=True, neg_type='head'):
        if pos and train:
            emb = {'curvature': self._relation_related_emb['curvature'](data['rel'], gpu_id=gpu_id, trace=True),
                   'rel': self._relation_related_emb['relation_emb'](data['rel'], gpu_id=gpu_id, trace=True),
                   'rel_diag': self._relation_related_emb['rel_diag'](data['rel'], gpu_id=gpu_id, trace=True),
                   'context_vec': self._relation_related_emb['context'](data['rel'], gpu_id=gpu_id, trace=True),
                   'head': self._entity_related_emb['entity_emb'](data['head'], gpu_id=gpu_id, trace=True),
                   'tail': self._entity_related_emb['entity_emb'](data['tail'], gpu_id=gpu_id, trace=True),
                   'head_bias': self._entity_related_emb['head_bias'](data['head'], gpu_id=gpu_id, trace=True),
                   'tail_bias': self._entity_related_emb['tail_bias'](data['tail'], gpu_id=gpu_id, trace=True), }
        elif not pos and train:
            if neg_type == 'head':
                emb = {'head': self._entity_related_emb['entity_emb'](data['neg'], gpu_id=gpu_id, trace=True),
                       'head_bias': self._entity_related_emb['head_bias'](data['neg'], gpu_id=gpu_id, trace=True)}
            else:
                emb = {'tail': self._entity_related_emb['entity_emb'](data['neg'], gpu_id=gpu_id, trace=True),
                       'tail_bias': self._entity_related_emb['tail_bias'](data['neg'], gpu_id=gpu_id, trace=True)}
        elif pos and not train:
            emb = {'curvature': self._relation_related_emb['curvature'](data['rel'], gpu_id=gpu_id, trace=False),
                   'head': self._entity_related_emb['entity_emb'](data['head'], gpu_id=gpu_id, trace=False),
                   'tail': self._entity_related_emb['entity_emb'](data['tail'], gpu_id=gpu_id, trace=False),
                   'rel': self._relation_related_emb['relation_emb'](data['rel'], gpu_id=gpu_id, trace=False),
                   'rel_diag': self._relation_related_emb['rel_diag'](data['rel'], gpu_id=gpu_id, trace=False),
                   'context_vec': self._relation_related_emb['context'](data['rel'], gpu_id=gpu_id, trace=False),
                   'head_bias': self._entity_related_emb['head_bias'](data['head'], gpu_id=gpu_id, trace=False),
                   'tail_bias': self._entity_related_emb['tail_bias'](data['tail'], gpu_id=gpu_id, trace=False), }
        else:
            emb = {'head': self._entity_related_emb['entity_emb'](data['neg'], gpu_id=gpu_id, trace=False),
                   'head_bias': self._entity_related_emb['head_bias'](data['neg'], gpu_id=gpu_id, trace=False),
                   'tail': self._entity_related_emb['entity_emb'](data['neg'], gpu_id=gpu_id, trace=False),
                   'tail_bias': self._entity_related_emb['tail_bias'](data['neg'], gpu_id=gpu_id, trace=False)}
        return emb

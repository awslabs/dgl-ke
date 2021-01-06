import torch as th
from .ke_model import GEModel, EMB_INIT_EPS
from .pytorch.score_fun import ConvEScore
from .pytorch.ke_tensor import KGEmbedding
from dglke.util import get_device


class ConvEModel(GEModel):
    def __init__(self, args, device, model_name):
        score_func = ConvEScore(hidden_dim=args.hidden_dim,
                                tensor_height=args.tensor_height,
                                dropout_ratio=args.dropout_ratio,
                                batch_norm=args.batch_norm)
        self._entity_bias = KGEmbedding(device)
        self.h = args.tensor_height
        self.w = args.hidden_dim // self.h
        self.gamma = args.gamma
        super(ConvEModel, self).__init__(args, device, model_name, score_func)

    def batch_concat(self, tensor_a, tensor_b, dim=-2):
        """ element wise concatenation

        """
        def _reshape(tensor):
            if tensor.dim() == 2:
                batch, hidden_dim = tensor.shape
                assert hidden_dim == self.h * self.w, 'hidden dimension must match %d' % self.h * self.w
                return tensor.reshape(batch,  self.h, self.w)
            elif tensor.dim() == 3:
                batch, h, w = tensor.shape
                assert h == self.h, 'tensor height must match %d' % h
                assert w == self.w, 'tensor width must match %d' % w
                return tensor
            elif tensor.dim() == 4:
                return tensor
            else:
                raise ValueError('tensor must have dimension larger than 1')

        tensor_a = _reshape(tensor_a)
        tensor_b = _reshape(tensor_b)
        return th.cat([tensor_a, tensor_b], dim=dim)

    def mutual_concat(self, tensor_a, tensor_b, chunk_size_a=16, chunk_size_b=16, mode='AxB', dim=-2):
        """ broadcast concatenation for tensor_a and tensor_b. it is used only when corrupt negative tensor_a

        """
        def _chunk_reshape(tensor, _chunk_size):
            if tensor.dim() == 1:
                raise ValueError('tensor with dimension %d is not supported' % tensor.dim())
            elif tensor.dim() == 2:
                batch, hidden_dim = tensor.shape
                assert hidden_dim == self.h * self.w, 'hidden dimension must be %d' % self.h * self.w
                return tensor.reshape([-1, _chunk_size, self.h, self.w])
            elif tensor.dim() == 3:
                batch, h, w = tensor.shape
                assert h == self.h, 'tensor height must match %d.' % self.h
                assert w == self.w, 'tensor width must match %d.' % self.w
                return tensor.reshape([-1, _chunk_size, self.h, self.w])

        tensor_a = _chunk_reshape(tensor_a, chunk_size_a)
        tensor_b = _chunk_reshape(tensor_b, chunk_size_b)
        num_chunk, _, h, w = tensor_a.shape
        if mode is 'AxB':
            tensor_a = tensor_a.unsqueeze(2)
            tensor_b = tensor_b.unsqueeze(1)
            tensor_a = tensor_a.repeat(1, 1, chunk_size_b, 1, 1)
            tensor_b = tensor_b.repeat(1, chunk_size_a, 1, 1, 1)
        elif mode is 'BxA':
            tensor_a = tensor_a.unsqueeze(1)
            tensor_b = tensor_b.unsqueeze(2)
            tensor_a = tensor_a.repeat(1, chunk_size_b, 1, 1, 1)
            tensor_b = tensor_b.repeat(1, 1, chunk_size_a, 1, 1)
        cat_res = th.cat([tensor_a, tensor_b], dim=dim)
        cat_res = cat_res.reshape(-1, 2 * h, w)
        return cat_res

    def categorize_embedding(self):
        self._entity_related_emb.update({'entity_emb': self._entity_emb,
                                         'entity_bias': self._entity_bias})
        self._relation_related_emb.update({'relation_emb': self._relation_emb})
        self._torch_model.update({'score_func': self._score_func})

    def initialize(self, n_entities, n_relations, init_strat='xavier'):
        args = self.args
        eps = EMB_INIT_EPS
        emb_init = (args.gamma + eps) / args.hidden_dim
        device = get_device(args)
        entity_related_device = th.device('cpu') if args.mix_cpu_gpu else device
        relation_related_device = th.device('cpu') if (args.mix_cpu_gpu or args.strict_rel_part or args.soft_rel_part) else device
        self._relation_emb.init(emb_init=emb_init, lr=self.lr, async_threads=None, num=n_relations, dim=self.hidden_dim,
                                init_strat=init_strat, device=relation_related_device)
        self._entity_emb.init(emb_init=emb_init, lr=self.lr, async_threads=None, num=n_entities, dim=self.hidden_dim,
                              init_strat=init_strat, device=entity_related_device)
        self._entity_bias.init(emb_init=0, lr=self.lr, async_threads=None, num=n_entities, dim=1, init_strat='uniform', device=entity_related_device)



    def pos_forward(self, pos_emb):
        concat_emb = self.batch_concat(pos_emb['head'], pos_emb['rel'], dim=-2)
        return self.gamma - self._score_func(embs=[concat_emb, pos_emb['tail'], pos_emb['tail_bias']],
                                mode='all',
                                bmm=False,)

    def neg_forward(self, pos_emb, neg_emb, neg_type, chunk_size, neg_sample_size, train):
        args = self.args
        # if neg_type == 'head':
        #     concat_emb = self.mutual_concat(neg_emb['neg'], pos_emb['rel'],
        #                                     chunk_size_a=neg_sample_size, chunk_size_b=chunk_size,
        #                                     mode='BxA', dim=-2)
        #     lhs = self._score_func(embs=[concat_emb], mode='lhs', bmm=False).reshape(-1, chunk_size, neg_sample_size, args.hidden_dim)
        #     tail_emb = pos_emb['tail'].reshape(-1, chunk_size, 1, args.hidden_dim)
        #     tail_bias = pos_emb['tail_bias'].reshape(-1, chunk_size, 1, 1)
        #     return self.gamma - self._score_func(embs=[lhs, tail_emb, tail_bias], mode='rhs', bmm=False)
        # else:
        concat_emb = self.batch_concat(pos_emb['head'], pos_emb['rel'], dim=-2)
        lhs = self._score_func(embs=[concat_emb], mode='lhs', bmm=False).reshape(-1, chunk_size, args.hidden_dim)

        tail_emb = neg_emb['neg']
        tail_bias = neg_emb['neg_bias']
        _, emb_dim = tail_emb.shape
        tail_emb = tail_emb.reshape(-1, neg_sample_size, emb_dim)
        tail_bias = tail_bias.reshape(-1, neg_sample_size, 1)
        # bmm
        score = self._score_func(embs=[lhs, tail_emb, tail_bias], mode='rhs', bmm=True)
        return self.gamma - score

    def acquire_embedding(self, data, gpu_id=-1, pos=True, train=True, neg_type='head'):
        if pos and train:
            emb = {'head': self._entity_related_emb['entity_emb'](data['head'], gpu_id=gpu_id, trace=True),
                   'tail': self._entity_related_emb['entity_emb'](data['tail'], gpu_id=gpu_id, trace=True),
                   'rel': self._relation_related_emb['relation_emb'](data['rel'], gpu_id=gpu_id, trace=True),
                   'tail_bias': self._entity_related_emb['entity_bias'](data['tail'], gpu_id=gpu_id, trace=True), }
        elif not pos and train:
            emb = {'neg': self._entity_related_emb['entity_emb'](data['neg'], gpu_id=gpu_id, trace=True),
                   'neg_bias': self._entity_related_emb['entity_bias'](data['neg'], gpu_id=gpu_id, trace=True)}
        elif pos and not train:
            emb = {'head': self._entity_related_emb['entity_emb'](data['head'], gpu_id=gpu_id, trace=False),
                   'rel': self._relation_related_emb['relation_emb'](data['rel'], gpu_id=gpu_id, trace=False),
                   'tail': self._entity_related_emb['entity_emb'](data['tail'], gpu_id=gpu_id, trace=False),
                   'tail_bias': self._entity_related_emb['entity_bias'](data['tail'], gpu_id=gpu_id, trace=False)
                   }
        else:
            emb = {'neg': self._entity_related_emb['entity_emb'](data['neg'], gpu_id=gpu_id, trace=False),
                   'neg_bias': self._entity_related_emb['entity_bias'](data['neg'], gpu_id=gpu_id, trace=False)}
        return emb

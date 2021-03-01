from .module import Module
from .ke_embedding import KGEmbedding
import torch as th
import itertools

# this is an interface so it will not have many parameters to be defined.
class Encoder(Module):
    def __init__(self, encoder_name):
        super(Encoder, self).__init__(encoder_name)

    def forward(self, data, gpu_id):
        raise NotImplementedError

class KGEEncoder(Encoder):
    def __init__(self,
                 hidden_dim,
                 lr,
                 optimizer,
                 n_entity,
                 n_relation,
                 init_func,
                 rel_parts=None, # ! rel_parts and cross_rels must be put here otherwise harm the general structure
                 cross_rels=None,
                 soft_rel_part=False,
                 strict_rel_part=False,
                 encoder_name='KGEEncoder',):
        super(KGEEncoder, self).__init__(encoder_name)
        self.entity_emb = KGEmbedding()
        self.relation_emb = KGEmbedding()
        self.entity_emb.init(lr=lr,
                             num=n_entity,
                             dim=hidden_dim,
                             init_func=init_func[0],
                             optimizer=optimizer)
        self.relation_emb.init(lr=lr,
                               num=n_relation,
                               dim=hidden_dim,
                               init_func=init_func[1],
                               optimizer=optimizer)
        self.rel_parts = rel_parts
        self.cross_rels = cross_rels
        self.soft_rel_part = soft_rel_part
        self.strict_rel_part = strict_rel_part
        self.init = False

    def save(self, save_path):
        self.entity_emb.save(save_path, 'entity_emb.npy')
        self.relation_emb.save(save_path, 'relation_emb.npy')

    def load(self, load_path):
        self.entity_emb.load(load_path, 'entity_emb.npy')
        self.relation_emb.load(load_path, 'relation_emb.npy')

    def train(self):
        self.entity_emb.train()
        self.relation_emb.train()

    def eval(self):
        self.entity_emb.eval()
        self.relation_emb.eval()

    # not optimized
    def sparse_parameters(self):
        yield self.entity_emb.curr_emb()
        yield self.relation_emb.curr_emb()

    def forward(self, data, gpu_id):
        head = self.entity_emb(data['head'], gpu_id=gpu_id, non_blocking=self.pin_memory)
        tail = self.entity_emb(data['tail'], gpu_id=gpu_id, non_blocking=self.pin_memory)
        rel = self.relation_emb(data['rel'], gpu_id=gpu_id, non_blocking=self.pin_memory)
        neg = self.entity_emb(data['neg'], gpu_id=gpu_id, non_blocking=self.pin_memory)

        return {'head': head,
                'rel': rel,
                'tail': tail,
                'neg': neg}

    def set_training_params(self, args):
        self.async_update = args.async_update
        self.async_threads = args.num_thread
        self.pin_memory = args.pin_memory

    def set_test_params(self, args):
        self.pin_memory = args.pin_memory

    def prepare_model(self, gpu_id, rank, world_size):
        if not self.init:
            if self.async_update:
                self.create_async_update(self.async_threads)
            if world_size > 1 and self.soft_rel_part:
                self.global_relation_emb = self.relation_emb
                self.relation_emb = self.relation_emb.to(gpu_id)
            else:
                self.entity_emb = self.entity_emb.to(gpu_id)
                self.relation_emb = self.relation_emb.to(gpu_id)
            self.init = True
        if self.soft_rel_part:
            self.prepare_cross_rels()

    def sync_model(self, gpu_id, rank, world_size):
        if self.strict_rel_part or self.soft_rel_part:
            self.writeback_relation(rank)

    def postprocess_model(self, gpu_id, rank, world_size):
        if self.async_update:
            self.finish_async_update()
        if self.strict_rel_part or self.soft_rel_part:
            self.writeback_relation(rank)

    def create_async_update(self, async_threads):
        self.entity_emb.create_async_update(async_threads)
        self.relation_emb.create_async_update(async_threads)

    def finish_async_update(self):
        self.entity_emb.finish_async_update()
        self.relation_emb.finish_async_update()

    def share_memory(self):
        self.relation_emb.share_memory()
        self.entity_emb.share_memory()

    def update(self, gpu_id):
        self.entity_emb.update(gpu_id)
        self.relation_emb.update(gpu_id)

    def prepare_cross_rels(self):
        self.relation_emb.setup_cross_rels(self.cross_rels, self.global_relation_emb)

    def writeback_relation(self, rank=0):
        idx = self.rel_parts[rank]
        if self.soft_rel_part:
            local_idx = self.relation_emb.get_noncross_idx(idx)
        else:
            local_idx = idx
        self.global_relation_emb.emb[local_idx] = self.relation_emb.emb.detach().clone().cpu()[local_idx]

    def evaluate(self, results: list, data, graph):
        pass
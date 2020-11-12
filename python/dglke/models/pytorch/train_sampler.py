import torch.multiprocessing as mp
import torch as th
import dgl.backend as F
import torch.multiprocessing as mp




class TrainSampler(object):
    # TODO lingfei
    # 1. [x] support nagative mode: head, tail, chunk-head, chunk-tail
    # 2. [x] add rel_weight, node_weight to non-uniformly sample rels
    # 3. [x] infinite sample or not?
    # 4. [ ] num_workers -> multiprocess
    # 5. [x] replacement
    # 6. [x] reset
    # 7. [x] negative mode -> head, tail
    # 8. [x] neg_sample_size ?
    # 9. [x] chunk_size
    # 10.[ ] exclude_positive -> mask

    def __init__(self,
                 train_data,
                 rank,
                 batch_size,
                 shuffle,
                 rel_weight,
                 neg_sample_size,
                 chunk_size,
                 exclude_positive=False,
                 replacement=False,
                 reset=True,
                 drop_last=True):
        # seed_edges are the index of triple
        g = train_data.g
        seed_edges = train_data.edge_parts[rank]
        if seed_edges is None:
            seed_edges = F.arange(0, g.number_of_edges())
        assert batch_size % chunk_size == 0, 'batch size {} must be divisible by chunk size {} to enable chunk negative sampling'.format(batch_size, chunk_size)
        self.rels = g.edata['tid'][seed_edges]
        heads, tails = g.all_edges(order='eid')
        self.heads = heads[seed_edges]
        self.tails = tails[seed_edges]
        self.node_pool = g.nodes()
        self.reset = reset
        self.replacement = replacement
        # self.chunk_size = chunk_size
        # self.neg_sample_size = neg_sample_size
        # TODO mask all false negative rels
        self.exclude_positive = exclude_positive
        self.drop_last = drop_last
        # might be replaced by rel weight vector provided
        self.rel_weight = th.ones(len(self.rels), dtype=th.float32) if rel_weight is None else rel_weight[seed_edges]
        # shuffle data
        if shuffle:
            # MARK - whether to shuffle data or shuffle indices only?
            self.node_pool = self.node_pool[th.randperm(len(self.node_pool))]
            idx = th.randperm(len(self.rels))
            self.rels = self.rels[idx]
            self.heads = self.heads[idx]
            self.tails = self.tails[idx]
            # the rel weight need to shuffle together to ensure consistency
            self.rel_weight = self.rel_weight[idx]

        self.batch_size = batch_size
        self.pool_size = self.batch_size // chunk_size * neg_sample_size
        self.iter_idx = 0
        self.pool_idx = 0
        self.step = 0

    def __iter__(self):
        return self

    # without multiprocess
    def __next__(self):
        pos = {}
        neg = {}
        if self.replacement:
            # choose with replacement with weight given, default weight for each rel is 1
            selected_idx = th.multinomial(self.rel_weight, num_samples=self.batch_size, replacement=True)
            pos['head'] = self.heads[selected_idx]
            pos['rel'] = self.rels[selected_idx]
            pos['tail'] = self.tails[selected_idx]
        else:
            end_iter_idx = min(self.iter_idx + self.batch_size, self.__len__())
            pos['head'] = self.heads[self.iter_idx: end_iter_idx]
            pos['rel'] = self.rels[self.iter_idx: end_iter_idx]
            pos['tail'] = self.tails[self.iter_idx: end_iter_idx]
        # need to setup lock to avoid mess
        end_pool_idx = min(self.pool_idx + self.pool_size, len(self.node_pool))
        neg_type = 'head' if self.step % 2 == 0 else 'tail'
        neg[neg_type] = self.node_pool[self.pool_idx: end_pool_idx]
        # neg['head' if self.step % 2 == 0 else 'tail'] = self.corrupt()
        self.iter_idx += self.batch_size
        self.pool_idx += self.pool_size
        self.step += 1

        if self.reset and self.iter_idx + self.batch_size >= self.__len__():
            self.iter_idx = 0
            # shuffle data after each epoch
            idx = th.randperm(len(self.rels))
            self.rels = self.rels[idx]
            self.heads = self.heads[idx]
            self.tails = self.tails[idx]
            # the rel weight need to shuffle together to ensure consistency
            self.rel_weight = self.rel_weight[idx]

        # if we run out of neg sample data, we shuffle it again
        if self.pool_idx + self.pool_size >= len(self.node_pool):
            self.pool_idx = 0
            idx = th.randperm(len(self.node_pool))
            self.node_pool = self.node_pool[idx]

        return pos, neg, neg_type

    # def corrupt(self):
    #     # we currently only support chunk_head and chunk_tail
    #     # MARK - discuss replacement with mentor
    #     if self.step % 2 == 0:
    #         chunk_idx = th.multinomial(self.h_weight, num_samples=self.neg_sample_size * (self.batch_size // self.chunk_size),replacement=True)
    #         return self.u_hid[chunk_idx]
    #     else:
    #         chunk_idx = th.multinomial(self.t_weight, num_samples=self.neg_sample_size * (self.batch_size // self.chunk_size), replacement=True)
    #         return self.u_tid[chunk_idx]

    def __len__(self):
        return self.rels.shape[0] if not self.drop_last \
            else (self.rels.shape[0] // self.batch_size * self.batch_size)





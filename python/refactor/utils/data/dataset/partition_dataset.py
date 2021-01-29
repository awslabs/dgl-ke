import pickle
import math
import numpy as np
import scipy as sp
import dgl.backend as F
import dgl
from torch.utils.data import Dataset
import copy
import tqdm
import torch as th
import os


def SoftRelationPartition(edges, n, has_importance=False, threshold=0.05):
    """This partitions a list of edges to n partitions according to their
    rel types. For any rel with number of edges larger than the
    threshold, its edges will be evenly distributed into all partitions.
    For any rel with number of edges smaller than the threshold, its
    edges will be put into one single partition.

    Algo:
    For r in rels:
        if r.size() > threshold
            Evenly divide edges of r into n parts and put into each rel.
        else
            Find partition with fewest edges, and put edges of r into
            this partition.

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list to partition
    n : int
        Number of partitions
    threshold : float
        The threshold of whether a rel is LARGE or SMALL
        Default: 5%

    Returns
    -------
    List of np.array
        Edges of each partition
    List of np.array
        Edge types of each partition
    bool
        Whether there exists some rels belongs to multiple partitions
    """
    if has_importance:
        heads, rels, tails, e_impts = edges
    else:
        heads, rels, tails = edges
    print('rel partition {} edges into {} parts'.format(len(heads), n))
    uniq, cnts = np.unique(rels, return_counts=True)
    idx = np.flip(np.argsort(cnts))
    cnts = cnts[idx]
    uniq = uniq[idx]
    assert cnts[0] > cnts[-1]
    edge_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_dict = {}
    rel_parts = []
    cross_rel_part = []
    for _ in range(n):
        rel_parts.append([])

    large_threshold = int(len(rels) * threshold)
    capacity_per_partition = int(len(rels) / n)
    # ensure any rel larger than the partition capacity will be split
    large_threshold = capacity_per_partition if capacity_per_partition < large_threshold \
        else large_threshold
    num_cross_part = 0
    for i in range(len(cnts)):
        cnt = cnts[i]
        r = uniq[i]
        r_parts = []
        if cnt > large_threshold:
            avg_part_cnt = (cnt // n) + 1
            num_cross_part += 1
            for j in range(n):
                part_cnt = avg_part_cnt if cnt > avg_part_cnt else cnt
                r_parts.append([j, part_cnt])
                rel_parts[j].append(r)
                edge_cnts[j] += part_cnt
                rel_cnts[j] += 1
                cnt -= part_cnt
            cross_rel_part.append(r)
        else:
            idx = np.argmin(edge_cnts)
            r_parts.append([idx, cnt])
            rel_parts[idx].append(r)
            edge_cnts[idx] += cnt
            rel_cnts[idx] += 1
        rel_dict[r] = r_parts

    for i, edge_cnt in enumerate(edge_cnts):
        print('part {} has {} edges and {} rels'.format(i, edge_cnt, rel_cnts[i]))
    print('{}/{} duplicated rel across partitions'.format(num_cross_part, len(cnts)))

    parts = []
    for i in range(n):
        parts.append([])
        rel_parts[i] = np.array(rel_parts[i])

    for i, r in enumerate(rels):
        r_part = rel_dict[r][0]
        part_idx = r_part[0]
        cnt = r_part[1]
        parts[part_idx].append(i)
        cnt -= 1
        if cnt == 0:
            rel_dict[r].pop(0)
        else:
            rel_dict[r][0][1] = cnt

    for i, part in enumerate(parts):
        parts[i] = np.array(part, dtype=np.int64)
    shuffle_idx = np.concatenate(parts)
    heads[:] = heads[shuffle_idx]
    rels[:] = rels[shuffle_idx]
    tails[:] = tails[shuffle_idx]
    if has_importance:
        e_impts[:] = e_impts[shuffle_idx]

    off = 0
    for i, part in enumerate(parts):
        parts[i] = np.arange(off, off + len(part))
        off += len(part)
    cross_rel_part = np.array(cross_rel_part)

    return parts, rel_parts, num_cross_part > 0, cross_rel_part

def BalancedRelationPartition(edges, n, has_importance=False):
    """This partitions a list of edges based on rels to make sure
    each partition has roughly the same number of edges and rels.
    Algo:
    For r in rels:
      Find partition with fewest edges
      if r.size() > num_of empty_slot
         put edges of r into this partition to fill the partition,
         find next partition with fewest edges to put r in.
      else
         put edges of r into this partition.

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list to partition
    n : int
        number of partitions

    Returns
    -------
    List of np.array
        Edges of each partition
    List of np.array
        Edge types of each partition
    bool
        Whether there exists some rels belongs to multiple partitions
    """
    if has_importance:
        heads, rels, tails, e_impts = edges
    else:
        heads, rels, tails = edges
    print('rel partition {} edges into {} parts'.format(len(heads), n))
    uniq, cnts = np.unique(rels, return_counts=True)
    idx = np.flip(np.argsort(cnts))
    cnts = cnts[idx]
    uniq = uniq[idx]
    assert cnts[0] > cnts[-1]
    edge_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_dict = {}
    rel_parts = []
    for _ in range(n):
        rel_parts.append([])

    max_edges = (len(rels) // n) + 1
    num_cross_part = 0
    for i in range(len(cnts)):
        cnt = cnts[i]
        r = uniq[i]
        r_parts = []

        while cnt > 0:
            idx = np.argmin(edge_cnts)
            if edge_cnts[idx] + cnt <= max_edges:
                r_parts.append([idx, cnt])
                rel_parts[idx].append(r)
                edge_cnts[idx] += cnt
                rel_cnts[idx] += 1
                cnt = 0
            else:
                cur_cnt = max_edges - edge_cnts[idx]
                r_parts.append([idx, cur_cnt])
                rel_parts[idx].append(r)
                edge_cnts[idx] += cur_cnt
                rel_cnts[idx] += 1
                num_cross_part += 1
                cnt -= cur_cnt
        rel_dict[r] = r_parts

    for i, edge_cnt in enumerate(edge_cnts):
        print('part {} has {} edges and {} rels'.format(i, edge_cnt, rel_cnts[i]))
    print('{}/{} duplicated rel across partitions'.format(num_cross_part, len(cnts)))

    parts = []
    for i in range(n):
        parts.append([])
        rel_parts[i] = np.array(rel_parts[i])

    for i, r in enumerate(rels):
        r_part = rel_dict[r][0]
        part_idx = r_part[0]
        cnt = r_part[1]
        parts[part_idx].append(i)
        cnt -= 1
        if cnt == 0:
            rel_dict[r].pop(0)
        else:
            rel_dict[r][0][1] = cnt

    for i, part in enumerate(parts):
        parts[i] = np.array(part, dtype=np.int64)
    shuffle_idx = np.concatenate(parts)
    heads[:] = heads[shuffle_idx]
    rels[:] = rels[shuffle_idx]
    tails[:] = tails[shuffle_idx]
    if has_importance:
        e_impts[:] = e_impts[shuffle_idx]

    off = 0
    for i, part in enumerate(parts):
        parts[i] = np.arange(off, off + len(part))
        off += len(part)

    return parts, rel_parts, num_cross_part > 0

def RandomPartition(edges, n, has_importance=False):
    """This partitions a list of edges randomly across n partitions

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list to partition
    n : int
        number of partitions

    Returns
    -------
    List of np.array
        Edges of each partition
    """
    if has_importance:
        heads, rels, tails, e_impts = edges
    else:
        heads, rels, tails = edges
    print('random partition {} edges into {} parts'.format(len(heads), n))
    idx = np.random.permutation(len(heads))
    heads[:] = heads[idx]
    rels[:] = rels[idx]
    tails[:] = tails[idx]
    if has_importance:
        e_impts[:] = e_impts[idx]

    part_size = int(math.ceil(len(idx) / n))
    parts = []
    for i in range(n):
        start = part_size * i
        end = min(part_size * (i + 1), len(idx))
        parts.append(idx[start:end])
        print('part {} has {} edges'.format(i, len(parts[-1])))
    return parts

def ConstructGraph(edges, n_entities, args):
    """Construct Graph for training

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list
    n_entities : int
        number of entities
    args :
        Global configs.
    """
    if args.has_edge_importance:
        src, etype_id, dst, e_impts = edges
    else:
        src, etype_id, dst = edges
    coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)), shape=[n_entities, n_entities])
    g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
    g.edata['tid'] = F.tensor(etype_id, F.int64)
    if args.has_edge_importance:
        g.edata['impts'] = F.tensor(e_impts, F.float32)
    return g

class BaseDataset(object):
    def __init__(self):
        pass

    def create_label(self, triples, n_entity, args, mode='train', type='tail'):
        path = os.path.join(args.data_path, args.dataset, f'{mode}_{type}_label.pkl')
        if not os.path.exists(path):
            print(f"{mode}_{type}_label.pkl does not exist, create label from scratch...")
            if type == 'tail':
                hr_dict = {}
                for idx in range(len(triples[0])):
                    if (triples[0][idx], triples[1][idx]) not in hr_dict:
                        hr_dict[(triples[0][idx], triples[1][idx])] = []
                    hr_dict[(triples[0][idx], triples[1][idx])].append(triples[2][idx])
                label_dict = {}
                for k, v in tqdm.tqdm(hr_dict.items()):
                    hr_dict[k] = np.hstack(v)
                    label_dict[k] = np.zeros(n_entity)
                    label_dict[k][hr_dict[k]] = 1
            else:
                rt_dict = {}
                for idx in range(len(triples[1])):
                    if(triples[1][idx], triples[2][idx]) not in rt_dict:
                        rt_dict[(triples[1][idx], triples[2][idx])] = []
                    rt_dict[(triples[1][idx], triples[2][idx])].append(triples[0][idx])
                label_dict = {}
                for k, v in tqdm.tqdm(rt_dict.items()):
                    rt_dict[k] = np.stack(v)
                    label_dict[k] = np.zeros(n_entity)
                    label_dict[k][rt_dict[k]] = 1

            print(f"save label dict...")
            with open(path, 'wb') as f:
                pickle.dump(label_dict, f)
        else:
            print(f"load label from cache...")
            with open(path, 'rb') as f:
                label_dict = pickle.load(f)
        return label_dict

    def partition(self, rank, world_size) -> th.utils.data.Dataset:
        raise NotImplementedError

class TrainDataset(BaseDataset):
    def __init__(self, dataset, args, ranks=1, has_importance=False, has_label=False):
        triples = dataset.train
        num_train = len(triples[0])
        self.has_importance = has_importance
        self.has_label = has_label
        print('|Train|:', num_train)

        if ranks > 1 and args.rel_part:
            self.edge_parts, self.rel_parts, self.cross_part, self.cross_rels = \
                SoftRelationPartition(triples, ranks, has_importance=has_importance)
            self.edge_parts = self.edge_parts
            self.rel_parts = self.rel_parts
            self.cross_rels = self.cross_rels
        elif ranks > 1:
            self.edge_parts = RandomPartition(triples, ranks, has_importance=has_importance)
            self.cross_part = True
        else:
            self.edge_parts = [np.arange(num_train)]
            self.rel_parts = [np.arange(dataset.n_relations)]
            self.cross_part = False

        self.g = ConstructGraph(triples, dataset.n_entities, args)
        if self.has_label:
            self.tail_label = self.create_label(triples, dataset.n_entities, args, 'train')

    def partition(self, rank, world_size) -> th.utils.data.Dataset:
        edges = self.edge_parts[rank]
        if edges is None:
            edges = th.arange(0, self.g.number_of_edges())
        heads, tails = self.g.all_edges(order='eid')
        heads = heads[edges].clone()
        rels = self.g.edata['tid'][edges].clone()
        if self.has_label:
            labels = self.label
            tails = None
        else:
            tails = tails[edges].clone()
            labels = None
        if self.has_importance:
            impts = self.g.edata['impts'][edges].clone()
        else:
            impts = None
        return PartitionDataset(heads, rels, tails, has_importance=self.has_importance, edge_impts=impts, label=labels)



class EvalDataset(BaseDataset):
    def __init__(self, dataset, args, has_label=True):
        src = [dataset.train[0]]
        etype_id = [dataset.train[1]]
        dst = [dataset.train[2]]
        self.num_train = len(dataset.train[0])
        if dataset.valid is not None:
            src.append(dataset.valid[0])
            etype_id.append(dataset.valid[1])
            dst.append(dataset.valid[2])
            self.num_valid = len(dataset.valid[0])
        else:
            self.num_valid = 0
        if dataset.test is not None:
            src.append(dataset.test[0])
            etype_id.append(dataset.test[1])
            dst.append(dataset.test[2])
            self.num_test = len(dataset.test[0])
        else:
            self.num_test = 0
        assert len(src) > 1, "we need to have at least validation set or test set."
        src = np.concatenate(src)
        etype_id = np.concatenate(etype_id)
        dst = np.concatenate(dst)
        triples = (src, etype_id, dst)

        coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)),
                                   shape=[dataset.n_entities, dataset.n_entities])
        g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
        g.edata['tid'] = F.tensor(etype_id, F.int64)
        self.g = g
        if has_label:
            self.tail_label = self.create_label(triples, dataset.n_entities, args, 'eval', type='tail')
            self.head_label = self.create_label(triples, dataset.n_entities, args, 'eval', type='head')

        if args.eval_percent < 1:
            self.valid = th.randint(low=0, high=self.num_valid, size=(int(self.num_valid * args.eval_percent),)) + self.num_train
        else:
            self.valid = th.arange(self.num_train, self.num_train + self.num_valid)
        print('|valid|:', len(self.valid))

    def partition(self, rank, world_size):
        edges = self.valid
        step_size = (edges.shape[0] + world_size - 1) // world_size
        beg = step_size * rank
        end = min(step_size * (rank + 1), edges.shape[0])
        edges = edges[beg: end]
        heads, tails = self.g.all_edges(order='eid')
        heads = heads[edges].clone()
        tails = tails[edges].clone()
        rels = self.g.edata['tid'][edges].clone()
        return PartitionDataset(heads, rels, tails)


class TestDataset(BaseDataset):
    def __init__(self, dataset, args, has_label=True):
        src = [dataset.train[0]]
        etype_id = [dataset.train[1]]
        dst = [dataset.train[2]]
        self.num_train = len(dataset.train[0])
        if dataset.valid is not None:
            src.append(dataset.valid[0])
            etype_id.append(dataset.valid[1])
            dst.append(dataset.valid[2])
            self.num_valid = len(dataset.valid[0])
        else:
            self.num_valid = 0
        if dataset.test is not None:
            src.append(dataset.test[0])
            etype_id.append(dataset.test[1])
            dst.append(dataset.test[2])
            self.num_test = len(dataset.test[0])
        else:
            self.num_test = 0
        assert len(src) > 1, "we need to have at least validation set or test set."
        src = np.concatenate(src)
        etype_id = np.concatenate(etype_id)
        dst = np.concatenate(dst)
        triples = (src, etype_id, dst)

        coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)),
                                   shape=[dataset.n_entities, dataset.n_entities])
        g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
        g.edata['tid'] = F.tensor(etype_id, F.int64)
        self.g = g
        if has_label:
            self.tail_label = self.create_label(triples, dataset.n_entities, args, 'eval', type='tail')
            self.head_label = self.create_label(triples, dataset.n_entities, args, 'eval', type='head')

        if args.eval_percent < 1:
            self.test = th.randint(low=0, high=self.num_test, size=(int(self.num_test * args.eval_percent),))
            self.test += self.num_train + self.num_valid
        else:
            self.test = th.arange(self.num_train + self.num_valid, self.g.number_of_edges())
        print('|test|:', len(self.test))

    def partition(self, rank, world_size):
        edges = self.test
        step_size = (edges.shape[0] + world_size - 1) // world_size
        beg = step_size * rank
        end = min(step_size * (rank + 1), edges.shape[0])
        edges = edges[beg: end]
        heads, tails = self.g.all_edges(order='eid')
        heads = heads[edges].clone()
        tails = tails[edges].clone()
        rels = self.g.edata['tid'][edges].clone()
        return PartitionDataset(heads, rels, tails)


class PartitionDataset(Dataset):
    def __init__(self, heads, rels, tails=None, has_importance=False, edge_impts=None, label=None):
        assert (tails is None and label is not None) or (tails is not None and label is None)
        self.has_importance = has_importance
        self.heads = heads
        self.rels = rels
        self.tails = tails
        self.edge_impts = edge_impts
        self.label = label

    def __getitem__(self, index):
        head, rel = self.heads[index], self.rels[index]
        ret = {'head': head,
               'rel': rel}

        if self.label is not None:
            label = self.label[(head, rel)]
            ret['label'] = label
        else:
            tail = self.tails[index]
            ret['tail'] = tail

        if self.has_importance:
            impt = self.impts[index]
            ret['impt'] = impt
        return ret

    def __len__(self):
        return len(self.heads)

class NegDataset(BaseDataset, Dataset):
    def __init__(self, num_of_nodes, batch_size, max_step):
        self.negs = self.create_neg_sample(num_of_nodes, batch_size, max_step)

    def __getitem__(self, index):
        neg = self.negs[index]
        return {'negs': neg}

    def __len__(self):
        return len(self.negs)

    def create_neg_sample(self, num_of_nodes, batch_size, max_step):
        node_ids = np.arange(num_of_nodes)
        neg = []
        num_epoch = (max_step * batch_size + num_of_nodes - 1) // num_of_nodes
        for _ in range(num_epoch):
            neg += [np.random.permutation(node_ids)]
        negs = np.concatenate(neg)[:max_step * batch_size]
        return th.from_numpy(negs)

    def partition(self, rank, world_size) -> th.utils.data.Dataset:
        return copy.deepcopy(self)



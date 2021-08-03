from .data import get_dataset, TestDataset
from torch import nn
import time
import dgl
import numpy as np
import torch as th
import argparse
import time
import scipy as sp
import dgl.backend as F

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


class TestDataset(BaseDataset):
    def __init__(self, dataset, args):
        src = [dataset.train[0]]
        etype_id = [dataset.train[1]]
        dst = [dataset.train[2]]
        self.num_train = len(dataset.train[0])
        self.batch_size = args.batch_size_eval
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

        coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)),
                                   shape=[dataset.n_entities, dataset.n_entities])
        g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
        g.edata['tid'] = F.tensor(etype_id, F.int64)
        self.g = g
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
        heads = heads[edges]
        tails = tails[edges]
        rels = self.g.edata['tid'][edges]
        return PartitionEvalDataset(heads, rels, tails)

def create_dataset_graph(args):
    g = None
    dataset = get_dataset(data_path=args.data_path,
                          data_name=args.dataset,
                          format_str=args.format,
                          delimiter=args.delimiter,
                          files=args.data_files,
                          has_edge_importance=args.has_edge_importance,
                          inverse_rel=args.inverse_rel
                          )
    train_dataset, eval_dataset, test_dataset = None, None, None
    # create training dataset needed parameters
    train_dataset = TestDataset(dataset, args)
    g = train_dataset.g
    args.strict_rel_part = args.mix_cpu_gpu and (train_dataset.cross_part is False)
    args.rel_parts = train_dataset.rel_parts if args.strict_rel_part else None
    args.n_entities = dataset.n_entities
    args.n_relations = dataset.n_relations

    return [train_dataset, eval_dataset, test_dataset], g

def partition_graph():
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument('--num_parts', type=int, default=2,
                           help='number of partitions')
    argparser.add_argument('--part_method', type=str, default='metis',
                           help='the partition method')
    argparser.add_argument('--balance_train', action='store_true',
                           help='balance the training size in each partition.')
    argparser.add_argument('--undirected', action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument('--balance_edges', action='store_true',
                           help='balance the number of edges in each partition.')
    argparser.add_argument('--output', type=str, default='data',
                           help='Output path of partitioned graph.')
    argparser.add_argument('--data_path', type=str, default='data',
                      help='The path of the directory where DGL-KE loads knowledge graph data.')
    argparser.add_argument('--dataset', type=str, default='FB15k',
                      help='The name of the builtin knowledge graph. Currently, the builtin knowledge ' \
                           'graphs include FB15k, FB15k-237, wn18, wn18rr and Freebase. ' \
                           'DGL-KE automatically downloads the knowledge graph and keep it under data_path.')
    argparser.add_argument('--format', type=str, default='built_in',
                      help='The format of the dataset. For builtin knowledge graphs,' \
                           'the foramt should be built_in. For users own knowledge graphs,' \
                           'it needs to be raw_udd_{htr} or udd_{htr}.')
    argparser.add_argument('--data_files', type=str, default=None, nargs='+',
                      help='A list of data file names. This is used if users want to train KGE' \
                           'on their own datasets. If the format is raw_udd_{htr},' \
                           'users need to provide train_file [valid_file] [test_file].' \
                           'If the format is udd_{htr}, users need to provide' \
                           'entity_file relation_file train_file [valid_file] [test_file].' \
                           'In both cases, valid_file and test_file are optional.')
    argparser.add_argument('--delimiter', type=str, default='\t',
                      help='Delimiter used in data files. Note all files should use the same delimiter.')

    argparser.add_argument('--mix_cpu_gpu', action='store_true',
                      help='Training a knowledge graph embedding model with both CPUs and GPUs.' \
                           'The embeddings are stored in CPU memory and the training is performed in GPUs.' \
                           'This is usually used for training a large knowledge graph embeddings.')
    argparser.add_argument('--valid', action='store_true',
                      help='Evaluate the model on the validation set in the training.')
    argparser.add_argument('--eval_interval', type=int, default=1,
                      help='Print evaluation results on the validation dataset every x steps' \
                           'if validation is turned on')
    argparser.add_argument('--rel_part', action='store_true',
                      help='Enable relation partitioning for multi-GPU training.')
    argparser.add_argument('--has_edge_importance', action='store_true',
                      help='Allow providing edge importance score for each edge during training.' \
                           'The positive score will be adjusted ' \
                           'as pos_score = pos_score * edge_importance')
    argparser.add_argument('--inverse_rel', action='store_true', dest='inverse_rel',
                      help='If specified, create a->inv_rel->b  based on b->rel->a.')
    argparser.add_argument('--num_proc', type=int, default=1,
                      help='The number of processes to train the model in parallel.' \
                           'In multi-GPU training, the number of processes by default is set to match the number of GPUs.' \
                           'If set explicitly, the number of processes needs to be divisible by the number of GPUs.')
    argparser.add_argument('--num_thread', type=int, default=1,
                      help='The number of CPU threads to train the model in each process.' \
                           'This argument is used for multiprocessing training.')
    argparser.add_argument('--batch_size', type=int, default=1024,
                      help='The batch size for training.')
    argparser.add_argument('--num_workers', type=int, default=0,
                      help='Number of process to fetch data for training/validation dataset.')
    argparser.add_argument('--max_step', type=int, default=80000,
                      help='The maximal number of steps to train the model.' \
                           'A step trains the model with a batch of data.')
    argparser.add_argument('--batch_size_eval', type=int, default=8,
                      help='The batch size used for validation and test.')
    argparser.add_argument('--neg_sample_size', type=int, default=256,
                      help='The number of negative samples we use for each positive sample in the training.')
    argparser.add_argument('--neg_deg_sample', action='store_true',
                      help='Construct negative samples proportional to vertex degree in the training.' \
                           'When this option is turned on, the number of negative samples per positive edge' \
                           'will be doubled. Half of the negative samples are generated uniformly while' \
                           'the other half are generated proportional to vertex degree.')
    argparser.add_argument('--neg_deg_sample_eval', action='store_true',
                      help='Construct negative samples proportional to vertex degree in the evaluation.')
    argparser.add_argument('--neg_sample_size_eval', type=int, default=-1,
                      help='The number of negative samples we use to evaluate a positive sample.')
    argparser.add_argument('--eval_percent', type=float, default=1,
                      help='Randomly sample some percentage of edges for evaluation.')
    argparser.add_argument('--no_eval_filter', action='store_false', dest='eval_filter',
                      help='Disable filter positive edges from randomly constructed negative edges for evaluation')
    argparser.add_argument('--self_loop_filter', action='store_true', dest='self_loop_filter',
                      help='Disable filter triple like (head - relation - head) score for evaluation')
    argparser.add_argument('--test', action='store_true',
                      help='Evaluate the model on the test set after the model is trained.')
    
    
    args = argparser.parse_args()

    start = time.time()

    dataset, g = create_dataset_graph(args)

    print('load {} takes {:.3f} seconds'.format(args.dataset, time.time() - start))
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))
    dgl.distributed.partition_graph(g, args.dataset, args.num_parts, args.output,
                                    part_method=args.part_method,
                                    balance_ntypes=None,
                                    balance_edges=args.balance_edges)
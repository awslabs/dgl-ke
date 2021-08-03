import os
os.environ['DGLBACKEND']='pytorch'
from multiprocessing import Process
import argparse, time, math
import numpy as np
from functools import wraps
import tqdm
import random

import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data.utils import load_graphs
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.distributed import DistDataLoader

import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from .utils.argparser import TrainArgParser
from .utils.misc import prepare_args
from .utils.logging import Logger
from .data import get_dataset, TrainDataset, TestDataset, ValidDataset
from .data.dataloader import KGETrainDataLoaderGenerator, KGEEvalDataLoaderGenerator
from .utils import EMB_INIT_EPS
from .nn.modules import KGEEncoder, TransREncoder
from .nn.modules import KGEDecoder, AttHDecoder, TransRDecoder
from .nn.loss import sLCWAKGELossGenerator
from .nn.loss import BCELoss, HingeLoss, LogisticLoss, LogsigmoidLoss
from .regularizer import Regularizer
from .nn.modules import TransEScore, TransRScore, DistMultScore, ComplExScore, RESCALScore, RotatEScore, SimplEScore
from .nn.metrics import RankingMetricsEvaluator
from functools import partial
import torch as th
from torch import nn
import time
from .nn.modules import KEModel

def dist_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, help='get rank of the process')
    parser.add_argument('--standalone', action='store_true', help='run in the standalone mode')
    args = parser.parse_args()

    dgl.distributed.initialize('ip_config.txt')
    g = dgl.distributed.DistGraph('FB15k', part_config='data/FB15k.json')

    if not args.standalone:
        th.distributed.init_process_group(backend='gloo')

    pb = g.get_partition_book()

    train_mask = th.cat((th.ones((483142,), dtype=th.bool), th.zeros((59071,), dtype=th.bool)))
    test_mask = th.cat((th.zeros((483142,), dtype=th.bool), th.ones((59071,), dtype=th.bool)))

    train_eids = dgl.distributed.edge_split(train_mask, pb, force_even=True)
    test_eids = dgl.distributed.edge_split(test_mask, pb, force_even=True)
    nids = pb.partid2nids(pb.partid)

    emb_init = (143.0 + 1.0) / 200
    init_func = [partial(th.nn.init.uniform_, a=-emb_init, b=emb_init), partial(th.nn.init.uniform_, a=-emb_init, b=emb_init)]
    encoder = KGEEncoder(hidden_dim=200,
                             n_entity=g.num_nodes(),
                             n_relation=1345, #need to get numbers of relations here
                             init_func=init_func,
                             score_func='TransE')

    
    emb_init = (143.0 + 2.0) / 200
    score_func = TransEScore(143.0, dist_func='l2')
    loss_gen = sLCWAKGELossGenerator(neg_adversarial_sampling=True,
                                         adversarial_temperature=1.0,
                                         pairwise=False,
                                         label_smooth=.0)
    criterion = LogsigmoidLoss()

    loss_gen.set_criterion(criterion)
    metrics_evaluator = RankingMetricsEvaluator(False)
    decoder = KGEDecoder('TransE',
                         score_func,
                         loss_gen,
                         metrics_evaluator)
    
    if not args.standalone:
        encoder = th.nn.parallel.DistributedDataParallel(encoder)
    

    def sample(eids):
        s, d = g.find_edges(eids)
        s = th.tensor(s)
        d = th.tensor(d)
        rel = g.edata['tid'][eids]
        rel = th.tensor(rel)
        neg = random.sample(list(nids.numpy()), 128)
        neg = th.tensor(neg)
        data = {'head' : s, 
                'tail' : d, 
                'rel' : rel, 
                'neg' : neg}
        return data

    dataloader = DistDataLoader(
        dataset=train_eids.numpy(),
        batch_size=512,
        collate_fn=sample,
        shuffle=True,
        drop_last=False)

    emb_optimizer = th.optim.SparseAdam(list(encoder.parameters()), lr=1e-1)
    print('optimize Pytorch sparse embedding:', encoder)

    regularizer = Regularizer(coef=2e-9, norm=3)


    for epoch in range(10):
        for step, data in enumerate(dataloader):
            encode_results = encoder.forward(data, -1)
            data['neg_type'] = 'head'
            data['chunk_size'] = data['head'].shape[0]
            data['neg_sample_size'] = 128
            decode_results = decoder.forward(encode_results, data, -1)
            loss = {}
            loss['decode'] = decoder.get_loss(decode_results)
            #loss['reg'] = regularizer.compute_regularization(encode_results)

            total_loss = 0
            for k, v in loss.items():
                total_loss += v

            emb_optimizer.zero_grad()
            total_loss.backward()
            emb_optimizer.step()
            if step % 100 == 0:
                print(loss.items())
    
    def test_sample(eids):
        s, d = g.find_edges(eids)
        s = th.tensor(s)
        d = th.tensor(d)
        rel = g.edata['tid'][eids]
        rel = th.tensor(rel)
        neg = random.sample(list(nids.numpy()), 128)
        neg = th.tensor(neg)
        data = {'head' : s, 
                'tail' : d, 
                'rel' : rel, 
                'neg' : neg}
        return data

    test_dataloader = DistDataLoader(
        dataset=test_eids.numpy(),
        batch_size=16,
        collate_fn=test_sample,
        shuffle=False,
        drop_last=False)
    

    with th.no_grad():
        logs = []
        for step, data in enumerate(test_dataloader):
            data['chunk_size'] = data['head'].shape[0]
            data['neg_sample_size'] = 128
            data['neg_type'] = 'head_tail'
            encode_results = encoder.forward(data, -1)
            results = decoder.forward(encode_results, data, -1)
            logs += decoder.evaluate(results, data, g)
        metrics = {}
        if len(logs) > 0:
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        for k, v in metrics.items():
            print('average {}: {}'.format(k, v))
    
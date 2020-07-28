# -*- coding: utf-8 -*-
#
# train_pytorch.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.optim as optim
import torch as th

from distutils.version import LooseVersion
TH_VERSION = LooseVersion(th.__version__)
if TH_VERSION.version[0] == 1 and TH_VERSION.version[1] < 2:
    raise Exception("DGL-ke has to work with Pytorch version >= 1.2")
from .models.pytorch.tensor_models import thread_wrapped_func
from .models import KEModel
from .utils import save_model, get_compatible_batch_size

import os
import logging
import time
from functools import wraps

import dgl
from dgl.contrib import KVClient
import dgl.backend as F

from .dataloader import EvalDataset
from .dataloader import get_dataset

class KGEClient(KVClient):
    """User-defined kvclient for DGL-KGE
    """
    def _push_handler(self, name, ID, data, target):
        """Row-Sparse Adagrad updater
        """
        original_name = name[0:-6]
        state_sum = target[original_name+'_state-data-']
        grad_sum = (data * data).mean(1)
        state_sum.index_add_(0, ID, grad_sum)
        std = state_sum[ID]  # _sparse_mask
        std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
        tmp = (-self.clr * data / std_values)
        target[name].index_add_(0, ID, tmp)


    def set_clr(self, learning_rate):
        """Set learning rate
        """
        self.clr = learning_rate


    def set_local2global(self, l2g):
        self._l2g = l2g


    def get_local2global(self):
        return self._l2g


def connect_to_kvstore(args, entity_pb, relation_pb, l2g):
    """Create kvclient and connect to kvstore service
    """
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)

    my_client = KGEClient(server_namebook=server_namebook)

    my_client.set_clr(args.lr)

    my_client.connect()

    if my_client.get_id() % args.num_client == 0:
        my_client.set_partition_book(name='entity_emb', partition_book=entity_pb)
        my_client.set_partition_book(name='relation_emb', partition_book=relation_pb)
    else:
        my_client.set_partition_book(name='entity_emb')
        my_client.set_partition_book(name='relation_emb')

    my_client.set_local2global(l2g)

    return my_client

def load_model(args, n_entities, n_relations, ckpt=None):
    model = KEModel(args, args.model_name, n_entities, n_relations,
                    args.hidden_dim, args.gamma,
                    double_entity_emb=args.double_ent, double_relation_emb=args.double_rel)
    if ckpt is not None:
        assert False, "We do not support loading model emb for genernal Embedding"
    return model

def load_model_from_checkpoint(args, n_entities, n_relations, ckpt_path):
    model = load_model(args, n_entities, n_relations)
    model.load_emb(ckpt_path, args.dataset)
    return model

def train(args, model, train_sampler, valid_samplers=None, rank=0, rel_parts=None, cross_rels=None, barrier=None, client=None):
    logs = []
    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.async_update:
        model.create_async_update()
    if args.strict_rel_part or args.soft_rel_part:
        model.prepare_relation(th.device('cuda:' + str(gpu_id)))
    if args.soft_rel_part:
        model.prepare_cross_rels(cross_rels)

    train_start = start = time.time()
    sample_time = 0
    update_time = 0
    forward_time = 0
    backward_time = 0
    for step in range(0, args.max_step):
        start1 = time.time()
        pos_g, neg_g = next(train_sampler)
        sample_time += time.time() - start1

        if client is not None:
            model.pull_model(client, pos_g, neg_g)

        start1 = time.time()
        loss, log = model.forward(pos_g, neg_g, gpu_id)
        forward_time += time.time() - start1

        start1 = time.time()
        loss.backward()
        backward_time += time.time() - start1

        start1 = time.time()
        if client is not None:
            model.push_gradient(client)
        else:
            model.update(gpu_id)
        update_time += time.time() - start1
        logs.append(log)

        # force synchronize embedding across processes every X steps
        if args.force_sync_interval > 0 and \
            (step + 1) % args.force_sync_interval == 0:
            barrier.wait()

        if (step + 1) % args.log_interval == 0:
            if (client is not None) and (client.get_machine_id() != 0):
                pass
            else:
                for k in logs[0].keys():
                    v = sum(l[k] for l in logs) / len(logs)
                    print('[proc {}][Train]({}/{}) average {}: {}'.format(rank, (step + 1), args.max_step, k, v))
                logs = []
                print('[proc {}][Train] {} steps take {:.3f} seconds'.format(rank, args.log_interval,
                                                                time.time() - start))
                print('[proc {}]sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                    rank, sample_time, forward_time, backward_time, update_time))
                sample_time = 0
                update_time = 0
                forward_time = 0
                backward_time = 0
                start = time.time()

        if args.valid and (step + 1) % args.eval_interval == 0 and step > 1 and valid_samplers is not None:
            valid_start = time.time()
            if args.strict_rel_part or args.soft_rel_part:
                model.writeback_relation(rank, rel_parts)
            # forced sync for validation
            if barrier is not None:
                barrier.wait()
            test(args, model, valid_samplers, rank, mode='Valid')
            print('[proc {}]validation take {:.3f} seconds:'.format(rank, time.time() - valid_start))
            if args.soft_rel_part:
                model.prepare_cross_rels(cross_rels)
            if barrier is not None:
                barrier.wait()

    print('proc {} takes {:.3f} seconds'.format(rank, time.time() - train_start))
    if args.async_update:
        model.finish_async_update()
    if args.strict_rel_part or args.soft_rel_part:
        model.writeback_relation(rank, rel_parts)

def test(args, model, test_samplers, rank=0, mode='Test', queue=None):
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.strict_rel_part or args.soft_rel_part:
        model.load_relation(th.device('cuda:' + str(gpu_id)))

    with th.no_grad():
        logs = []
        for sampler in test_samplers:
            for pos_g, neg_g in sampler:
                model.forward_test(pos_g, neg_g, logs, gpu_id)

        metrics = {}
        if len(logs) > 0:
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        if queue is not None:
            queue.put(logs)
        else:
            for k, v in metrics.items():
                print('[{}]{} average {}: {}'.format(rank, mode, k, v))
    test_samplers[0] = test_samplers[0].reset()
    test_samplers[1] = test_samplers[1].reset()

@thread_wrapped_func
def train_mp(args, model, train_sampler, valid_samplers=None, rank=0, rel_parts=None, cross_rels=None, barrier=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    train(args, model, train_sampler, valid_samplers, rank, rel_parts, cross_rels, barrier)

@thread_wrapped_func
def test_mp(args, model, test_samplers, rank=0, mode='Test', queue=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    test(args, model, test_samplers, rank, mode, queue)

@thread_wrapped_func
def dist_train_test(args, model, train_sampler, entity_pb, relation_pb, l2g, rank=0, rel_parts=None, cross_rels=None, barrier=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)

    client = connect_to_kvstore(args, entity_pb, relation_pb, l2g)
    client.barrier()
    train_time_start = time.time()
    train(args, model, train_sampler, None, rank, rel_parts, cross_rels, barrier, client)
    total_train_time = time.time() - train_time_start
    client.barrier()

    # Release the memory of local model
    model = None

    if (client.get_machine_id() == 0) and (client.get_id() % args.num_client == 0): # pull full model from kvstore
        # Pull model from kvstore
        args.num_test_proc = args.num_client
        dataset_full = dataset = get_dataset(args.data_path,
                                             args.dataset,
                                             args.format,
                                             args.delimiter,
                                             args.data_files)
        args.train = False
        args.valid = False
        args.test = True
        args.strict_rel_part = False
        args.soft_rel_part = False
        args.async_update = False

        args.eval_filter = not args.no_eval_filter
        if args.neg_deg_sample_eval:
            assert not args.eval_filter, "if negative sampling based on degree, we can't filter positive edges."

        print('Full data n_entities: ' + str(dataset_full.n_entities))
        print("Full data n_relations: " + str(dataset_full.n_relations))

        eval_dataset = EvalDataset(dataset_full, args)

        if args.neg_sample_size_eval < 0:
            args.neg_sample_size_eval = dataset_full.n_entities
        args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval, args.neg_sample_size_eval)

        model_test = load_model(args, dataset_full.n_entities, dataset_full.n_relations)

        print("Pull relation_emb ...")
        relation_id = F.arange(0, model_test.n_relations)
        relation_data = client.pull(name='relation_emb', id_tensor=relation_id)
        model_test.relation_emb.emb[relation_id] = relation_data
 
        print("Pull entity_emb ... ")
        # split model into 100 small parts
        start = 0
        percent = 0
        entity_id = F.arange(0, model_test.n_entities)
        count = int(model_test.n_entities / 100)
        end = start + count
        while True:
            print("Pull model from kvstore: %d / 100 ..." % percent)
            if end >= model_test.n_entities:
                end = -1
            tmp_id = entity_id[start:end]
            entity_data = client.pull(name='entity_emb', id_tensor=tmp_id)
            model_test.entity_emb.emb[tmp_id] = entity_data
            if end == -1:
                break
            start = end
            end += count
            percent += 1
    
        if not args.no_save_emb:
            print("save model to %s ..." % args.save_path)
            save_model(args, model_test)

        print('Total train time {:.3f} seconds'.format(total_train_time))

        if args.test:
            model_test.share_memory()
            start = time.time()
            test_sampler_tails = []
            test_sampler_heads = []
            for i in range(args.num_test_proc):
                test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                                args.neg_sample_size_eval,
                                                                args.neg_sample_size_eval,
                                                                args.eval_filter,
                                                                mode='chunk-head',
                                                                num_workers=args.num_workers,
                                                                rank=i, ranks=args.num_test_proc)
                test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                                args.neg_sample_size_eval,
                                                                args.neg_sample_size_eval,
                                                                args.eval_filter,
                                                                mode='chunk-tail',
                                                                num_workers=args.num_workers,
                                                                rank=i, ranks=args.num_test_proc)
                test_sampler_heads.append(test_sampler_head)
                test_sampler_tails.append(test_sampler_tail)

            eval_dataset = None
            dataset_full = None

            print("Run test, test processes: %d" % args.num_test_proc)

            queue = mp.Queue(args.num_test_proc)
            procs = []
            for i in range(args.num_test_proc):
                proc = mp.Process(target=test_mp, args=(args,
                                                        model_test,
                                                        [test_sampler_heads[i], test_sampler_tails[i]],
                                                        i,
                                                        'Test',
                                                        queue))
                procs.append(proc)
                proc.start()

            total_metrics = {}
            metrics = {}
            logs = []
            for i in range(args.num_test_proc):
                log = queue.get()
                logs = logs + log
            
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

            print("-------------- Test result --------------")
            for k, v in metrics.items():
                print('Test average {} : {}'.format(k, v))
            print("-----------------------------------------")

            for proc in procs:
                proc.join()

            print('testing takes {:.3f} seconds'.format(time.time() - start))

        client.shut_down() # shut down kvserver

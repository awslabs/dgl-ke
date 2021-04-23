# -*- coding: utf-8 -*-
#
# misc.py
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

import math
import os
import csv
import json
import numpy as np
import torch as th
import glob

to_device = lambda x, gpu_id: x.to(th.device('cpu')) if gpu_id == -1 else x.to(th.device('cuda:%d' % gpu_id))
none = lambda x: x
norm = lambda x, p: x.norm(p=p) ** p
get_scalar = lambda x: x.detach().item() if type(x) is th.Tensor else x
reshape = lambda arr, x, y: arr.view(x, y)

def get_device(args):
    return th.device('cpu') if args.gpu[0] < 0 else th.device('cuda:' + str(args.gpu[0]))

def to_tensor(part):
    if type(part) is list:
        return [to_tensor(p) for p in part]
    elif type(part) is np.ndarray:
        return th.from_numpy(part)
    else:
        return part

def get_compatible_batch_size(batch_size, neg_sample_size):
    if neg_sample_size < batch_size and batch_size % neg_sample_size != 0:
        old_batch_size = batch_size
        batch_size = int(math.ceil(batch_size / neg_sample_size) * neg_sample_size)
        print('batch size ({}) is incompatible to the negative sample size ({}). Change the batch size to {}'.format(
            old_batch_size, neg_sample_size, batch_size))
    return batch_size

def save_model(args, model, emap_file=None, rmap_file=None):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    print('Save model to {}'.format(args.save_path))
    model.save_emb(args.save_path, args.dataset)

    # We need to save the model configurations as well.
    conf_file = os.path.join(args.save_path, 'config.json')
    dict = {}
    config = args
    dict.update(vars(config))
    dict.update({'emp_file': emap_file,
                 'rmap_file': rmap_file})
    with open(conf_file, 'w') as outfile:
        json.dump(dict, outfile, indent=4)

def load_model_config(config_f):
    print(config_f)
    with open(config_f, "r") as f:
        config = json.loads(f.read())
        #config = json.load(f)

    print(config)
    return config

def load_raw_triplet_data(head_f=None, rel_f=None, tail_f=None, emap_f=None, rmap_f=None):
    if emap_f is not None:
        eid_map = {}
        id2e_map = {}
        with open(emap_f, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                eid_map[row[1]] = int(row[0])
                id2e_map[int(row[0])] = row[1]

    if rmap_f is not None:
        rid_map = {}
        id2r_map = {}
        with open(rmap_f, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                rid_map[row[1]] = int(row[0])
                id2r_map[int(row[0])] = row[1]

    if head_f is not None:
        head = []
        with open(head_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                head.append(eid_map[id[:-1]])
                id = f.readline()
        head = np.asarray(head)
    else:
        head = None

    if rel_f is not None:
        rel = []
        with open(rel_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                rel.append(rid_map[id[:-1]])
                id = f.readline()
        rel = np.asarray(rel)
    else:
        rel = None

    if tail_f is not None:
        tail = []
        with open(tail_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                tail.append(eid_map[id[:-1]])
                id = f.readline()
        tail = np.asarray(tail)
    else:
        tail = None

    return head, rel, tail, id2e_map, id2r_map

def load_triplet_data(head_f=None, rel_f=None, tail_f=None):
    if head_f is not None:
        head = []
        with open(head_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                head.append(int(id))
                id = f.readline()
        head = np.asarray(head)
    else:
        head = None

    if rel_f is not None:
        rel = []
        with open(rel_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                rel.append(int(id))
                id = f.readline()
        rel = np.asarray(rel)
    else:
        rel = None

    if tail_f is not None:
        tail = []
        with open(tail_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                tail.append(int(id))
                id = f.readline()
        tail = np.asarray(tail)
    else:
        tail = None

    return head, rel, tail

def load_raw_emb_mapping(map_f):
    assert map_f is not None
    id2e_map = {}
    with open(map_f, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            id2e_map[int(row[0])] = row[1]

    return id2e_map


def load_raw_emb_data(file, map_f=None, e2id_map=None):
    if map_f is not None:
        e2id_map = {}
        id2e_map = {}
        with open(map_f, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                e2id_map[row[1]] = int(row[0])
                id2e_map[int(row[0])] = row[1]
    elif e2id_map is not None:
        id2e_map = [] # dummpy return value
    else:
        assert False, 'There should be an ID mapping file provided'

    ids = []
    with open(file, 'r') as f:
        line = f.readline()
        while len(line) > 0:
            ids.append(e2id_map[line[:-1]])
            line = f.readline()
        ids = np.asarray(ids)

    return ids, id2e_map, e2id_map

def load_entity_data(file=None):
    if file is None:
        return None

    entity = []
    with open(file, 'r') as f:
        id = f.readline()
        while len(id) > 0:
            entity.append(int(id))
            id = f.readline()
    entity = np.asarray(entity)
    return entity

def prepare_args(args):
    prepare_save_path(args)
    if len(args.gpu) > 1:
        args.num_test_proc = args.num_proc if args.num_proc < len(args.gpu) else len(args.gpu)
    else:
        args.num_test_proc = args.num_proc

def prepare_save_path(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    folder = '{}_{}_'.format(args.model, args.dataset)
    n = len([x for x in os.listdir(args.save_path) if x.startswith(folder)])
    folder += str(n)
    args.save_path = os.path.join(args.save_path, folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    args.batch_size = get_compatible_batch_size(args.batch_size, args.neg_sample_size)
    args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval, args.neg_sample_size_eval)
    # We should turn on mix CPU-GPU training for multi-GPU training.
    if len(args.gpu) > 1:
        args.mix_cpu_gpu = True
        if args.num_proc < len(args.gpu):
            args.num_proc = len(args.gpu)
    # We need to ensure that the number of processes should match the number of GPUs.
    if len(args.gpu) > 1 and args.num_proc > 1:
        assert args.num_proc % len(args.gpu) == 0, \
            'The number of processes needs to be divisible by the number of GPUs'

def evaluate_best_result(model_name, dataset, save_path, threshold=2):
    file_pattern = '{}/{}_{}_*/result.txt'.format(save_path, model_name, dataset)
    files = glob.glob(file_pattern)
    best_result = None
    best_dir = None
    for file in files:
        dir = file.split('/')[-2]
        with open(file, 'r') as f:
            result = json.load(f)
        if best_result is None:
            best_result = result
            best_dir = dir
            continue
        else:
            cnt = 0
            for k in result.keys():
                if k == 'MR':
                    if result[k] <= best_result[k]:
                        cnt += 1
                else:
                    if result[k] >= best_result[k]:
                        cnt += 1
            if cnt >= threshold:
                best_result = result
                best_dir = dir
    print(f'''{model_name} training on {dataset} best result is in folder {best_dir}\nbest result:''')
    for k, v in best_result.items():
        print(f'{k}: {v}')


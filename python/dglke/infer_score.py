# -*- coding: utf-8 -*-
#
# infer_score.py
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

import os
import time
import argparse

from .utils import load_model_config, load_raw_triplet_data, load_triplet_data
from .infer import ScoreInfer

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()
        self.add_argument('--data_path', type=str, default='data',
                          help='root path of all dataset including id mapping files')
        self.add_argument('--model_path', type=str, default='ckpts',
                          help='the place where to load the model')
        self.add_argument('--format', type=str,
                          help='The format of input data'\
                                'h_r_t: all lists of head, relation and tail are provied\n' \
                                'h_r_*: both lists of head and relation are provided and tail includes all entities\n' \
                                'h_*_t: both lists of head and tail are provied and relation includes all kinds of relations\n' \
                                '*_r_t: both lists of relation and tail are provied and head includes all entities\n' \
                                'h_*_*: only lists of head is provided and both relation and tail include all possible ones\n' \
                                '*_r_*: only lists of relation is provided and both head and tail include all possible ones;\n' \
                                '*_*_t: only lists of tail is provided and both head and relation include all possible ones;\n')
        self.add_argument('--data_files', type=str, default=None, nargs='+',
                          help='A list of data file names. This is used to provide necessary files containing the requried data ' \
                               'according to the format, e.g., for h_r_t, three files are required as h_data, r_data and t_data, ' \
                               'while for h_*_t, two files are required as h_data and t_data')
        self.add_argument('--raw_data', default=False, action='store_true',
                          help='whether the data profiled in data_files is in the raw object naming space or in mapped id space \n' \
                                'If True, the data is in the original naming space and the inference program will do the id translation' \
                                'according to id mapping files generated during the training progress. \n' \
                                'If False, the data is just interger ids and it is assumed that user has already done the id translation')
        self.add_argument('--bcast', type=str, default=None,
                          help='Whether to broadcast topK in a specific side: \n' \
                               'None: do not broadcast, return an universal topK across all results\n' \
                               'head: broadcast at head, return topK for each head\n' \
                               'rel: broadcast at relation, return topK for each relation\n' \
                               'tail: broadcast at tail, return topK for each tail')
        self.add_argument('--topK', type=int, default=10,
                          help='How many results are returned')
        self.add_argument('--score_func', type=str, default='none',
                          help='What kind of score is used in ranking and will be output: \n' \
                                'none: $score = x$ \n'
                                'logsigmoid: $score = log(sigmoid(x))$')
        self.add_argument('--output', type=str, default='result.tsv',
                          help='Where to store the result, should be a single file')
        self.add_argument('--entity_mfile', type=str, default=None,
                          help='Entity ID mapping file.')
        self.add_argument('--rel_mfile', type=str, default=None,
                          help='Relation ID mapping file.')
        self.add_argument('--gpu', type=int, default=-1,
                          help='GPU device to use in inference, -1 means CPU')

def main():
    args = ArgParser().parse_args()

    config = load_model_config(os.path.join(args.model_path, 'config.json'))
    if args.entity_mfile != None:
        config['emap_file'] = os.path.join(self.data_path, self.entity_mfile)
    if args.rel_mfile != None:
        config['rmap_file'] = os.path.join(self.data_path, self.rel_mfile)

    data_files = args.data_files
    # parse input data first
    if args.format == 'h_r_t':
        if args.raw_data:
            head, rel, tail, id2e_map, id2r_map = load_raw_triplet_data(head_f=data_files[0],
                                                                        rel_f=data_files[1],
                                                                        tail_f=data_files[2],
                                                                        emap_f=config['emap_file'],
                                                                        rmap_f=config['rmap_file'])
        else:
            head, rel, tail = load_triplet_data(head_f=data_files[0],
                                                rel_f=data_files[1],
                                                tail_f=data_files[2])

    elif args.format == 'h_r_*':
        if args.raw_data:
            head, rel, tail, id2e_map, id2r_map = load_raw_triplet_data(head_f=data_files[0],
                                                                        rel_f=data_files[1],
                                                                        tail_f=None,
                                                                        emap_f=config['emap_file'],
                                                                        rmap_f=config['rmap_file'])
        else:
            head, rel, tail = load_triplet_data(head_f=data_files[0],
                                                rel_f=data_files[1],
                                                tail_f=None)

    elif args.format == 'h_*_t':
        if args.raw_data:
            head, rel, tail, id2e_map, id2r_map = load_raw_triplet_data(head_f=data_files[0],
                                                                        rel_f=None,
                                                                        tail_f=data_files[2],
                                                                        emap_f=config['emap_file'],
                                                                        rmap_f=config['rmap_file'])
        else:
            head, rel, tail = load_triplet_data(head_f=data_files[0],
                                                rel_f=None,
                                                tail_f=data_files[1])

    elif args.format == '*_r_t':
        if args.raw_data:
            head, rel, tail, id2e_map, id2r_map = load_raw_triplet_data(head_f=None,
                                                                        rel_f=data_files[1],
                                                                        tail_f=data_files[2],
                                                                        emap_f=config['emap_file'],
                                                                        rmap_f=config['rmap_file'])
        else:
            head, rel, tail = load_triplet_data(head_f=None,
                                                rel_f=data_files[0],
                                                tail_f=data_files[1])

    elif args.format == 'h_*_*':
        if args.raw_data:
            head, rel, tail, id2e_map, id2r_map = load_raw_triplet_data(head_f=data_files[0],
                                                                        rel_f=None,
                                                                        tail_f=None,
                                                                        emap_f=config['emap_file'],
                                                                        rmap_f=config['rmap_file'])
        else:
            head, rel, tail = load_triplet_data(head_f=data_files[0],
                                                rel_f=None,
                                                tail_f=None)

    elif args.format == '*_r_*':
        if args.raw_data:
            head, rel, tail, id2e_map, id2r_map = load_raw_triplet_data(head_f=None,
                                                                        rel_f=data_files[1],
                                                                        tail_f=None,
                                                                        emap_f=config['emap_file'],
                                                                        rmap_f=config['rmap_file'])
        else:
            head, rel, tail = load_triplet_data(head_f=None,
                                                rel_f=data_files[0],
                                                tail_f=None)

    elif args.format == '*_*_t':
        if args.raw_data:
            head, rel, tail, id2e_map, id2r_map = load_raw_triplet_data(head_f=None,
                                                                        rel_f=None,
                                                                        tail_f=data_files[2],
                                                                        emap_f=config['emap_file'],
                                                                        rmap_f=config['rmap_file'])
        else:
            head, rel, tail = load_triplet_data(head_f=None,
                                                rel_f=None,
                                                tail_f=data_files[0])

    else:
        assert False, "Unsupported format {}".format(args.format)

    model = ScoreInfer(args.gpu, config, args.model_path, args.score_func)
    model.load_model()
    result = model.topK(head, rel, tail, args.bcast, args.topK)

    with open(args.output, 'w+') as f:
        f.write('head\trel\ttail\tscore\n')
        for res in result:
            hl, rl, tl, sl = res
            hl = hl.tolist()
            rl = rl.tolist()
            tl = tl.tolist()
            sl = sl.tolist()

            for h, r, t, s in zip(hl, rl, tl, sl):
                if args.raw_data:
                    h = id2e_map[h]
                    r = id2r_map[r]
                    t = id2e_map[t]
                f.write('{}\t{}\t{}\t{}\n'.format(h, r, t, s))
    print('Inference Done')

if __name__ == '__main__':
    main()
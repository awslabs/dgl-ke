# -*- coding: utf-8 -*-
#
# infer_emb.py
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

from .utils import load_entity_data, load_raw_emb_data, load_raw_emb_mapping
from .models.infer import EmbSimInfor

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()
        self.add_argument('--mfile', type=str, default=None,
                          help='ID mapping file.')
        self.add_argument('--emb_file', type=str, default=None,
                          help='Numpy file containing the embeddings. Can be omitted if model_path is provided')
        self.add_argument('--format', type=str,
                          help='The format of input data'\
                                'e_e_pw: two list of objects are provided, and we will calculate the similarity pair by pair \n'
                                'e_e: two list of objects are provided, and we will calculate the similarity in an N x N manner\n' \
                                'e_*: only one list of objects is provided and we will calculate similarity between objects in e ' \
                                'and all objects in the KG in an N_e x N_* manner \n'
                                '*: treat all objects as input and calculate similarity in an N_* x N_* manner')
        self.add_argument('--data_files', type=str, default=None, nargs='+',
                          help='A list of data file names. This is used to provide necessary files containing the requried data ' \
                               'according to the format, e.g., for e_e_pw, two files are required as left_data and right_data; ' \
                               'for e_*, only one file is required; for *, no file is required')
        self.add_argument('--raw_data', default=False, action='store_true',
                          help='whether the data profiled in data_files is in the raw object naming space or in mapped id space \n' \
                                'If True, the data is in the original naming space and the inference program will do the id translation' \
                                'according to id mapping files generated during the training progress. \n' \
                                'If False, the data is just interger ids and it is assumed that user has already done the id translation')
        self.add_argument('--bcast', default=False, action='store_true',
                          help='Whether to broadcast topK at entity level. \n' \
                               'If False, no broadcast is done \n' \
                               'Otherwise, broadcast at left e')
        self.add_argument('--topK', type=int, default=10,
                          help='How many results are returned')
        self.add_argument('--sim_func', type=str, default='cosine',
                          help='What kind of distance function is used in ranking and will be output: \n' \
                                'cosine: use cosine distance\n' \
                                'l2: use l2 distance \n' \
                                'l1: use l1 distance \n' \
                                'dot: use dot product as distance \n' \
                                'ext_jaccard: use extended jaccard as distance \n')
        self.add_argument('--output', type=str, default='result.tsv',
                          help='Where to store the result, should be a single file')
        self.add_argument('--gpu', type=int, default=-1,
                          help='GPU device to use in inference, -1 means CPU')

def main():
    args = ArgParser().parse_args()
    assert args.emb_file != None, 'emb_file should be provided for entity embeddings'

    data_files = args.data_files
    pair_wise = False
    if args.format == 'e_e_pw':
        if args.raw_data:
            head, id2e_map, e2id_map = load_raw_emb_data(file=data_files[0],
                                               map_f=args.mfile)
            tail, _, _ = load_raw_emb_data(file=data_files[1],
                                           e2id_map=e2id_map)
        else:
            head = load_entity_data(data_files[0])
            tail = load_entity_data(data_files[1])
        args.bcast = False
        pair_wise = True
    elif args.format == 'e_e':
        if args.raw_data:
            head, id2e_map, e2id_map = load_raw_emb_data(file=data_files[0],
                                                         map_f=args.mfile)
            tail, _, _ = load_raw_emb_data(file=data_files[1],
                                           e2id_map=e2id_map)
        else:
            head = load_entity_data(data_files[0])
            tail = load_entity_data(data_files[1])
    elif args.format == 'e_*':
        if args.raw_data:
            head, id2e_map, e2id_map = load_raw_emb_data(file=data_files[0],
                                                         map_f=args.mfile)
        else:
            head = load_entity_data(data_files[0])
        tail = load_entity_data()
    elif args.format == '*':
        if args.raw_data:
            id2e_map = load_raw_emb_mapping(map_f=args.mfile)
        head = load_entity_data()
        tail = load_entity_data()

    model = EmbSimInfor(args.gpu, args.emb_file, args.sim_func)
    model.load_emb()
    result = model.topK(head, tail, bcast=args.bcast, pair_ws=pair_wise, k=args.topK)

    with open(args.output, 'w+') as f:
        f.write('head\ttail\tscore\n')
        for res in result:
            hl, tl, sl = res
            hl = hl.tolist()
            tl = tl.tolist()
            sl = sl.tolist()

            for h, t, s in zip(hl, tl, sl):
                if args.raw_data:
                    h = id2e_map[h]
                    t = id2e_map[t]
                f.write('{}\t{}\t{}\n'.format(h, t, s))
    print('Inference Done')

if __name__ == '__main__':
    main()
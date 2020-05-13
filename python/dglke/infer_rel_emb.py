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

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()
        self.add_argument('--data_path', type=str, default='data',
                          help='root path of all dataset including id mapping files')
        self.add_argument('--model_path', type=str, default='ckpts',
                          help='the place where to load the model')
        self.add_argument('--format', type=str,
                          help='The format of input data'\
                                'r_r_pw: two list of relations are provided, and we will calculate the similarity pair by pair \n'
                                'r_r: two list of relations are provided, and we will calculate the similarity in an N x N manner\n' \
                                'r_*: only one list of relations is provided and we will calculate similarity between relations in r ' \
                                'and all relations in the KG in an N_r x N_* manner \n'
                                '*: treat all relations as input and calculate similarity in an N_* x N_* manner'
        self.add_argument('--data_files', type=str, default=None, nargs='+',
                          help='A list of data file names. This is used to provide necessary files containing the requried data ' \
                               'according to the format, e.g., for r_r_pw, two files are required as left_data and right_data; ' \
                               'for r_*, only one file is required; for *, no file is required'
        self.add_argument('--bcast', default=False, action='store_true',
                          help='Whether to broadcast topK at relation level. \n' \
                               'If False, no broadcast is done \n' \
                               'Otherwise, broadcast at left r'
        self.add_argument('--topK', type=int, default=10,
                          help='How many results are returned')
        self.add_argument('--sim_func' type=str, default='cosine',
                          help='What kind of distance function is used in ranking and will be output: \n' \
                                'cosine: use cosine distance\n'
                                'l2: use l2 distance \n'
                                'l1: use l1 distance')
        self.add_argument('--output', type=str, default='result.tsv',
                          help='Where to store the result, should be a single file')

def main():
    args = ArgParser().parse_args()

if __name__ == '__main__':
    main()
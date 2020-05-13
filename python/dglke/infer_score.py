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
                               'while for h_*_t, two files are required as h_data and t_data'
        self.add_argument('--bcast', type=str, default=None,
                          help='Whether to broadcast topK in a specific side: \n',
                               'none: do not broadcast, return an universal topK across all results\n'
                               'head: broadcast at head, return topK for each head\n'
                               'rel: broadcast at relation, return topK for each relation\n'
                               'tail: broadcast at tail, return topK for each tail')
        self.add_argument('--topK', type=int, default=10,
                          help='How many results are returned')
        self.add_argument('--score_func' type=str, default='L1',
                          help='What kind of score is used in ranking and will be output: \n' \
                                'l1: score = $|x|$ \n'
                                'logsigmoid: score $log(sigmoid(x))$')
        self.add_argument('--output', type=str, default='result.tsv',
                          help='Where to store the result, should be a single file')

def main():
    args = ArgParser().parse_args()

if __name__ == '__main__':
    main()
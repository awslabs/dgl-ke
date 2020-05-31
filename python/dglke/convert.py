# -*- coding: utf-8 -*-
#
# convert.py
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
import argparse

from .dataloader import get_dataset

def main():
    parser = argparse.ArgumentParser(description='Convert knowledge graph format')
    parser.add_argument('--data_path', type=str, default='data',
                        help='The path of the directory where DGL-KE loads knowledge graph data.')
    parser.add_argument('--data_files', type=str, required=True, nargs='+',
                        help='A list of data file names. This is used if users want to train KGE'\
                                'on their own datasets. If the format is raw_udd_{htr},'\
                                'users need to provide train_file [valid_file] [test_file].'\
                                'If the format is udd_{htr}, users need to provide'\
                                'entity_file relation_file train_file [valid_file] [test_file].'\
                                'In both cases, valid_file and test_file are optional.')
    parser.add_argument('--delimiter', type=str, default='\t',
                        help='Delimiter used in data files. Note all files should use the same delimiter.')
    parser.add_argument('--input_format', type=str, default='raw_udd_{htr}',
                        help='The format of the input dataset.')
    parser.add_argument('--output_format', type=str, default='udd_{htr}',
                        help='The format of the output dataset.')
    parser.add_argument('--output_path', type=str, default='data',
                        help='The path of the output files.')
    args = parser.parse_args()

    # load dataset and samplers
    dataset = get_dataset(args.data_path,
                          None,
                          args.input_format,
                          args.delimiter,
                          args.data_files)

    assert args.input_format[0:7] == 'raw_udd'
    assert args.output_format[0:3] == 'udd'

    def write_triplets(output_file, triplets, format):
        with open(os.path.join(args.output_path, output_file), 'w') as f:
            assert format == 'hrt' or format == 'htr', 'Unsupported format'
            if format == 'hrt':
                for h, r, t in zip(triplets[0], triplets[1], triplets[2]):
                    triple = ['{}{}{}{}{}\n'.format(h, args.delimiter, r, args.delimiter, t)]
                    f.writelines(triple)
            elif format == 'htr':
                for h, r, t in zip(triplets[0], triplets[1], triplets[2]):
                    triple = ['{}{}{}{}{}\n'.format(h, args.delimiter, t, args.delimiter, r)]
                    f.writelines(triple)

    write_triplets('train_output.txt', dataset.train, args.output_format[4:])
    if dataset.valid is not None:
        write_triplets('valid_output.txt', dataset.valid, args.output_format[4:])
    if dataset.test is not None:
        write_triplets('test_output.txt', dataset.test, args.output_format[4:])

if __name__ == '__main__':
    main()

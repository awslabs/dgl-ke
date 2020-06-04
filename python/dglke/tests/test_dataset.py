# -*- coding: utf-8 -*-
#
# setup.py
#
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os, unittest
from dataloader import get_dataset

def gen_udd_files(extra_train=None, extra_valid=None, extra_test=None):
    train_f = 'train.tsv'
    valid_f = 'valid.tsv'
    test_f = 'test.tsv'
    emap = 'entity.dict'
    rmap = 'relation.dict'
    with open(train_f, 'w+') as f:
        f.write('1\t0\t0\n')
        f.write('2\t0\t0\n')
        f.write('3\t0\t4\n')
        f.write('4\t0\t0\n')
        if extra_train is not None:
            for l in extra_train:
                f.write(l)

    with open(valid_f, 'w+') as f:
        f.write('2\t0\t1\n')
        if extra_valid is not None:
            for l in extra_valid:
                f.write(l)

    with open(test_f, 'w+') as f:
        f.write('2\t0\t3\n')
        if extra_test is not None:
            for l in extra_test:
                f.write(l)

    with open(emap, 'w+') as f:
        f.write('a\t0\n')
        f.write('b\t1\n')
        f.write('c\t2\n')
        f.write('d\t3\n')
        f.write('d\t4\n')

    with open(rmap, 'w+') as f:
        f.write('A\t0\n')

    return [emap, rmap, train_f, valid_f, test_f]

def cleanup(files):
    for f in files:
        os.remove(f)

class TestUDDDataset(unittest.TestCase):
    def test_udd_noint_triplets(self):
        """test one of (h, r, t) is not an integer
        """
        extra_train = ['deadbeaf\t0\t1\n']
        files = gen_udd_files(extra_train=extra_train)
        with self.assertRaises(ValueError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

        extra_train = ['0\tdeadbeaf\t1\n']
        files = gen_udd_files(extra_train=extra_train)
        with self.assertRaises(ValueError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

        extra_train = ['0\t0\tdeadbeaf\n']
        files = gen_udd_files(extra_train=extra_train)
        with self.assertRaises(ValueError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

        extra_valid = ['deadbeaf\t0\t1\n']
        files = gen_udd_files(extra_valid=extra_valid)
        with self.assertRaises(ValueError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

        extra_test = ['deadbeaf\t0\t1\n']
        files = gen_udd_files(extra_test=extra_test)
        with self.assertRaises(ValueError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

    def test_udd_ofr_nids(self):
        """test one of haed and tail entity ID is larger than number of entities
        """
        extra_train = ['7\t0\t1\n']
        files = gen_udd_files(extra_train=extra_train)
        with self.assertRaises(AssertionError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

        extra_train = ['0\t0\t6\n']
        files = gen_udd_files(extra_train=extra_train)
        with self.assertRaises(AssertionError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

        extra_valid = ['7\t0\t1\n']
        files = gen_udd_files(extra_valid=extra_valid)
        with self.assertRaises(AssertionError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

        extra_test = ['7\t0\t1\n']
        files = gen_udd_files(extra_test=extra_test)
        with self.assertRaises(AssertionError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

    def test_udd_ofr_rids(self):
        """test one of relation ID is larger than number of relations
        """
        extra_train = ['0\t2\t1\n']
        files = gen_udd_files(extra_train=extra_train)
        with self.assertRaises(AssertionError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

        extra_valid = ['0\t2\t1\n']
        files = gen_udd_files(extra_valid=extra_valid)
        with self.assertRaises(AssertionError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

        extra_test = ['0\t2\t1\n']
        files = gen_udd_files(extra_test=extra_test)
        with self.assertRaises(AssertionError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

    def test_udd_error_nids(self):
        """test one of head and tail entity ID < 0
        """
        extra_train = ['-1\t0\t1\n']
        files = gen_udd_files(extra_train=extra_train)
        with self.assertRaises(AssertionError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

        extra_train = ['0\t0\t-1\n']
        files = gen_udd_files(extra_train=extra_train)
        with self.assertRaises(AssertionError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

        extra_valid = ['-1\t0\t1\n']
        files = gen_udd_files(extra_valid=extra_valid)
        with self.assertRaises(AssertionError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

        extra_test = ['-1\t0\t1\n']
        files = gen_udd_files(extra_test=extra_test)
        with self.assertRaises(AssertionError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

    def test_udd_error_rids(self):
        """test one of relation ID < 0
        """
        extra_train = ['0\t-1\t1\n']
        files = gen_udd_files(extra_train=extra_train)
        with self.assertRaises(AssertionError):
            dataset = get_dataset('./',
                                  'udd_test',
                                  'udd_hrt',
                                  '\t',
                                  files)
        cleanup(files)

if __name__ == '__main__':
    unittest.main()
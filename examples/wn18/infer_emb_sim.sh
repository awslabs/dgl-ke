# -*- coding: utf-8 -*-
#
# infer_emb_sim.sh
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

#!/bin/bash

# Here we use only TransE_l2 pretrained model.

# Pair wise embedding similarity using TransE_l2 pretrained model using different similarity functions
dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_r' --data_files head.list tail.list --exec_mode 'pairwise' --gpu 0 --sim_func cosine
dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_r' --data_files head.list tail.list --exec_mode 'pairwise' --gpu 0 --sim_func l2
dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_r' --data_files head.list tail.list --exec_mode 'pairwise' --gpu 0 --sim_func l1
dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_r' --data_files head.list tail.list --exec_mode 'pairwise' --gpu 0 --sim_func dot
dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_r' --data_files head.list tail.list --exec_mode 'pairwise' --gpu 0 --sim_func ext_jaccard

# Embedding similarity using TransE_l2 pretrained model w/ and w/o batched calculate topk for each embeddings of the left object list
dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_r' --data_files head.list tail.list --gpu 0
dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_r' --data_files head.list tail.list --gpu 0 --exec_mode 'batch_left'

# Embedding similarity with the whole entity set as tail using TransE_l2 pretrained model w/ and w/o batched calculate topk for each embeddings of the left object list
dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_*' --data_files head.list --gpu 0
dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_*' --data_files head.list --gpu 0 --exec_mode 'batch_left'

# Embedding similarity with the whole relation set as tail using TransE_l2 pretrained model and original ID space w/ and w/o batched calculate topk for each embeddings of the left object list
dglke_emb_sim --mfile data/wn18/relations.dict --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_relation.npy --format 'l_*' --data_files raw_rel.list --topK 5 --raw_data --gpu 0
dglke_emb_sim --mfile data/wn18/relations.dict --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_relation.npy --format 'l_*' --data_files raw_rel.list --topK 5 --raw_data --gpu 0 --exec_mode 'batch_left'
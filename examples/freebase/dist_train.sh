# -*- coding: utf-8 -*-
#
# dist_train.sh
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

# Partiton data into 4 parts
dglke_partition --dataset Freebase -k 4 --data_path ~/my_task

################## Script Result #################
# part 0 has 24662084 nodes and 78777006 edges. 21677970 nodes and 73585700 edges are inside the partition
# part 1 has 24133605 nodes and 61047280 edges. 21677971 nodes and 57314517 edges are inside the partition
# part 2 has 27253496 nodes and 91204738 edges. 21721814 nodes and 83908110 edges are inside the partition
# part 3 has 22593432 nodes and 73698626 edges. 20976396 nodes and 70204032 edges are inside the partition
# write graph to txt file...
# write graph 0...
# write graph 1...
# write graph 2...
# write graph 3...
# there are 304727650 edges in the graph and 19715291 edge cuts for 4 partitions.
##################################################

# TransE-l2
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 40 --model TransE_l2 --dataset Freebase --data_path ~/my_task --hidden_dim 400 \
--gamma 10.0 --lr 0.1 --batch_size 1000 --neg_sample_size 200 --max_step 12500 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --regularization_coef 1.00E-09 \
--no_save_emb --num_thread 1

################## Script Result #################
# Total train time 1633.196 seconds
# Run test, test processes: 40
# -------------- Test result --------------
# Test average MRR : 0.7647393461883437
# Test average MR : 34.256990746461696
# Test average HITS@1 : 0.7059665994617146
# Test average HITS@3 : 0.8025900408923979
# Test average HITS@10 : 0.8696407141981232
# -----------------------------------------
##################################################

# DistMult
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 40 --model DistMult --dataset Freebase --data_path ~/my_task --hidden_dim 400 \
--gamma 143.0 --lr 0.08 --batch_size 1000 --neg_sample_size 200 --max_step 12500 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --no_save_emb --num_thread 1

################## Script Result #################
# Total train time 1679.641 seconds
# Run test, test processes: 40
# -------------- Test result --------------
# Test average MRR : 0.7698761009336426
# Test average MR : 75.15721779649824
# Test average HITS@1 : 0.7510264743248809
# Test average HITS@3 : 0.7790210031030211
# Test average HITS@10 : 0.8014770302483717
# -----------------------------------------
##################################################

# ComplEx
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 40 --model ComplEx --dataset Freebase --data_path ~/my_task --hidden_dim 400 \
--gamma 143.0 --lr 0.1 --batch_size 1000 --neg_sample_size 200 --max_step 12500 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --no_save_emb --num_thread 1

################## Script Result #################
# Total train time 2293.365 seconds
# Run test, test processes: 40
# -------------- Test result --------------
# Test average MRR : 0.7718485286354038
# Test average MR : 77.837799749405
# Test average HITS@1 : 0.7542992897287946
# Test average HITS@3 : 0.7798451064863372
# Test average HITS@10 : 0.8024167024428878
# -----------------------------------------
##################################################
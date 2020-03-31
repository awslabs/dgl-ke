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

#!/bin/bash

# Partiton data into 4 parts
dglke_partition --dataset wn18 -k 4 --data_path ~/my_task

################## Script Result #################
# part 0 has 12545 nodes and 34664 edges. 10255 nodes and 31722 edges are inside the partition
# part 1 has 10799 nodes and 35796 edges. 10178 nodes and 35018 edges are inside the partition
# part 2 has 12511 nodes and 35991 edges. 10255 nodes and 32977 edges are inside the partition
# part 3 has 12611 nodes and 34991 edges. 10255 nodes and 31858 edges are inside the partition
# write graph to txt file...
# write graph 0...
# write graph 1...
# write graph 2...
# write graph 3...
# there are 141442 edges in the graph and 9867 edge cuts for 4 partitions.
##################################################

# TransE-l1 training
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 4 --model_name TransE_l1 --dataset wn18 --data_path ~/my_task --hidden_dim 512 \
--gamma 12.0 --lr 0.007 --batch_size 2048 --neg_sample_size 128 --max_step 2000 --log_interval 100 \
--batch_size_eval 16 --test -adv --regularization_coef 2.00E-07 --num_thread 1

################## Script Result #################
# Total train time 759.241 seconds
# Run test, test processes: 4
# -------------- Test result --------------
# Test average MRR : 0.8480922131760245
# Test average MR : 136.0093
# Test average HITS@1 : 0.768
# Test average HITS@3 : 0.9277
# Test average HITS@10 : 0.9509
# -----------------------------------------
##################################################

# TransE-l2 training
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 4 --model_name TransE_l2 --dataset wn18 --data_path ~/my_task --hidden_dim 512 \
--gamma 6.0 --lr 0.1 --batch_size 1024 --neg_sample_size 256 --max_step 2000 --log_interval 100 \
--batch_size_eval 16 --test -adv --regularization_coef 0.0000001 --num_thread 1

################## Script Result #################
# Total train time 144.569 seconds
# Run test, test processes: 4
# -------------- Test result --------------
# Test average MRR : 0.7977801978151159
# Test average MR : 85.0437
# Test average HITS@1 : 0.6726
# Test average HITS@3 : 0.921
# Test average HITS@10 : 0.9588
# -----------------------------------------
##################################################

# DistMult training
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 4 --model_name DistMult --dataset wn18 --data_path ~/my_task --hidden_dim 512 \
--gamma 20.0 --lr 0.14 --batch_size 2048 --neg_sample_size 128 --max_step 2000 --log_interval 100 \
--batch_size_eval 16 --test -adv --num_thread 1

################## Script Result #################
# Total train time 275.386 seconds
# Run test, test processes: 4
# -------------- Test result --------------
# Test average MRR : 0.8722113353716384
# Test average MR : 278.5113
# Test average HITS@1 : 0.8169
# Test average HITS@3 : 0.926
# Test average HITS@10 : 0.9396
# -----------------------------------------
##################################################

# ComplEx training
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 4 --model_name ComplEx --dataset wn18 --data_path ~/my_task --hidden_dim 512 \
--gamma 200.0 --lr 0.1 --batch_size 1024 --neg_sample_size 1024 --max_step 2000 --log_interval 100 \
--batch_size_eval 16 --test -adv --num_thread 1

################## Script Result #################
# Total train time 273.489 seconds
# Run test, test processes: 4
# -------------- Test result --------------
# Test average MRR : 0.838040106526406
# Test average MR : 333.8747
# Test average HITS@1 : 0.7969
# Test average HITS@3 : 0.8701
# Test average HITS@10 : 0.9068
# -----------------------------------------
##################################################

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
dglke_partition --dataset FB15k -k 4 --data_path ~/my_task

################## Script Result #################
# part 0 has 10579 nodes and 116786 edges. 3816 nodes and 93297 edges are inside the partition
# part 1 has 7774 nodes and 93084 edges. 3657 nodes and 75846 edges are inside the partition
# part 2 has 11014 nodes and 176169 edges. 3850 nodes and 132733 edges are inside the partition
# part 3 has 9597 nodes and 97103 edges. 3628 nodes and 61835 edges are inside the partition
# write graph to txt file...
# write graph 0...
# write graph 1...
# write graph 2...
# write graph 3...
# there are 483142 edges in the graph and 119431 edge cuts for 4 partitions.
##################################################

# TransE-l1 training
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 16 --model_name TransE_l1 --dataset FB15k --data_path ~/my_task --hidden_dim 400 \
--gamma 16.0 --lr 0.01 --batch_size 1000 --neg_sample_size 200 --max_step 500 --log_interval 100 \
--batch_size_eval 16 --test -adv --regularization_coef 1.00E-07 --num_thread 1

################## Script Result #################
# Total train time 104.566 seconds
# Run test, test processes: 16
# -------------- Test result --------------
# Test average MRR : 0.6917546724443662
# Test average MR : 38.26867667721894
# Test average HITS@1 : 0.5918978855953005
# Test average HITS@3 : 0.7651470264596841
# Test average HITS@10 : 0.8538538369081276
# -----------------------------------------
##################################################

# TransE-l2 training
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 16 --model_name TransE_l2 --dataset FB15k --data_path ~/my_task --hidden_dim 400 \
--gamma 19.9 --lr 0.25 --batch_size 1000 --neg_sample_size 200 --max_step 500 --log_interval 100 \
--batch_size_eval 16 --test -adv --regularization_coef 1.00E-09 --num_thread 1

################## Script Result #################
# Total train time 31.804 seconds
# Run test, test processes: 16
# -------------- Test result --------------
# Test average MRR : 0.645860559655554
# Test average MR : 34.84051395777962
# Test average HITS@1 : 0.5109867786223359
# Test average HITS@3 : 0.7546173249140864
# Test average HITS@10 : 0.8548272417937736
# -----------------------------------------
##################################################

# DistMult training
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 4 --model_name DistMult --dataset FB15k --data_path ~/my_task --hidden_dim 400 \
--gamma 143.0 --lr 0.08 --batch_size 1000 --neg_sample_size 200 --max_step 1000 --log_interval 100 \
--batch_size_eval 16 --test -adv --num_thread 1

################## Script Result #################
# Total train time 57.126 seconds
# Run test, test processes: 4
# -------------- Test result --------------
# Test average MRR : 0.6610339350436203
# Test average MR : 51.85362529836976
# Test average HITS@1 : 0.532096968055391
# Test average HITS@3 : 0.7622014186318159
# Test average HITS@10 : 0.8642988945506255
# -----------------------------------------
##################################################

# ComplEx training
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 4 --model_name ComplEx --dataset FB15k --data_path ~/my_task --hidden_dim 400 \
--gamma 143.0 --lr 0.1 --batch_size 1000 --neg_sample_size 200 --max_step 1000 --log_interval 100 \
--batch_size_eval 16 --test -adv --num_thread 1

################## Script Result #################
# Total train time 65.755 seconds
# Run test, test processes: 4
# -------------- Test result --------------
# Test average MRR : 0.6679197157395993
# Test average MR : 62.525232347514006
# Test average HITS@1 : 0.5678674815053072
# Test average HITS@3 : 0.7378409033197338
# Test average HITS@10 : 0.8363410133568079
# -----------------------------------------
##################################################

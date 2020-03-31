# -*- coding: utf-8 -*-
#
# multi_cpu.sh
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

# TransE_l1 training
dglke_train --model_name TransE_l1 --dataset wn18 --batch_size 2048 --log_interval 100 \
--neg_sample_size 128 --regularization_coef 2e-07 --hidden_dim 512 --gamma 12.0 \
--lr 0.007 --batch_size_eval 16 --test -adv --num_proc 10 --num_thread 1 --max_step 2000

################## Script Result #################
# training takes 925.475359916687 seconds
# -------------- Test result --------------
# Test average MRR : 0.5935077415003989
# Test average MR : 376.3582
# Test average HITS@1 : 0.2642
# Test average HITS@3 : 0.9266
# Test average HITS@10 : 0.9499
# -----------------------------------------
# testing takes 34.608 seconds
##################################################

# TransE_l2 training
dglke_train --model_name TransE_l2 --dataset wn18 --batch_size 1024 --log_interval 100 \
--neg_sample_size 256 --regularization_coef 0.0000001 --hidden_dim 512 --gamma 6.0 \
--lr 0.1 --batch_size_eval 16 --test -adv --num_proc 10 --num_thread 1 --max_step 2000

################## Script Result #################
# training takes 210.65357303619385 seconds
# -------------- Test result --------------
# Test average MRR : 0.5284641294400955
# Test average MR : 218.3454
# Test average HITS@1 : 0.2594
# Test average HITS@3 : 0.7775
# Test average HITS@10 : 0.9394
# -----------------------------------------
# testing takes 27.303 seconds
##################################################

# DistMult training 
dglke_train --model_name DistMult --dataset wn18 --batch_size 2048 --log_interval 100 \
--neg_sample_size 128 --regularization_coef 1e-06 --hidden_dim 512 --gamma 20.0 \
--lr 0.14 --batch_size_eval 16 --test -adv --num_proc 8 --num_thread 1 --max_step 2500

################## Script Result #################
# training takes 362.3043715953827 seconds
# -------------- Test result --------------
# Test average MRR : 0.791480223501196
# Test average MR : 837.4457
# Test average HITS@1 : 0.675
# Test average HITS@3 : 0.9046
# Test average HITS@10 : 0.9339
# -----------------------------------------
# testing takes 11.053 seconds
##################################################

# ComplEx training
dglke_train --model_name ComplEx --dataset wn18 --batch_size 1024 --log_interval 100 \
--neg_sample_size 1024 --regularization_coef 0.00001 --hidden_dim 512 --gamma 200.0 \
--lr 0.1 --batch_size_eval 16 --test -adv --num_proc 10 --num_thread 1 --max_step 2000

################## Script Result #################
# training takes 281.57570791244507 seconds
# -------------- Test result --------------
# Test average MRR : 0.9047145350870336
# Test average MR : 806.3966
# Test average HITS@1 : 0.881
# Test average HITS@3 : 0.9261
# Test average HITS@10 : 0.9372
# -----------------------------------------
# testing takes 9.119 seconds
##################################################

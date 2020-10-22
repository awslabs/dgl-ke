# -*- coding: utf-8 -*-
#
# multi_gpu.sh
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

# TransE_l1 1 GPU training
dglke_train --model_name TransE_l1 --dataset wn18 --batch_size 2048 --log_interval 1000 \
--neg_sample_size 128 --regularization_coef 2e-07 --hidden_dim 512 --gamma 12.0 \
--lr 0.007 --batch_size_eval 16 --test -adv --gpu 0 --max_step 32000

################## Script Result #################
# training takes 327.0762197971344 seconds
# [0]Test average MRR: 0.7644073490215693
# [0]Test average MR: 355.4182
# [0]Test average HITS@1: 0.602
# [0]Test average HITS@3: 0.9283
# [0]Test average HITS@10: 0.9499
# testing takes 62.344 seconds
##################################################

# TransE_l1 8 GPU training
dglke_train --model_name TransE_l1 --dataset wn18 --batch_size 2048 --log_interval 1000 \
--neg_sample_size 128 --regularization_coef 2e-07 --hidden_dim 512 --gamma 12.0 \
--lr 0.007 --batch_size_eval 16 --test -adv --max_step 4000 --num_thread 4 \
--mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --force_sync_interval 1000

################## Script Result #################
# training takes 111.51463556289673 seconds
# -------------- Test result --------------
# Test average MRR : 0.7396491217914677
# Test average MR : 348.8292
# Test average HITS@1 : 0.5537
# Test average HITS@3 : 0.9277
# Test average HITS@10 : 0.9485
# -----------------------------------------
# testing takes 18.328 seconds
##################################################

# TransE_l2 1 GPU training
dglke_train --model_name TransE_l2 --dataset wn18 --batch_size 1024 --log_interval 1000 \
--neg_sample_size 256 --regularization_coef 0.0000001 --hidden_dim 512 --gamma 6.0 \
--lr 0.1 --batch_size_eval 16 --test -adv --gpu 0 --max_step 32000

################## Script Result #################
# training takes 223.37764406204224 seconds
# [0]Test average MRR: 0.5606213503330412
# [0]Test average MR: 209.4984
# [0]Test average HITS@1: 0.3066
# [0]Test average HITS@3: 0.797
# [0]Test average HITS@10: 0.9436
# testing takes 59.992 seconds
##################################################

# TransE_l2 8 GPU training
dglke_train --model_name TransE_l2 --dataset wn18 --batch_size 1024 --log_interval 1000 \
--neg_sample_size 256 --regularization_coef 0.0000001 --hidden_dim 512 --gamma 6.0 \
--lr 0.1 --batch_size_eval 16 --test -adv --max_step 4000 --num_thread 4 \
--mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --force_sync_interval 1000

################## Script Result #################
# training takes 71.15329003334045 seconds
# -------------- Test result --------------
# Test average MRR : 0.5599572920553614
# Test average MR : 198.9239
# Test average HITS@1 : 0.3059
# Test average HITS@3 : 0.7985
# Test average HITS@10 : 0.9423
# -----------------------------------------
# testing takes 18.086 seconds
##################################################

# DistMult 1 GPU training 
dglke_train --model_name DistMult --dataset wn18 --batch_size 2048 --log_interval 1000 \
--neg_sample_size 128 --regularization_coef 1e-06 --hidden_dim 512 --gamma 20.0 \
--lr 0.14 --batch_size_eval 16 --test -adv --gpu 0 --max_step 20000

################## Script Result #################
# training takes 133.62837743759155 seconds
# [0]Test average MRR: 0.8134953067345918
# [0]Test average MR: 419.0092
# [0]Test average HITS@1: 0.7022
# [0]Test average HITS@3: 0.9216
# [0]Test average HITS@10: 0.9484
# testing takes 57.108 seconds
##################################################

# DistMult 8 GPU training
dglke_train --model_name DistMult --dataset wn18 --batch_size 2048 --log_interval 1000 \
--neg_sample_size 128 --regularization_coef 1e-06 --hidden_dim 512 --gamma 20.0 \
--lr 0.14 --batch_size_eval 16 --test -adv --max_step 2500 --num_thread 4 \
--mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --force_sync_interval 1000

################## Script Result #################
# training takes 66.86845016479492 seconds
# -------------- Test result --------------
# Test average MRR : 0.8064785286339088
# Test average MR : 798.8149
# Test average HITS@1 : 0.705
# Test average HITS@3 : 0.9036
# Test average HITS@10 : 0.932
# -----------------------------------------
# testing takes 16.589 seconds
##################################################

# ComplEx 1 GPU
dglke_train --model_name ComplEx --dataset wn18 --batch_size 1024 --log_interval 1000 \
--neg_sample_size 1024 --regularization_coef 0.00001 --hidden_dim 512 --gamma 200.0 \
--lr 0.1 --batch_size_eval 16 --test -adv --gpu 0 --max_step 20000

################## Script Result #################
# training takes 144.398681640625 seconds
# [0]Test average MRR: 0.9329441975940886
# [0]Test average MR: 318.271
# [0]Test average HITS@1: 0.9145
# [0]Test average HITS@3: 0.9488
# [0]Test average HITS@10: 0.9593
# testing takes 63.667 seconds
##################################################

# ComplEx 8 GPU 
dglke_train --model_name ComplEx --dataset wn18 --batch_size 1024 --log_interval 1000 \
--neg_sample_size 1024 --regularization_coef 0.00001 --hidden_dim 512 --gamma 200.0 \
--lr 0.1 --batch_size_eval 16 --test -adv --max_step 2500 --mix_cpu_gpu \
--num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --force_sync_interval 1000

################## Script Result #################
# training takes 53.70948600769043 seconds
# -------------- Test result --------------
# Test average MRR : 0.9385695872109829
# Test average MR : 535.0447
# Test average HITS@1 : 0.9311
# Test average HITS@3 : 0.9443
# Test average HITS@10 : 0.9491
# -----------------------------------------
# testing takes 17.807 seconds
##################################################

# RESCAL 1 GPU training
dglke_train --model_name RESCAL --dataset wn18 --batch_size 1024 --log_interval 1000 \
--neg_sample_size 256 --hidden_dim 250 --gamma 24.0 --lr 0.03 --batch_size_eval 16 \
--test -adv --gpu 0 --max_step 20000

################## Script Result #################
# training takes 308.8598370552063 seconds
# [0]Test average MRR: 0.848852719307795
# [0]Test average MR: 563.6445
# [0]Test average HITS@1: 0.7921
# [0]Test average HITS@3: 0.8988
# [0]Test average HITS@10: 0.9282
# testing takes 66.937 seconds
##################################################

# TransR 1 GPU training
dglke_train --model_name TransR --dataset wn18 --batch_size 1024 --log_interval 1000 \
--neg_sample_size 256 --hidden_dim 250 --gamma 16.0 --lr 0.1 --batch_size_eval 16 \
--test -adv --gpu 0 --max_step 30000

################## Script Result #################
# training takes 906.1187422275543 seconds
# [0]Test average MRR: 0.609019858347323
# [0]Test average MR: 432.8342
# [0]Test average HITS@1: 0.452
# [0]Test average HITS@3: 0.7367
# [0]Test average HITS@10: 0.8508
# testing takes 70.645 seconds
##################################################

# RotatE 1 GPU training
dglke_train --model_name RotatE --dataset wn18 --batch_size 2048 --log_interval 1000 \
--neg_sample_size 64 --regularization_coef 2e-07 --hidden_dim 256 --gamma 9.0 \
--lr 0.0025 -de --batch_size_eval 16 --neg_deg_sample --test -adv --gpu 0 \
--num_thread 4 --max_step 24000 

################## Script Result #################
# training takes 671.8658378124237 seconds
# [0]Test average MRR: 0.9440866891244517
# [0]Test average MR: 451.6556
# [0]Test average HITS@1: 0.9409
# [0]Test average HITS@3: 0.9453
# [0]Test average HITS@10: 0.9501
# testing takes 89.074 seconds
##################################################

# RotatE 8 GPU training
dglke_train --model_name RotatE --dataset wn18 --batch_size 2048 --log_interval 1000 \
--neg_sample_size 64 --regularization_coef 2e-07 --hidden_dim 256 --gamma 9.0 \
--lr 0.0025 -de --batch_size_eval 16 --neg_deg_sample --test -adv --num_thread 4 \
--max_step 3000 --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
--force_sync_interval 1000

################## Script Result #################
# training takes 127.96926379203796 seconds
# -------------- Test result --------------
# Test average MRR : 0.9434433683762773
# Test average MR : 487.7839
# Test average HITS@1 : 0.9397
# Test average HITS@3 : 0.945
# Test average HITS@10 : 0.9515
# -----------------------------------------
# testing takes 18.805 seconds
##################################################

# SimplE 1 GPU training
dglke_train --model_name SimplE --dataset wn18 --batch_size 2048 --log_interval 1000 \
--neg_sample_size 128 --regularization_coef 2e-06 --hidden_dim 512 --gamma 300.0 \
--lr 0.2 --batch_size_eval 16 --test -adv --gpu 0 --max_step 20000 --double_ent

################## Script Result #################
# training takes 151.13316130638123 seconds
# -------------- Test result --------------
# Test average MRR : 0.9380122609495815
# Test average MR : 370.2417
# Test average HITS@1 : 0.9254
# Test average HITS@3 : 0.9493
# Test average HITS@10 :0.9569
# -----------------------------------------
##################################################

# SimplE 8 GPU training
dglke_train --model_name SimplE --dataset wn18 --batch_size 2048 --log_interval 1000 \
--neg_sample_size 128 --regularization_coef 2e-06 --hidden_dim 512 --gamma 300.0 \
--lr 0.2 --batch_size_eval 16 --test -adv --max_step 2500 --num_thread 4 \
--mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --force_sync_interval 1000 --double_ent --double_rel

################## Script Result #################
# training takes 121.00198698043823 seconds
# -------------- Test result --------------
# Test average MRR : 0.9454892615965981
# Test average MR : 513.4067
# Test average HITS@1 : 0.9403
# Test average HITS@3 : 0.9495
# Test average HITS@10 : 0.9536
# -----------------------------------------
##################################################


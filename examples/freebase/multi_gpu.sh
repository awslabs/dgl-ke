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

# TransE_l2 8 GPU training
dglke_train --model_name TransE_l2 --dataset Freebase --batch_size 1000 \
--neg_sample_size 200 --hidden_dim 400 --gamma 10 --lr 0.1 --regularization_coef 1e-9 \
--batch_size_eval 1000 --test -adv --mix_cpu_gpu --num_proc 8 --num_thread 4 \
--gpu 0 1 2 3 4 5 6 7 --max_step 320000 --neg_sample_size_eval 1000 \
--log_interval 1000 --async_update --rel_part --force_sync_interval 10000 --no_save_emb

################## Script Result #################
# training takes 4767.735562801361 seconds
# -------------- Test result --------------
# Test average MRR : 0.736305062465261
# Test average MR : 23.568136216790432
# Test average HITS@1 : 0.6630866424073565
# Test average HITS@3 : 0.7820537909759808
# Test average HITS@10 : 0.8733898042377161
# -----------------------------------------
# testing takes 665.085 seconds
##################################################

# DistMult 8 GPU training
dglke_train --model_name DistMult --dataset Freebase --batch_size 1024 \
--neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 1000 \
--test -adv --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --max_step 300000 \
--neg_sample_size_eval 1000 --eval_interval 100000 --log_interval 1000 --async_update \
--rel_part --force_sync_interval 10000 --num_thread 4 --no_save_emb

################## Script Result #################
# training takes 4281.121680021286 seconds
# -------------- Test result --------------
# Test average MRR : 0.8334487481335136
# Test average MR : 46.196215610230496
# Test average HITS@1 : 0.8136404335014756
# Test average HITS@3 : 0.8425535172494942
# Test average HITS@10 : 0.8699551098012984
# -----------------------------------------
# testing takes 597.371 seconds
##################################################

# ComplEx 8 GPU training
dglke_train --model_name ComplEx --dataset Freebase --batch_size 1024 \
--neg_sample_size 256 --hidden_dim 400 --gamma 143 --lr 0.1 --num_thread 4 \
--regularization_coef 2.00E-06 --batch_size_eval 1000 --test -adv \
--mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --max_step 360000 \
--neg_sample_size_eval 1000 --log_interval 1000 \
--async_update --rel_part --force_sync_interval 10000 --no_save_emb

################## Script Result #################
# training takes 8356.189607143402 seconds
# -------------- Test result --------------
# Test average MRR : 0.8346977074432389
# Test average MR : 46.706477045606356
# Test average HITS@1 : 0.8158006812800618
# Test average HITS@3 : 0.8436128930963983
# Test average HITS@10 : 0.8692575916274901
# -----------------------------------------
# testing takes 884.900 seconds
##################################################

# TransR 8 GPU training
dglke_train --model_name TransR --dataset Freebase --batch_size 1024 \
--neg_sample_size 256 --regularization_coef 5e-8 --hidden_dim 200 --gamma 8.0 \
--lr 0.015 --batch_size_eval 1000 --test -adv --mix_cpu_gpu --num_proc 8 \
--gpu 0 1 2 3 4 5 6 7 --max_step 300000 --neg_sample_size_eval 1000 \
--log_interval 1000 --async_update --rel_part --num_thread 4 \
--force_sync_interval 10000 --no_save_emb

################## Script Result #################
# training takes 14235.306294202805 seconds
# -------------- Test result --------------
# Test average MRR : 0.6961484987092021
# Test average MR : 49.68131963810925
# Test average HITS@1 : 0.6531622851920469
# Test average HITS@3 : 0.7167594800685296
# Test average HITS@10 : 0.7739315156886507
# -----------------------------------------
# testing takes 730.981 seconds
##################################################

# RotatE 8 GPU training
dglke_train --model_name RotatE --dataset Freebase --batch_size 1024 \
--neg_sample_size 256 -de --hidden_dim 200 --gamma 12.0 --lr 0.01 \
--regularization_coef 1e-7 --batch_size_eval 1000 --test -adv --mix_cpu_gpu \
--num_proc 8 --gpu 0 1 2 3 4 5 6 7 --max_step 300000 --neg_sample_size_eval 1000 \
--log_interval 1000 --async_update --rel_part --num_thread 4 \
--force_sync_interval 10000 --no_save_emb

################## Script Result #################
# training takes 9060.67327284813 seconds
# -------------- Test result --------------
# Test average MRR : 0.7690681320390669
# Test average MR : 93.20243639019385
# Test average HITS@1 : 0.7485689314648891
# Test average HITS@3 : 0.7795547815657912
# Test average HITS@10 : 0.8048694016317737
# -----------------------------------------
# testing takes 785.671 seconds
##################################################
# -*- coding: utf-8 -*-
#
# multi_cpu.sh
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

#!/bin/bash

# TransE-l1 training
dglke_train --model_name TransE_l1 --dataset FB15k --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 \
--gamma 16.0 --lr 0.01 --max_step 500 --log_interval 100 --batch_size_eval 16 --test -adv \
--regularization_coef 1.00E-07 --num_thread 1 --num_proc 48

################## Script Result #################
# training takes 140.13480591773987 seconds
# -------------- Test result --------------
# Test average MRR : 0.6459630135315562
# Test average MR : 48.324702476680606
# Test average HITS@1 : 0.5218719845609521
# Test average HITS@3 : 0.7419461326200674
# Test average HITS@10 : 0.8380338914188011
# -----------------------------------------
# testing takes 27.371 seconds
##################################################

# TransE-l1 eval
dglke_eval --model_name TransE_l1 --dataset FB15k --hidden_dim 400 --gamma 16.0 --batch_size_eval 16 \
--num_thread 1 --num_proc 48 --model_path ~/my_task/ckpts/TransE_l1_FB15k_0/

################## Script Result #################
# -------------- Test result --------------
# Test average MRR : 0.6459630135315507
# Test average MR : 48.324702476680606
# Test average HITS@1 : 0.5218719845609521
# Test average HITS@3 : 0.7419461326200674
# Test average HITS@10 : 0.8380338914188011
# -----------------------------------------
# Test takes 28.029 seconds
##################################################

# TransE-l2 training
dglke_train --model_name TransE_l2 --dataset FB15k --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 \
--gamma 19.9 --lr 0.25 --max_step 500 --log_interval 100 --batch_size_eval 16 --test -adv \
--regularization_coef 1.00E-09 --num_thread 1 --num_proc 48

################## Script Result #################
# training takes 58.7625195980072 seconds
# -------------- Test result --------------
# Test average MRR : 0.6335627270530246
# Test average MR : 45.287459159316754
# Test average HITS@1 : 0.5019468097712921
# Test average HITS@3 : 0.735809449645342
# Test average HITS@10 : 0.8404462426571414
# -----------------------------------------
# testing takes 24.250 seconds
##################################################

# TransE-l2 eval
dglke_eval --model_name TransE_l2 --dataset FB15k --hidden_dim 400 --gamma 19.9 --batch_size_eval 16 \
--num_thread 1 --num_proc 48 --model_path ~/my_task/ckpts/TransE_l2_FB15k_0/

################## Script Result #################
# -------------- Test result --------------
# Test average MRR : 0.6335627270530305
# Test average MR : 45.287459159316754
# Test average HITS@1 : 0.5019468097712921
# Test average HITS@3 : 0.735809449645342
# Test average HITS@10 : 0.8404462426571414
# -----------------------------------------
# Test takes 22.812 seconds
##################################################

# DistMult training
dglke_train --model_name DistMult --dataset FB15k --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 \
--gamma 143.0 --lr 0.08 --max_step 1000 --log_interval 100 --batch_size_eval 16 --test -adv \
--num_thread 1 --num_proc 10

################## Script Result #################
# training takes 58.42871427536011 seconds
# -------------- Test result --------------
# Test average MRR : 0.6475228321640765
# Test average MR : 62.63024157369944
# Test average HITS@1 : 0.5291936821790726
# Test average HITS@3 : 0.7336256369453709
# Test average HITS@10 : 0.8465998544124866
# -----------------------------------------
# testing takes 33.218 seconds
##################################################

# DistMult eval
dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 400 --gamma 143.0 --batch_size_eval 16 \
--num_thread 1 --num_proc 48 --model_path ~/my_task/ckpts/DistMult_FB15k_0/

################## Script Result #################
# -------------- Test result --------------
# Test average MRR : 0.6475228321640787
# Test average MR : 62.63024157369944
# Test average HITS@1 : 0.5291936821790726
# Test average HITS@3 : 0.7336256369453709
# Test average HITS@10 : 0.8465998544124866
# -----------------------------------------
# Test takes 10.347 seconds
##################################################

# ComplEx training
dglke_train --model_name ComplEx --dataset FB15k --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 \
--gamma 143.0 --lr 0.1 --max_step 1000 --log_interval 100 --batch_size_eval 16 --test -adv \
--regularization_coef 2.00E-06 --num_thread 1 --num_proc 10

################## Script Result #################
# training takes 69.87786984443665 seconds
# -------------- Test result --------------
# Test average MRR : 0.6941813635232874
# Test average MR : 67.83826242995717
# Test average HITS@1 : 0.5905943694875658
# Test average HITS@3 : 0.7723417582231552
# Test average HITS@10 : 0.8636048145452083
# -----------------------------------------
# testing takes 34.523 seconds
##################################################

# ComplEx eval
dglke_eval --model_name ComplEx --dataset FB15k --hidden_dim 400 --gamma 143.0 --batch_size_eval 16 \
--num_thread 1 --num_proc 48 --model_path ~/my_task/ckpts/ComplEx_FB15k_0/

################## Script Result #################
# -------------- Test result --------------
# Test average MRR : 0.6941813635232933
# Test average MR : 67.83826242995717
# Test average HITS@1 : 0.5905943694875658
# Test average HITS@3 : 0.7723417582231552
# Test average HITS@10 : 0.8636048145452083
# -----------------------------------------
# Test takes 9.582 seconds
##################################################
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

# TransE-l2
dglke_train --model_name TransE_l2 --dataset Freebase --no_save_emb --log_interval 100 \
--batch_size 1000 --neg_sample_size 200 --hidden_dim 400 --gamma 10 --lr 0.1 --max_step 50000 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --regularization_coef 1e-9 --num_thread 1 --num_proc 48

################## Script Result #################
# training takes 6993.861527442932 seconds
# -------------- Test result --------------
# Test average MRR : 0.8156297683149505
# Test average MR : 30.82207423953773
# Test average HITS@1 : 0.7665608659255299
# Test average HITS@3 : 0.8489767567581616
# Test average HITS@10 : 0.9020039980370137
# -----------------------------------------
# testing takes 275.527 seconds
##################################################

# DistMult
dglke_train --model_name DistMult --dataset Freebase --no_save_emb --log_interval 100 \
--batch_size 1024 --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --max_step 50000 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --num_thread 1 --num_proc 48

################## Script Result #################
# training takes 7146.087579250336 seconds
# -------------- Test result --------------
# Test average MRR : 0.8345510291090454
# Test average MR : 44.16684887533501
# Test average HITS@1 : 0.8152792778062754
# Test average HITS@3 : 0.84313971959161
# Test average HITS@10 : 0.869641098147662
# -----------------------------------------
# testing takes 255.810 seconds
##################################################

# ComplEx
dglke_train --model_name ComplEx --dataset Freebase --no_save_emb --log_interval 100 \
--batch_size 1024 --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.1 --max_step 50000 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --num_thread 1 --num_proc 48

################## Script Result #################
# training takes 8732.690290689468 seconds
# -------------- Test result --------------
# Test average MRR : 0.8358079670934815
# Test average MR : 45.62506095937294
# Test average HITS@1 : 0.8175541492895043
# Test average HITS@3 : 0.8439161836975262
# Test average HITS@10 : 0.8701293638227859
# -----------------------------------------
# testing takes 256.143 seconds
##################################################
# -*- coding: utf-8 -*-
#
# multi_gpu.sh
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

# TransE_l2
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset Freebase --batch_size 1000 \
    --neg_sample_size 200 --hidden_dim 400 --gamma 10 --lr 0.1 --max_step 50000 \
    --log_interval 100 --batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv \
    --regularization_coef 1e-9 --num_thread 1 --num_proc 48

# DistMult
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --max_step 50000 \
    --log_interval 100 --batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv \
    --num_thread 1 --num_proc 48

# ComplEx
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.1 --max_step 50000 \
    --log_interval 100 --batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv \
    --num_thread 1 --num_proc 48

# Freebase multi-gpu
# TransE_l2 8GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset Freebase --batch_size 1000 \
    --neg_sample_size 200 --hidden_dim 400 --gamma 10 --lr 0.1 --regularization_coef 1e-9 \
    --batch_size_eval 1000 --valid --test -adv --mix_cpu_gpu --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --max_step 320000 --neg_sample_size_eval 1000 --eval_interval \
    100000 --log_interval 10000 --async_update --soft_rel_part --force_sync_interval 10000

# DistMult 8GPU
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 1000 \
    --valid --test -adv --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --max_step 300000 \
    --neg_sample_size_eval 1000 --eval_interval 100000 --log_interval 10000 --async_update \
    --soft_rel_part --force_sync_interval 10000

# ComplEx 8GPU
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143 --lr 0.1 \
    --regularization_coef 2.00E-06 --batch_size_eval 1000 --valid --test -adv \
    --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --max_step 360000 \
    --neg_sample_size_eval 1000 --eval_interval 100000 --log_interval 10000 \
    --async_update --soft_rel_part --force_sync_interval 10000

# TransR 8GPU
DGLBACKEND=pytorch python3 train.py --model TransR --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 5e-8 --hidden_dim 200 --gamma 8.0 \
    --lr 0.015 --batch_size_eval 1000 --valid --test -adv --mix_cpu_gpu --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --max_step 300000 --neg_sample_size_eval 1000 \
    --eval_interval 100000 --log_interval 10000 --async_update --soft_rel_part \
    --force_sync_interval 10000

# RotatE 8GPU
DGLBACKEND=pytorch python3 train.py --model RotatE --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 -de --hidden_dim 200 --gamma 12.0 --lr 0.01 \
    --regularization_coef 1e-7 --batch_size_eval 1000 --valid --test -adv --mix_cpu_gpu \
    --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --max_step 300000 --neg_sample_size_eval 1000 \
    --eval_interval 100000 --log_interval 10000 --async_update --soft_rel_part \
    --force_sync_interval 10000
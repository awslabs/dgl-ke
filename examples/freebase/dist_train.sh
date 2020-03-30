# -*- coding: utf-8 -*-
#
# dist_train.sh
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

# Partiton data into 4 parts
dglke_partition --dataset Freebase -k 4 --data_path ~/my_task

# TransE-l2
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 40 --model TransE_l2 --dataset Freebase --data_path ~/my_task --hidden_dim 400 \
--gamma 10.0 --lr 0.1 --batch_size 1000 --neg_sample_size 200 --max_step 12500 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --regularization_coef 1.00E-09 \
--no_save_emb --num_thread 1

# DistMult
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 40 --model DistMult --dataset Freebase --data_path ~/my_task --hidden_dim 400 \
--gamma 143.0 --lr 0.08 --batch_size 1000 --neg_sample_size 200 --max_step 12500 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --no_save_emb --num_thread 1

# ComplEx
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 40 --model ComplEx --dataset Freebase --data_path ~/my_task --hidden_dim 400 \
--gamma 143.0 --lr 0.1 --batch_size 1000 --neg_sample_size 200 --max_step 12500 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --no_save_emb --num_thread 1
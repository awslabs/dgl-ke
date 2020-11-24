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

# TransE_l1 1 GPU training
dglke_train --model_name TransE_l1 --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --regularization_coef 1e-07 --hidden_dim 400 --gamma 16.0 \
--lr 0.01 --batch_size_eval 16 --test -adv --gpu 0 --max_step 24000

################## Script Result #################
# training takes 201.90872430801392 seconds
# [0]Test average MRR: 0.6729707889330725
# [0]Test average MR: 47.344009750977634
# [0]Test average HITS@1: 0.5573716375209493
# [0]Test average HITS@3: 0.7632848605914916
# [0]Test average HITS@10: 0.849740143217484
# testing takes 286.497 seconds
##################################################

# TransE_l1 1 GPU eval
dglke_eval --model_name TransE_l1 --dataset FB15k --hidden_dim 400 --gamma 16.0 --batch_size_eval 16 \
--gpu 0 --model_path ~/my_task/ckpts/TransE_l1_FB15k_1/

################## Script Result #################
# [0]Test average MRR: 0.6729707889330725
# [0]Test average MR: 47.344009750977634
# [0]Test average HITS@1: 0.5573716375209493
# [0]Test average HITS@3: 0.7632848605914916
# [0]Test average HITS@10: 0.849740143217484
# Test takes 280.351 seconds
##################################################

# TransE_l1 8 GPU training
dglke_train --model_name TransE_l1 --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --regularization_coef 1e-07 --hidden_dim 400 --gamma 16.0 \
--lr 0.01 --batch_size_eval 16 --test -adv --max_step 3000 --mix_cpu_gpu --num_thread 4 \
--num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --force_sync_interval 1000

################## Script Result #################
# training takes 53.92943000793457 seconds
# -------------- Test result --------------
# Test average MRR : 0.6629663821980686
# Test average MR : 48.592930541213114
# Test average HITS@1 : 0.5429059944812176
# Test average HITS@3 : 0.7568096020043676
# Test average HITS@10 : 0.8466252475834166
# -----------------------------------------
# testing takes 47.976 seconds
##################################################

# TransE_l1 8 GPU eval
dglke_eval --model_name TransE_l1 --dataset FB15k --hidden_dim 400 --gamma 16.0 --batch_size_eval 16 \
--gpu 0 1 2 3 4 5 6 7 --model_path ~/my_task/ckpts/TransE_l1_FB15k_0/

################## Script Result #################
# -------------- Test result --------------
# Test average MRR: 0.6629663821980687
# Test average MR: 48.592930541213114
# Test average HITS@1: 0.5429059944812176
# Test average HITS@3: 0.7568096020043676
# Test average HITS@10: 0.8466252475834166
# -----------------------------------------
# Test takes 43.934 seconds
##################################################

# TransE_l2 1 GPU training
dglke_train --model_name TransE_l2 --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 400 --gamma 19.9 \
--lr 0.25 --batch_size_eval 16 --test -adv --gpu 0 --max_step 24000

################## Script Result #################
# training takes 167.10396814346313 seconds
# [0]Test average MRR: 0.6493311923026039
# [0]Test average MR: 47.0424658461851
# [0]Test average HITS@1: 0.5251646323915289
# [0]Test average HITS@3: 0.7462291141169102
# [0]Test average HITS@10: 0.8446022582993347
# testing takes 246.640 seconds
##################################################

# TransE_l2 1 GPU eval
dglke_eval --model_name TransE_l2 --dataset FB15k --hidden_dim 400 --gamma 19.9 --batch_size_eval 16 \
--gpu 0 --model_path ~/my_task/ckpts/TransE_l2_FB15k_0/

################## Script Result #################
# [0]Test average MRR: 0.6493311923026039
# [0]Test average MR: 47.0424658461851
# [0]Test average HITS@1: 0.5251646323915289
# [0]Test average HITS@3: 0.7462291141169102
# [0]Test average HITS@10: 0.8446022582993347
# Test takes 271.990 seconds
##################################################

# TransE_l2 8 GPU training
dglke_train --model_name TransE_l2 --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 400 --gamma 19.9 \
--lr 0.25 --batch_size_eval 16 --test -adv --gpu 0 --max_step 3000 --mix_cpu_gpu --num_thread 4 \
--num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --force_sync_interval 1000

################## Script Result #################
# training takes 49.43748354911804 seconds
# -------------- Test result --------------
# Test average MRR : 0.6272645091016518
# Test average MR : 47.52452133872797
# Test average HITS@1 : 0.492280476037311
# Test average HITS@3 : 0.7332701325523523
# Test average HITS@10 : 0.8380084982478712
# -----------------------------------------
# testing takes 51.879 seconds
##################################################

# TransE_l2 8 GPU eval
dglke_eval --model_name TransE_l2 --dataset FB15k --hidden_dim 400 --gamma 19.9 --batch_size_eval 16 \
--gpu 0 1 2 3 4 5 6 7 --model_path ~/my_task/ckpts/TransE_l2_FB15k_0/

################## Script Result #################
# -------------- Test result --------------
# Test average MRR: 0.6354831172118882
# Test average MR: 46.50962401178243
# Test average HITS@1: 0.5030217873406578
# Test average HITS@3: 0.7391698125983985
# Test average HITS@10: 0.8430278817016811
# -----------------------------------------
# Test takes 43.081 seconds
##################################################

# DistMult 1 GPU training
dglke_train --model_name DistMult --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 \
--test -adv --gpu 0 --max_step 24000

################## Script Result #################
# training takes 150.90872359275818 seconds
# [0]Test average MRR: 0.6964916432212612
# [0]Test average MR: 61.43418936534002
# [0]Test average HITS@1: 0.5869462172639704
# [0]Test average HITS@3: 0.782101200250546
# [0]Test average HITS@10: 0.8736859034043778
# testing takes 270.463 seconds
##################################################

# DistMult 1 GPU eval
dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 400 --gamma 143.0 --batch_size_eval 16 \
--gpu 0 --model_path ~/my_task/ckpts/DistMult_FB15k_0/

################## Script Result #################
# [0]Test average MRR: 0.6964916432212612
# [0]Test average MR: 61.43418936534002
# [0]Test average HITS@1: 0.5869462172639704
# [0]Test average HITS@3: 0.782101200250546
# [0]Test average HITS@10: 0.8736859034043778
# Test takes 257.484 seconds
##################################################

# DistMult 8 GPU training
dglke_train --model_name DistMult --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 \
--test -adv --max_step 3000 --mix_cpu_gpu --num_proc 8 --num_thread 4 \
--gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --force_sync_interval 1000

################## Script Result #################
# training takes 47.62438082695007 seconds
# -------------- Test result --------------
# Test average MRR : 0.6792720529045182
# Test average MR : 59.44423659663794
# Test average HITS@1 : 0.5666486093006721
# Test average HITS@3 : 0.7646560918217061
# Test average HITS@10 : 0.8643581452827953
# -----------------------------------------
# testing takes 47.160 seconds
##################################################

# DistMult 8 GPU eval
dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 400 --gamma 143.0 --batch_size_eval 16 \
--gpu 0 1 2 3 4 5 6 7 --model_path ~/my_task/ckpts/DistMult_FB15k_1/

################## Script Result #################
# -------------- Test result --------------
# Test average MRR: 0.6792720529045196
# Test average MR: 59.44423659663794
# Test average HITS@1: 0.5666486093006721
# Test average HITS@3: 0.7646560918217061
# Test average HITS@10: 0.8643581452827953
# -----------------------------------------
# Test takes 43.123 seconds
##################################################

# ComplEx 1 GPU training
dglke_train --model_name ComplEx --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 400 --gamma 143.0 --lr 0.1 --regularization_coef 2.00E-06 \
--batch_size_eval 16 --test -adv --gpu 0 --max_step 24000

################## Script Result #################
# training takes 171.46629428863525 seconds
# [0]Test average MRR: 0.7577800768928615
# [0]Test average MR: 64.73522540671395
# [0]Test average HITS@1: 0.6725889184202062
# [0]Test average HITS@3: 0.826116029862369
# [0]Test average HITS@10: 0.886585634236766
# testing takes 277.114 seconds
##################################################

# ComplEx 1 GPU eval
dglke_eval --model_name ComplEx --dataset FB15k --hidden_dim 400 --gamma 143.0 --batch_size_eval 16 \
--gpu 0 --model_path ~/my_task/ckpts/ComplEx_FB15k_0/

################## Script Result #################
# [0]Test average MRR: 0.7577800768928615
# [0]Test average MR: 64.73522540671395
# [0]Test average HITS@1: 0.6725889184202062
# [0]Test average HITS@3: 0.826116029862369
# [0]Test average HITS@10: 0.886585634236766
# Test takes 252.176 seconds
##################################################

# ComplEx 8 GPU training
dglke_train --model_name ComplEx --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 400 --gamma 143.0 --lr 0.1 --num_thread 4 \
--regularization_coef 2.00E-06 --batch_size_eval 16 --test -adv --max_step 3000 --mix_cpu_gpu \
--num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --force_sync_interval 1000

################## Script Result #################
# training takes 49.82103204727173 seconds
# -------------- Test result --------------
# Test average MRR : 0.7508887190664094
# Test average MR : 64.9883699277141
# Test average HITS@1 : 0.6685598686326624
# Test average HITS@3 : 0.8141558463543871
# Test average HITS@10 : 0.8836315620185878
# -----------------------------------------
# testing takes 45.008 seconds
##################################################

# ComplEx 8 GPU eval
dglke_eval --model_name ComplEx --dataset FB15k --hidden_dim 400 --gamma 143.0 --batch_size_eval 16 \
--gpu 0 1 2 3 4 5 6 7 --model_path ~/my_task/ckpts/ComplEx_FB15k_0/

################## Script Result #################
# -------------- Test result --------------
# Test average MRR: 0.750888719066414
# Test average MR: 64.9883699277141
# Test average HITS@1: 0.6685598686326624
# Test average HITS@3: 0.8141558463543871
# Test average HITS@10: 0.8836315620185878
# -----------------------------------------
# Test takes 43.370 seconds
##################################################

# RESCAL 1 GPU training
dglke_train --model_name RESCAL --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 500 --gamma 24.0 --lr 0.03 --batch_size_eval 16 \
--gpu 0 --test -adv --max_step 24000

################## Script Result #################
# training takes 1252.5546259880066 seconds
# [0]Test average MRR: 0.6615993307596552
# [0]Test average MR: 124.54916964331059
# [0]Test average HITS@1: 0.5896125002116097
# [0]Test average HITS@3: 0.7042034162279291
# [0]Test average HITS@10: 0.7874253017555145
# testing takes 275.740 seconds
##################################################

# RESCAL 8 GPU training
dglke_train --model_name RESCAL --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 500 --gamma 24.0 --lr 0.03 --batch_size_eval 16 \
--test -adv --max_step 3000 --mix_cpu_gpu --num_proc 8 --num_thread 4 \
--gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --force_sync_interval 1000

################## Script Result #################
# training takes 179.60524344444275 seconds
# -------------- Test result --------------
# Test average MRR : 0.6435677811671531
# Test average MR : 133.31993702493608
# Test average HITS@1 : 0.5700766873762083
# Test average HITS@3 : 0.6851500736401956
# Test average HITS@10 : 0.7736875962824398
# -----------------------------------------
# testing takes 47.309 seconds
##################################################

# RESCAL 8 GPU eval
dglke_eval --model_name RESCAL --dataset FB15k --hidden_dim 500 --gamma 24.0 --batch_size_eval 16 \
--gpu 0 1 2 3 4 5 6 7 --model_path ~/my_task/ckpts/RESCAL_FB15k_1/

################## Script Result #################
# -------------- Test result --------------
# Test average MRR: 0.6435677811671509
# Test average MR: 133.31993702493608
# Test average HITS@1: 0.5700766873762083
# Test average HITS@3: 0.6851500736401956
# Test average HITS@10: 0.7736875962824398
# -----------------------------------------
# Test takes 53.463 seconds
##################################################

# RotatE 1 GPU training
dglke_train --model_name RotatE --dataset FB15k --batch_size 2048 --log_interval 1000 \
--neg_sample_size 256 --regularization_coef 1e-07 --hidden_dim 200 --gamma 12.0 \
--lr 0.009 --batch_size_eval 16 --test -adv -de --max_step 20000 \
--neg_deg_sample --gpu 0

################## Script Result #################
# training takes 1405.7569651603699 seconds
# [0]Test average MRR: 0.7267354480519053
# [0]Test average MR: 43.85800985255032
# [0]Test average HITS@1: 0.6320275600548493
# [0]Test average HITS@3: 0.7996309525824855
# [0]Test average HITS@10: 0.8737536185268575
# testing takes 352.678 seconds
##################################################

# RotatE 8 GPU training
dglke_train --model_name RotatE --dataset FB15k --batch_size 1024 --log_interval 1000 \
--neg_sample_size 256 --regularization_coef 1e-07 --hidden_dim 200 --gamma 12.0 \
--lr 0.009 --batch_size_eval 16 --test -adv -de --max_step 2500 --num_thread 4 \
--neg_deg_sample --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
--rel_part --force_sync_interval 1000

################## Script Result #################
# training takes 120.12461948394775 seconds
# -------------- Test result --------------
# Test average MRR : 0.6852732234606304
# Test average MR : 50.046562611095126
# Test average HITS@1 : 0.5815544006365222
# Test average HITS@3 : 0.7638858323034992
# Test average HITS@10 : 0.8513991637182374
# -----------------------------------------
# testing takes 52.559 seconds
##################################################

# RotatE 8 GPU eval
dglke_eval --model_name RotatE --dataset FB15k --hidden_dim 200 --gamma 12.0 --batch_size_eval 16 \
--gpu 0 1 2 3 4 5 6 7 --model_path ~/my_task/ckpts/RotatE_FB15k_1/

################## Script Result #################
# -------------- Test result --------------
# Test average MRR: 0.6852732234606344
# Test average MR: 50.046562611095126
# Test average HITS@1: 0.5815544006365222
# Test average HITS@3: 0.7638858323034992
# Test average HITS@10: 0.8513991637182374
# -----------------------------------------
# Test takes 48.808 seconds
##################################################

# TransR 1 GPU training
dglke_train --model_name TransR --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --regularization_coef 5e-8 --hidden_dim 200 --gamma 8.0 \
--lr 0.015 --batch_size_eval 16 --test -adv --gpu 0 --max_step 24000

################## Script Result #################
# training takes 530.2469162940979 seconds
# [0]Test average MRR: 0.6705026761537457
# [0]Test average MR: 59.999365170726755
# [0]Test average HITS@1: 0.5854734133500364
# [0]Test average HITS@3: 0.7280814612923431
# [0]Test average HITS@10: 0.8088148160687986
# testing takes 307.236 seconds
##################################################

# TransR 8 GPU training
dglke_train --model_name TransR --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --regularization_coef 5e-8 --hidden_dim 200 --gamma 8.0 \
--lr 0.015 --batch_size_eval 16 --test -adv --max_step 3000 --mix_cpu_gpu \
--num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --num_thread 4 \
--force_sync_interval 1000

################## Script Result #################
# training takes 90.18287920951843 seconds
# -------------- Test result --------------
# Test average MRR : 0.6665183992974268
# Test average MR : 66.51772443330907
# Test average HITS@1 : 0.5815459362462122
# Test average HITS@3 : 0.7240354827241794
# Test average HITS@10 : 0.8038123613956086
# -----------------------------------------
# testing takes 45.268 seconds
##################################################

# TransR 8 GPU eval
dglke_eval --model_name TransR --dataset FB15k --hidden_dim 200 --gamma 8.0 --batch_size_eval 16 \
--gpu 0 1 2 3 4 5 6 7 --model_path ~/my_task/ckpts/TransR_FB15k_1/

################## Script Result #################
# -------------- Test result --------------
# Test average MRR: 0.6665183992974216
# Test average MR: 66.51772443330907
# Test average HITS@1: 0.5815459362462122
# Test average HITS@3: 0.7240354827241794
# Test average HITS@10: 0.8038123613956086
# -----------------------------------------
# Test takes 42.986 seconds
##################################################

# SimplE 1 GPU training
dglke_train --model_name SimplE --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 400 --gamma 295 \
--lr 0.82 --batch_size_eval 16 --test -adv --gpu 0 --max_step 24000 \
--double_ent --double_rel --regularization_coef 1e-06

################## Script Result #################
# training takes 193.98580360412598 seconds
# -------------- Test result --------------
# [0]Test average MRR: 0.7087428847978637
# [0]Test average MR: 58.852601107142256
# [0]Test average HITS@1: 0.6189416126356418
# [0]Test average HITS@3: 0.7726464762743139
# [0]Test average HITS@10: 0.8622674408762336
# -----------------------------------------
# testing takes 200.421 seconds
##################################################

# SimplE 1 GPU eval
dglke_eval --model_name SimplE --dataset FB15k --hidden_dim 400 --gamma 295 --batch_size_eval 16 \
--gpu 0 --model_path ~/dglke/ckpts/SimplE_13/ --double_ent --double_rel

################## Script Result #################
# training takes 193.98580360412598 seconds
# -------------- Test result --------------
# [0]Test average MRR: 0.7087428847978637
# [0]Test average MR: 58.852601107142256
# [0]Test average HITS@1: 0.6189416126356418
# [0]Test average HITS@3: 0.7726464762743139
# [0]Test average HITS@10: 0.8622674408762336
# -----------------------------------------
# testing takes 200.421 seconds
##################################################

# SimplE 8 GPU training
dglke_train --model_name SimplE --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 400 --gamma 295 --lr 0.82 --batch_size_eval 16 \
--test -adv --max_step 3000 --mix_cpu_gpu --num_proc 8 --num_thread 4 \
--gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --force_sync_interval 1000 \
--double_ent --double_rel --regularization_coef 1e-06

################## Script Result #################
# training takes 72.49912905693054 seconds
# -------------- Test result --------------
# Test average MRR : 0.742840211912461
# Test average MR : 64.74111662236969
# Test average HITS@1 : 0.6662152325168018
# Test average HITS@3 : 0.7969138832929864
# Test average HITS@10 : 0.8731526468148499
# -----------------------------------------
# testing takes 61.748 seconds
##################################################

# SimplE 8 GPU eval
dglke_eval --model_name SimplE --dataset FB15k --hidden_dim 400 --gamma 295 --batch_size_eval 16 \
--gpu 0 1 2 3 4 5 6 7 --model_path ~/dglke/ckpts/SimplE_4/ --double_ent --double_rel
################## Script Result #################
# training takes 72.49912905693054 seconds
# -------------- Test result --------------
# Test average MRR : 0.742840211912461
# Test average MR : 64.74111662236969
# Test average HITS@1 : 0.6662152325168018
# Test average HITS@3 : 0.7969138832929864
# Test average HITS@10 : 0.8731526468148499
# -----------------------------------------
# testing takes 61.748 seconds
##################################################

# Loss function training on 8-GPU

# DistMult + Hinge
dglke_train --model_name DistMult --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 400 --gamma 150.0 --lr 0.1 --batch_size_eval 16 \
--test -adv --max_step 3000 --mix_cpu_gpu --num_proc 8 --num_thread 4 \
--gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --force_sync_interval 1000 --loss_genre Hinge --margin 3
################## Script Result #################
# training takes 59.78469157218933 seconds
# -------------- Test result --------------
# Test average MRR : 0.7754755155370301
# Test average MR : 111.75126542635134
# Test average HITS@1 : 0.709476731391038
# Test average HITS@3 : 0.8274026171894838
# Test average HITS@10 : 0.8829628751841004
# -----------------------------------------
# testing takes 52.860 seconds
##################################################

# DistMult + pairwise Hinge
dglke_train --model_name DistMult --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 400 --gamma 143.0 --lr 0.1 --batch_size_eval 16 \
--test --max_step 3000 --mix_cpu_gpu --num_proc 8 --num_thread 4 \
--gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --force_sync_interval 1000 --loss_genre Hinge --margin 3 --pairwise
################## Script Result #################
# training takes 59.09430718421936 seconds
# -------------- Test result --------------
# Test average MRR : 0.5389195751412187
# Test average MR : 100.28602867735437
# Test average HITS@1 : 0.4113947622352762
# Test average HITS@3 : 0.6195510487379594
# Test average HITS@10 : 0.766662152325168
# -----------------------------------------
# testing takes 54.329 seconds
##################################################

# DistMult + Logsigmoid
dglke_train --model_name DistMult --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 400 --gamma 143.0 --lr 0.1 --batch_size_eval 16 \
--test -adv --max_step 3000 --mix_cpu_gpu --num_proc 8 --num_thread 4 \
--gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --force_sync_interval 1000 --loss_genre Logsigmoid
################## Script Result #################
# training takes 59.721519231796265 seconds
# -------------- Test result --------------
# Test average MRR : 0.6871380814878363
# Test average MR : 57.77132603138596
# Test average HITS@1 : 0.5767720201113914
# Test average HITS@3 : 0.7724094733456349
# Test average HITS@10 : 0.8696991755683838
# -----------------------------------------
# testing takes 55.849 seconds
##################################################

# Distmult + Logistic
dglke_train --model_name DistMult --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 400 --gamma 143.0 --lr 0.1 --batch_size_eval 16 \
--test -adv --max_step 3000 --mix_cpu_gpu --num_proc 8 --num_thread 4 \
--gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --force_sync_interval 1000 --loss_genre Logistic
################## Script Result #################
# training takes 56.380531787872314 seconds
# -------------- Test result --------------
# Test average MRR : 0.6896071641924044
# Test average MR : 57.0942763792724
# Test average HITS@1 : 0.5784818269540045
# Test average HITS@3 : 0.7734505933537608
# Test average HITS@10 : 0.8731695755954698
# -----------------------------------------
# testing takes 54.097 seconds
##################################################

# DistMult + pairwise Logistic
dglke_train --model_name DistMult --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 400 --gamma 150.0 --lr 0.1 --batch_size_eval 16 \
--test --max_step 3000 --mix_cpu_gpu --num_proc 8 --num_thread 4 \
--gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --force_sync_interval 1000 --loss_genre Logistic --pairwise
################## Script Result #################
# training takes 57.917012214660645 seconds
# -------------- Test result --------------
# Test average MRR : 0.38454335306219695
# Test average MR : 74.24669465558395
# Test average HITS@1 : 0.2713175669956493
# Test average HITS@3 : 0.4369741497519934
# Test average HITS@10 : 0.6018604729901306
# -----------------------------------------
# testing takes 58.437 seconds
##################################################

# RotatE + Hinge
dglke_train --model_name RotatE --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 200 --gamma 24.0 --lr 0.015 --batch_size_eval 16 \
--test -adv --max_step 3000 --mix_cpu_gpu --num_proc 8 --num_thread 4 \
--gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --double_ent --force_sync_interval 1000 --loss_genre Hinge --margin 3
################## Script Result #################
# training takes 130.32349681854248 seconds
# -------------- Test result --------------
# Test average MRR : 0.6917848564388618
# Test average MR : 64.32532037717323
# Test average HITS@1 : 0.6046537217924193
# Test average HITS@3 : 0.75125696196103
# Test average HITS@10 : 0.8372043811684244
# -----------------------------------------
# testing takes 58.523 seconds
##################################################

# RotatE + pairwise Hinge
dglke_train --model_name RotatE --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 200 --gamma 12.0 --lr 0.015 --batch_size_eval 16 \
--test --max_step 3000 --mix_cpu_gpu --num_proc 8 --num_thread 4 \
--gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --double_ent --force_sync_interval 1000 --loss_genre Hinge --margin 3 --pairwise
################## Script Result #################
# training takes 131.73789978027344 seconds
# -------------- Test result --------------
# Test average MRR : 0.6022995211193525
# Test average MR : 80.89336561087505
# Test average HITS@1 : 0.48531428281220906
# Test average HITS@3 : 0.6834995175297524
# Test average HITS@10 : 0.7969477408542263
# -----------------------------------------
# testing takes 56.194 seconds
##################################################

# RotatE + Logsigmoid
dglke_train --model_name RotatE --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 200 --gamma 12.0 --lr 0.015 --batch_size_eval 16 \
--test -adv --max_step 3000 --mix_cpu_gpu --num_proc 8 --num_thread 4 \
--gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --double_ent --force_sync_interval 1000 --loss_genre Logsigmoid
################## Script Result #################
# training takes 127.18009305000305 seconds
# -------------- Test result --------------
# Test average MRR : 0.7203891052727601
# Test average MR : 39.418750317414634
# Test average HITS@1 : 0.6311134059013729
# Test average HITS@3 : 0.7863333954055289
# Test average HITS@10 : 0.8652553706556517
# -----------------------------------------
# testing takes 60.876 seconds
##################################################

# RotatE + Logistic
dglke_train --model_name RotatE --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 200 --gamma 12.0 --lr 0.015 --batch_size_eval 16 \
--test -adv --max_step 3000 --mix_cpu_gpu --num_proc 8 --num_thread 4 \
--gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --double_ent --force_sync_interval 1000 --loss_genre Logistic
################## Script Result #################
# training takes 131.61905479431152 seconds
# -------------- Test result --------------
# Test average MRR : 0.7194889832031803
# Test average MR : 39.242623283844864
# Test average HITS@1 : 0.6289549863723316
# Test average HITS@3 : 0.7850552724687241
# Test average HITS@10 : 0.8672952887203534
# -----------------------------------------
# testing takes 59.633 seconds
##################################################

# RotatE + pairwise Logistic
dglke_train --model_name RotatE --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --hidden_dim 200 --gamma 12.0 --lr 0.015 --batch_size_eval 16 \
--test --max_step 3000 --mix_cpu_gpu --num_proc 8 --num_thread 4 \
--gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --double_ent --force_sync_interval 1000 --loss_genre Logistic --pairwise
################## Script Result #################
# training takes 128.0637414455413 seconds
# -------------- Test result --------------
# Test average MRR : 0.5857013188720447
# Test average MR : 63.71882141829324
# Test average HITS@1 : 0.46566843290277804
# Test average HITS@3 : 0.666435306664861
# Test average HITS@10 : 0.788796532985729
# -----------------------------------------
# testing takes 57.576 seconds
##################################################

# TransE_l1 + Hinge
dglke_train --model_name TransE_l1 --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --regularization_coef 1e-07 --hidden_dim 400 --gamma 32.0 \
--lr 0.08 --batch_size_eval 16 --test -adv --max_step 3000 --mix_cpu_gpu --num_thread 4 \
--num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --force_sync_interval 1000 --loss_genre Hinge --margin 3
################## Script Result #################
# training takes 68.79845666885376 seconds
# -------------- Test result --------------
# Test average MRR : 0.703791435136054
# Test average MR : 44.73922906333057
# Test average HITS@1 : 0.6066682466861912
# Test average HITS@3 : 0.7778943982664929
# Test average HITS@10 : 0.8546494895972643
# -----------------------------------------
# testing takes 48.091 seconds
##################################################

# TransE_l1 + pairwise Hinge
dglke_train --model_name TransE_l1 --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --regularization_coef 1e-07 --hidden_dim 400 --gamma 16.0 \
--lr 0.08 --batch_size_eval 16 --test --max_step 3000 --mix_cpu_gpu --num_thread 4 \
--num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --force_sync_interval 1000 --loss_genre Hinge --margin 7 --pairwise
################## Script Result #################
# training takes 63.92800164222717 seconds
# -------------- Test result --------------
# Test average MRR : 0.48780717724802114
# Test average MR : 69.5792436220819
# Test average HITS@1 : 0.2837856139222292
# Test average HITS@3 : 0.6532816441231738
# Test average HITS@10 : 0.7959404784073403
# -----------------------------------------
# testing takes 52.687 seconds
##################################################

# TransE_l1 + Logsigmoid
dglke_train --model_name TransE_l1 --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --regularization_coef 1e-07 --hidden_dim 400 --gamma 16.0 \
--lr 0.01 --batch_size_eval 16 --test -adv --max_step 3000 --mix_cpu_gpu --num_thread 4 \
--num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --force_sync_interval 1000 --loss_genre Logsigmoid
################## Script Result #################
# training takes 67.23829913139343 seconds
# -------------- Test result --------------
# Test average MRR : 0.6676945471491244
# Test average MR : 43.106600531563714
# Test average HITS@1 : 0.549008819894703
# Test average HITS@3 : 0.7599075688578152
# Test average HITS@10 : 0.8517631325015659
# -----------------------------------------
# testing takes 54.306 seconds
##################################################

# TransE_l1 + Logistic
dglke_train --model_name TransE_l1 --dataset FB15k --batch_size 1000 --log_interval 1000 \
--neg_sample_size 200 --regularization_coef 1e-07 --hidden_dim 400 --gamma 16.0 \
--lr 0.01 --batch_size_eval 16 --test -adv --max_step 3000 --mix_cpu_gpu --num_thread 4 \
--num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --rel_part --force_sync_interval 1000 --loss_genre Logistic
################## Script Result #################
# training takes 64.75518202781677 seconds
# -------------- Test result --------------
# Test average MRR : 0.6692648917018105
# Test average MR : 42.38613702155034
# Test average HITS@1 : 0.5516920316229622
# Test average HITS@3 : 0.7610502615496606
# Test average HITS@10 : 0.8515938446953666
# -----------------------------------------
# testing takes 70.450 seconds
##################################################

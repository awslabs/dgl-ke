#!/bin/bash

# TransE-l1
DGLBACKEND=pytorch dglke_train --model TransE_l1 --dataset FB15k --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 \
--gamma 16.0 --lr 0.01 --max_step 500 --log_interval 100 --batch_size_eval 16 --test -adv \
--regularization_coef 1.00E-07 --num_thread 1 --num_proc 48

# TransE-l2
DGLBACKEND=pytorch dglke_train --model TransE_l2 --dataset FB15k --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 \
--gamma 19.9 --lr 0.25 --max_step 500 --log_interval 100 --batch_size_eval 16 --test -adv \
--regularization_coef 1.00E-09 --num_thread 1 --num_proc 48

# DistMult
DGLBACKEND=pytorch dglke_train --model DistMult --dataset FB15k --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 \
--gamma 143.0 --lr 0.08 --max_step 1000 --log_interval 100 --batch_size_eval 16 --test -adv \
--num_thread 1 --num_proc 10

# ComplEx
DGLBACKEND=pytorch dglke_train --model ComplEx --dataset FB15k --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 \
--gamma 143.0 --lr 0.1 --max_step 1000 --log_interval 100 --batch_size_eval 16 --test -adv \
--regularization_coef 2.00E-06 --num_thread 1 --num_proc 10
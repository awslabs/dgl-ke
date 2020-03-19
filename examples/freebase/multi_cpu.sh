#!/bin/bash

# TransE-l1
DGLBACKEND=pytorch dglke_train --model TransE_l1 --dataset Freebase \
--batch_size 1000 --neg_sample_size 200 --hidden_dim 400 --gamma 10 --lr 0.1 --max_step 50000 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --regularization_coef 1e-9 --num_thread 1 --num_proc 48

# TransE-l2
DGLBACKEND=pytorch dglke_train --model TransE_l2 --dataset Freebase \
--batch_size 1000 --neg_sample_size 200 --hidden_dim 400 --gamma 10 --lr 0.1 --max_step 50000 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --regularization_coef 1e-9 --num_thread 1 --num_proc 48

# DistMult
DGLBACKEND=pytorch dglke_train --model DistMult --dataset Freebase \
--batch_size 1024 --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --max_step 50000 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --num_thread 1 --num_proc 48

# ComplEx
DGLBACKEND=pytorch dglke_train --model ComplEx --dataset Freebase \
--batch_size 1024 --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.1 --max_step 50000 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --num_thread 1 --num_proc 48
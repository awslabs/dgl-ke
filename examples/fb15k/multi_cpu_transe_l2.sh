#!/bin/bash

DGLBACKEND=pytorch dglke_train --model TransE_l2 --dataset FB15k --data_path . \
--batch_size 1000 --neg_sample_size 200 --hidden_dim 400 --gamma 19.9 --lr 0.25 --max_step 500 --log_interval 100 \
--batch_size_eval 16 --test -adv --regularization_coef 1e-9 --num_thread 1 --num_proc 48
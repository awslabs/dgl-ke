#!/bin/bash

# TransE-l2
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 16 --model TransE_l2 --dataset FB15k --data_path ~/my_task --hidden_dim 400 \
--gamma 19.9 --lr 0.25 --batch_size 1000 --neg_sample_size 200 --max_step 1000 --log_interval 100 \
--batch_size_eval 16 --test -adv --regularization_coef 1.00E-07 --no_save_emb --num_thread 1
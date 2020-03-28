#!/bin/bash

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
#!/bin/bash

# Partiton data into 4 parts
dglke_partition --dataset FB15k -k 4 --data_path ~/my_task

# TransE-l1 training
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 16 --model TransE_l2 --dataset FB15k --data_path ~/my_task --hidden_dim 400 \
--gamma 19.9 --lr 0.25 --batch_size 1000 --neg_sample_size 200 --max_step 1000 --log_interval 100 \
--batch_size_eval 16 --test -adv --regularization_coef 1.00E-07 --no_save_emb --num_thread 1

# TransE-l2 training
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 16 --model TransE_l2 --dataset FB15k --data_path ~/my_task --hidden_dim 400 \
--gamma 19.9 --lr 0.25 --batch_size 1000 --neg_sample_size 200 --max_step 500 --log_interval 100 \
--batch_size_eval 16 --test -adv --regularization_coef 1.00E-07 --num_thread 1

################## Script Result #################
# Total train time 31.804 seconds
# save model to ckpts/TransE_l2_FB15k_0 ...
# Run test, test processes: 16
# -------------- Test result --------------
# Test average MRR : 0.645860559655554
# Test average MR : 34.84051395777962
# Test average HITS@1 : 0.5109867786223359
# Test average HITS@3 : 0.7546173249140864
# Test average HITS@10 : 0.8548272417937736
# -----------------------------------------
##################################################

# TransE-l2 eval
dglke_eval --model_name TransE_l2 --dataset FB15k --hidden_dim 400 --gamma 19.9 --batch_size_eval 16 \
--num_thread 1 --num_proc 16 --model_path ~/my_task/ckpts/TransE_l2_FB15k_0/

# DistMult training
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 16 --model TransE_l2 --dataset FB15k --data_path ~/my_task --hidden_dim 400 \
--gamma 19.9 --lr 0.25 --batch_size 1000 --neg_sample_size 200 --max_step 1000 --log_interval 100 \
--batch_size_eval 16 --test -adv --regularization_coef 1.00E-07 --no_save_emb --num_thread 1

# ComplEx training
dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
--num_client_proc 16 --model TransE_l2 --dataset FB15k --data_path ~/my_task --hidden_dim 400 \
--gamma 19.9 --lr 0.25 --batch_size 1000 --neg_sample_size 200 --max_step 1000 --log_interval 100 \
--batch_size_eval 16 --test -adv --regularization_coef 1.00E-07 --no_save_emb --num_thread 1
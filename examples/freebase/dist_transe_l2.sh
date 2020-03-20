#!/bin/bash

##################################################################################
# This script runing distmult model on Freebase dataset in distributed setting.
# You can change the hyper-parameter in this file but DO NOT run script manually
##################################################################################
SERVER_ID_LOW=$1
SERVER_ID_HIGH=$2

##################################################################################
# Start kvserver
##################################################################################
while [ $SERVER_ID_LOW -lt $SERVER_ID_HIGH ]
do
    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch dglke_server --model TransE_l2 --dataset Freebase --data_path . \
    --ip_config ./ip_config.txt --hidden_dim 400 --gamma 10 --lr 0.1 --total_client 160 --server_id $SERVER_ID_LOW &
    let SERVER_ID_LOW+=1
done

##################################################################################
# Start kvclient
##################################################################################
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch dglke_client --model TransE_l2 --dataset Freebase --data_path . \
--ip_config ./ip_config.txt --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 --gamma 10 --lr 0.1 --max_step 12500 --log_interval 100 --num_thread 1 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --regularization_coef 1e-9 --num_client 40
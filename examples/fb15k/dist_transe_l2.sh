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
    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch dglke_server --model TransE_l2 --dataset FB15k --data_path . \
    --hidden_dim 400 --gamma 19.9 --lr 0.25 --total_client 64 --server_id $SERVER_ID_LOW &
    let SERVER_ID_LOW+=1
done

##################################################################################
# Start kvclient
##################################################################################
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch dglke_client --model TransE_l2 --dataset FB15k --data_path . \
--batch_size 1000 --neg_sample_size 200 --hidden_dim 400 --gamma 19.9 --lr 0.25 --max_step 500 --log_interval 100 --num_thread 1 \
--batch_size_eval 16 --test -adv --regularization_coef 1e-9 --num_client 16
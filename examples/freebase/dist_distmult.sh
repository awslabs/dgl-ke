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
    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch dglke_server --model DistMult --dataset Freebase --data_path . \
    --hidden_dim 400 --gamma 143.0 --lr 0.08 --total_client 160 --server_id $SERVER_ID_LOW &
    let SERVER_ID_LOW+=1
done

##################################################################################
# Start kvclient
##################################################################################
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch dglke_client --model DistMult --dataset Freebase --data_path . \
--batch_size 1024 --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --max_step 12500 --log_interval 100 \
--batch_size_eval 1000 --neg_sample_size_eval 1000 --test -adv --num_thread 1 --num_client 40
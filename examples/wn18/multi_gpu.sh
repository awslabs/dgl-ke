#!/bin/bash

# DistMult 1GPU 
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset wn18 --batch_size 2048 \
    --neg_sample_size 128 --regularization_coef 1e-06 --hidden_dim 512 --gamma 20.0 \
    --lr 0.14 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 20000

# DistMult 8GPU 
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset wn18 --batch_size 2048 \
    --neg_sample_size 128 --regularization_coef 1e-06 --hidden_dim 512 --gamma 20.0 \
    --lr 0.14 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 2500 \
    --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
    --force_sync_interval 1000

# ComplEx 1GPU
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset wn18 --batch_size 1024 \
    --neg_sample_size 1024 --regularization_coef 0.00001 --hidden_dim 512 --gamma 200.0 \
    --lr 0.1 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 20000

# ComplEx 8GPU 
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset wn18 --batch_size 1024 \
    --neg_sample_size 1024 --regularization_coef 0.00001 --hidden_dim 512 --gamma 200.0 \
    --lr 0.1 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 2500 \
    --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
    --force_sync_interval 1000

# TransE_l1 1GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l1 --dataset wn18 --batch_size 2048 \
    --neg_sample_size 128 --regularization_coef 2e-07 --hidden_dim 512 --gamma 12.0 \
    --lr 0.007 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 32000

# TransE_l1 8GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l1 --dataset wn18 --batch_size 2048 \
    --neg_sample_size 128 --regularization_coef 2e-07 --hidden_dim 512 --gamma 12.0 \
    --lr 0.007 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 4000 \
    --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
    --force_sync_interval 1000

# TransE_l2 1GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 0.0000001 --hidden_dim 512 --gamma 6.0 \
    --lr 0.1 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 32000

# TransE_l2 8GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 0.0000001 --hidden_dim 512 --gamma 6.0 \
    --lr 0.1 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 4000 \
    --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
    --force_sync_interval 1000

# RESCAL 1GPU
DGLBACKEND=pytorch python3 train.py --model RESCAL --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 250 --gamma 24.0 --lr 0.03 --batch_size_eval 16 \
    --valid --test -adv --gpu 0 --max_step 20000

# RESCAL 8GPU
DGLBACKEND=pytorch python3 train.py --model RESCAL --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 250 --gamma 24.0 --lr 0.03 --batch_size_eval 16 \
    --valid --test -adv --gpu 0 --max_step 2500  --mix_cpu_gpu --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --async_update --force_sync_interval 1000 --soft_rel_part

# TransR 1GPU
DGLBACKEND=pytorch python3 train.py --model TransR --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 250 --gamma 16.0 --lr 0.1 --batch_size_eval 16 \
    --valid --test -adv --gpu 0 --max_step 30000

# TransR 8GPU
DGLBACKEND=pytorch python3 train.py --model TransR --dataset wn18 --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 250 --gamma 16.0 --lr 0.1 --batch_size_eval 16 \
    --valid --test -adv --max_step 2500  --mix_cpu_gpu --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --async_update --force_sync_interval 1000 --soft_rel_part

# RotatE 1GPU
DGLBACKEND=pytorch python3 train.py --model RotatE --dataset wn18 --batch_size 2048 \
    --neg_sample_size 64 --regularization_coef 2e-07 --hidden_dim 256 --gamma 9.0 \
    --lr 0.0025 -de --batch_size_eval 16 --neg_deg_sample --valid --test -adv --gpu 0 \
    --max_step 24000 

# RotatE 8GPU
DGLBACKEND=pytorch python3 train.py --model RotatE --dataset wn18 --batch_size 2048 \
    --neg_sample_size 64 --regularization_coef 2e-07 --hidden_dim 256 --gamma 9.0 \
    --lr 0.0025 -de --batch_size_eval 16 --neg_deg_sample --valid --test -adv \
    --max_step 3000 --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
    --force_sync_interval 1000


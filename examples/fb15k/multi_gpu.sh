#!/bin/bash

# DistMult 1GPU
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 \
    --valid --test -adv --gpu 0 --max_step 40000

# DistMult 8GPU
DGLBACKEND=pytorch python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 \
    --valid --test -adv --max_step 5000 --mix_cpu_gpu --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --async_update --soft_rel_part --force_sync_interval 1000

# ComplEx 1GPU
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 1024 --hidden_dim 400 --gamma 143.0 --lr 0.1 \
    --regularization_coef 2.00E-06 --batch_size_eval 16 --valid --test -adv --gpu 0 \
    --max_step 32000

# ComplEx 8GPU
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 1024 --hidden_dim 400 --gamma 143.0 --lr 0.1 \
    --regularization_coef 2.00E-06 --batch_size_eval 16 --valid --test -adv \
    --max_step 4000 --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
    --soft_rel_part --force_sync_interval 1000

# TransE_l1 1GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l1 --dataset FB15k --batch_size 1024 \
    --neg_sample_size 64 --regularization_coef 1e-07 --hidden_dim 400 --gamma 16.0 \
    --lr 0.01 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 48000

# TransE_l1 8GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l1 --dataset FB15k --batch_size 1024 \
    --neg_sample_size 64 --regularization_coef 1e-07 --hidden_dim 400 --gamma 16.0 \
    --lr 0.01 --batch_size_eval 16 --valid --test -adv --max_step 6000 --mix_cpu_gpu \
    --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --soft_rel_part \
    --force_sync_interval 1000

# TransE_l2 1GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef=1e-9 --hidden_dim 400 --gamma 19.9 \
    --lr 0.25 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 32000

# TransE_l2 8GPU
DGLBACKEND=pytorch python3 train.py --model TransE_l2 --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef=1e-9 --hidden_dim 400 --gamma 19.9 \
    --lr 0.25 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 4000 \
    --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --soft_rel_part \
    --force_sync_interval 1000

# RESCAL 1GPU
DGLBACKEND=pytorch python3 train.py --model RESCAL --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 500 --gamma 24.0 --lr 0.03 --batch_size_eval 16 \
    --gpu 0 --valid --test -adv --max_step 30000

# RESCAL 8GPU
DGLBACKEND=pytorch python3 train.py --model RESCAL --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 500 --gamma 24.0 --lr 0.03 --batch_size_eval 16 \
    --valid --test -adv --max_step 4000 --mix_cpu_gpu --num_proc 8 \
    --gpu 0 1 2 3 4 5 6 7 --async_update --soft_rel_part --force_sync_interval 1000

# TransR 1GPU
DGLBACKEND=pytorch python3 train.py --model TransR --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 5e-8 --hidden_dim 200 --gamma 8.0 \
    --lr 0.015 --batch_size_eval 16 --valid --test -adv --gpu 0 --max_step 32000

# TransR 8GPU
DGLBACKEND=pytorch python3 train.py --model TransR --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 5e-8 --hidden_dim 200 --gamma 8.0 \
    --lr 0.015 --batch_size_eval 16 --valid --test -adv --max_step 4000 --mix_cpu_gpu \
    --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update --soft_rel_part \
    --force_sync_interval 1000

# RotatE 1GPU
DGLBACKEND=pytorch python3 train.py --model RotatE --dataset FB15k --batch_size 2048 \
    --neg_sample_size 256 --regularization_coef 1e-07 --hidden_dim 200 --gamma 12.0 \
    --lr 0.009 --batch_size_eval 16 --valid --test -adv -de --max_step 20000 \
    --neg_deg_sample --gpu 0

# RotatE 8GPU
DGLBACKEND=pytorch python3 train.py --model RotatE --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --regularization_coef 1e-07 --hidden_dim 200 --gamma 12.0 \
    --lr 0.009 --batch_size_eval 16 --valid --test -adv -de --max_step 2500 \
    --neg_deg_sample --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
    --soft_rel_part --force_sync_interval 1000


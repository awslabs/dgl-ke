#!/bin/bash
. /opt/conda/etc/profile.d/conda.sh
KG_DIR="${PWD}/python/"
KG_DIR_TEST="${PWD}/python/dglke"

function fail {
    echo FAIL: $@
    exit -1
}

function usage {
    echo "Usage: $0 backend device"
}

# check arguments
if [ $# -ne 2 ]; then
    usage
    fail "Error: must specify device and bakend"
fi

if [ "$2" == "cpu" ]; then
    dev=-1
elif [ "$2" == "gpu" ]; then
    export CUDA_VISIBLE_DEVICES=0
    dev=0
else
    usage
    fail "Unknown device $2"
fi

export DGLBACKEND=$1
export PYTHONPATH=${PWD}/python:.:$PYTHONPATH
conda activate ${DGLBACKEND}-ci
# test
if [ "$2" == "cpu" ]; then
    pip install --pre dgl
else
    pip install --pre dgl-cu101
fi

pushd $KG_DIR> /dev/null
python3 setup.py install

pushd $KG_DIR_TEST> /dev/null
echo $KG_DIR_TEST
python3 -m pytest tests/test_score.py || fail "run test_score.py on $1"
python3 -m pytest tests/test_infer.py || fail "run test_score.py on $1"
python3 -m pytest tests/test_topk.py || fail "run test_score.py on $1"
popd

if [ "$2" == "cpu" ]; then
    # verify CPU training DistMult
    dglke_train --model DistMult --dataset FB15k --batch_size 128 \
        --neg_sample_size 16 --hidden_dim 100 --gamma 500.0 --lr 0.1 --max_step 100 \
        --batch_size_eval 16 --valid --test -adv --eval_interval 30 --eval_percent 0.01 \
        --data_path /data/kg || fail "run DistMult on $2"

    # verify saving training result
    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 500.0 --batch_size 16 --eval_percent 0.01 --model_path ckpts/DistMult_FB15k_0/ \
        --data_path /data/kg || fail "eval DistMult on $2"

    # verify score sim
    printf '1\n2\n3\n4\n5\n' > head.list
    printf '6\n7\n8\n9\n10\n' > tail.list
    dglke_score --data_path data/FB15k/ --model_path ckpts/DistMult_FB15k_0/ --format h_*_t --data_files head.list tail.list --topK 5 || fail "run dglke_score DistMult"
    dglke_score --data_path data/FB15k/ --model_path ckpts/DistMult_FB15k_0/ --format h_*_t --data_files head.list tail.list --topK 5 --bcast head || fail "run dglke_score DistMult with bcast head"
    dglke_score --data_path data/FB15k/ --model_path ckpts/DistMult_FB15k_0/ --format h_*_t --data_files head.list tail.list --topK 5 --bcast rel || fail "run dglke_score DistMult with bcast rel"
    dglke_score --data_path data/FB15k/ --model_path ckpts/DistMult_FB15k_0/ --format h_*_t --data_files head.list tail.list --topK 5 --bcast tail || fail "run dglke_score DistMult with bcast tail"

    # verify emb sim
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --exec_mode 'batch_left' || fail "run dglke_emb_sim DistMult with cosine"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --sim_func 'l2' || fail "run dglke_emb_sim DistMult with l2"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --sim_func 'l1' || fail "run dglke_emb_sim DistMult with l1"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --sim_func 'dot' || fail "run dglke_emb_sim DistMult with dot"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format '*_r' --data_files head.list --sim_func 'ext_jaccard' --exec_mode 'batch_left' || fail "run dglke_emb_sim DistMult with extended jaccard"

elif [ "$2" == "gpu" ]; then
    # verify GPU training DistMult
    dglke_train --model DistMult --dataset FB15k --batch_size 128 \
        --neg_sample_size 16 --hidden_dim 100 --gamma 500.0 --lr 0.1 --max_step 100 \
        --batch_size_eval 16 --gpu 0 --valid --test -adv --eval_interval 30 --eval_percent 0.01 \
        --data_path /data/kg || fail "run DistMult on $2"

    # verify mixed CPU GPU training
    dglke_train --model DistMult --dataset FB15k --batch_size 128 \
        --neg_sample_size 16 --hidden_dim 100 --gamma 500.0 --lr 0.1 --max_step 100 \
        --batch_size_eval 16 --gpu 0 --valid --test -adv --mix_cpu_gpu --eval_percent 0.01 \
        --data_path /data/kg || fail "run mix with async CPU/GPU DistMult"

    # verify saving training result
    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 500.0 --batch_size 16 --gpu 0 --model_path ckpts/DistMult_FB15k_0/ \
        --eval_percent 0.01 --data_path /data/kg || fail "eval DistMult on $2"

    # verify score sim
    printf '1\n2\n3\n4\n5\n' > head.list
    dglke_score --data_path data/FB15k/ --model_path ckpts/DistMult_FB15k_0/ --format h_*_* --data_files head.list --topK 5 --gpu 0 || fail "run dglke_score DistMult"
    dglke_score --data_path data/FB15k/ --model_path ckpts/DistMult_FB15k_0/ --format h_*_* --data_files head.list --topK 5 --bcast head --gpu 0 || fail "run dglke_score DistMult with bcast head"
    dglke_score --data_path data/FB15k/ --model_path ckpts/DistMult_FB15k_0/ --format h_*_* --data_files head.list --topK 5 --bcast rel --gpu 0 || fail "run dglke_score DistMult with bcast rel"
    dglke_score --data_path data/FB15k/ --model_path ckpts/DistMult_FB15k_0/ --format *_*_t --data_files head.list --topK 5 --bcast tail --gpu 0 || fail "run dglke_score DistMult with bcast tail"

    # verify emb sim
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --exec_mode 'batch_left' --gpu 0 || fail "run dglke_emb_sim DistMult with cosine"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --sim_func l2 --gpu 0 || fail "run dglke_emb_sim DistMult with l2"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --sim_func l1 --gpu 0 || fail "run dglke_emb_sim DistMult with l1"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --sim_func dot --gpu 0 || fail "run dglke_emb_sim DistMult with dot"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format '*_r' --data_files head.list --sim_func ext_jaccard --gpu 0 --exec_mode 'batch_left' || fail "run dglke_emb_sim DistMult with extended jaccard"

    if [ "$1" == "pytorch" ]; then
        # verify mixed CPU GPU training with async_update
        dglke_train --model DistMult --dataset FB15k --batch_size 128 \
            --neg_sample_size 16 --hidden_dim 100 --gamma 500.0 --lr 0.1 --max_step 100 \
            --batch_size_eval 16 --gpu 0 --valid --test -adv --mix_cpu_gpu --eval_percent 0.01 \
            --async_update --data_path /data/kg || fail "run mix CPU/GPU DistMult"

        # verify mixed CPU GPU training with random partition
        dglke_train --model DistMult --dataset FB15k --batch_size 128 \
            --neg_sample_size 16 --hidden_dim 100 --gamma 500.0 --lr 0.1 --max_step 100 \
            --batch_size_eval 16 --num_proc 2 --gpu 0 --valid --test -adv --mix_cpu_gpu \
            --eval_percent 0.01 --async_update --force_sync_interval 100 \
            --data_path /data/kg || fail "run multiprocess async CPU/GPU DistMult"

        # verify mixed CPU GPU training with random partition async_update
        dglke_train --model DistMult --dataset FB15k --batch_size 128 \
            --neg_sample_size 16 --hidden_dim 100 --gamma 500.0 --lr 0.1 --max_step 100 \
            --batch_size_eval 16 --num_proc 2 --gpu 0 --valid --test -adv --mix_cpu_gpu \
            --eval_percent 0.01 --rel_part --async_update --force_sync_interval 100 \
            --data_path /data/kg || fail "run multiprocess async CPU/GPU DistMult"

        # multi process training TransR
        dglke_train --model TransR --dataset FB15k --batch_size 128 \
            --neg_sample_size 16 --hidden_dim 100 --gamma 500.0 --lr 0.1 --max_step 100 \
            --batch_size_eval 16 --num_proc 2 --gpu 0 --valid --test -adv --eval_interval 30 \
            --eval_percent 0.01 --data_path /data/kg --mix_cpu_gpu --rel_part \
            --async_update || fail "run multiprocess TransR on $2"

        dglke_eval --model_name TransR --dataset FB15k --hidden_dim 100 \
            --gamma 500.0 --batch_size 16 --num_proc 2 --gpu 0 --model_path ckpts/TransR_FB15k_0/ \
            --eval_percent 0.01 --mix_cpu_gpu --data_path /data/kg || fail "eval multiprocess TransR on $2"
    fi
fi

popd > /dev/null

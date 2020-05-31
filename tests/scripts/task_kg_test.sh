#!/bin/bash
. /opt/conda/etc/profile.d/conda.sh
KG_DIR="${PWD}/python/"

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
export PYTHONPATH=${PWD}/python:$PYTHONPATH
conda activate ${DGLBACKEND}-ci
# test
if [ "$2" == "cpu" ]; then
    pip install --pre dgl
else
    pip install --pre dgl-cu101
fi

pushd $KG_DIR> /dev/null
python3 setup.py install

#python3 -m pytest tests/test_score.py || fail "run test_score.py on $1"

if [ "$2" == "cpu" ]; then
    rm -fr ckpts/
    # verify CPU training DistMult
    dglke_train --model DistMult --dataset FB15k --batch_size 128 \
        --neg_sample_size 16 --hidden_dim 100 --gamma 500.0 --lr 0.1 --max_step 100 \
        --batch_size_eval 16 --valid --test -adv --eval_interval 30 --eval_percent 0.01 \
        --data_path /data/kg || fail "run DistMult on $2"

    # verify saving training result
    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 500.0 --batch_size 16 --eval_percent 0.01 --model_path ckpts/DistMult_FB15k_0/ \
        --data_path /data/kg || fail "eval DistMult on $2"

    rm -fr ckpts/
    # udd test
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --max_step 100 --dataset 'udd_test' --format 'udd_hrt' --data_path fake_data/udd/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_udd_test_0/ \
        --dataset 'udd_test' --format 'udd_hrt' --data_path fake_data/udd/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

    rm -fr ckpts/
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 --delimiter '|' \
        --test -adv --max_step 100 --dataset 'udd_test' --format 'udd_hrt' --data_path fake_data/udd_1/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_udd_test_0/ --delimiter '|' \
        --dataset 'udd_test' --format 'udd_hrt' --data_path fake_data/udd_1/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

    rm -fr ckpts/
    # udd raw test
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --max_step 100 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path fake_data/raw_udd/ --delimiter '|' \
        --data_files train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_raw_udd_test_0/ \
        --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path fake_data/raw_udd/ --delimiter '|' \
        --data_files train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

    rm -fr ckpts/
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --max_step 100 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path fake_data/raw_udd_1/ --delimiter ',' \
        --data_files train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_raw_udd_test_0/ \
        --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path fake_data/raw_udd_1/ --delimiter ',' \
        --data_files train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

    rm -fr ckpts/
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --max_step 100 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path fake_data/raw_udd_2/ --delimiter ';' \
        --data_files train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_raw_udd_test_0/ \
        --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path fake_data/raw_udd_2/ --delimiter ';' \
        --data_files train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"
elif [ "$2" == "gpu" ]; then
    rm -fr ckpts/
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

    rm -fr ckpts/
    # udd test
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 512 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --gpu 0 --max_step 100 --dataset 'udd_test' --format 'udd_hrt' --data_path fake_data/udd/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_udd_test_0/ \
        --gpu 0 --dataset 'udd_test' --format 'udd_hrt' --data_path fake_data/udd/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

    rm -fr ckpts/
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 512 --gamma 20.0 --lr 0.14 --batch_size_eval 1 --delimiter '|' \
        --test -adv --gpu 0 --max_step 100 --dataset 'udd_test' --format 'udd_hrt' --data_path fake_data/udd_1/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_udd_test_0/ --delimiter '|' \
        --gpu 0 --dataset 'udd_test' --format 'udd_hrt' --data_path fake_data/udd_1/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

    rm -fr ckpts/
    # udd raw test
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --gpu 0 --max_step 100 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path fake_data/raw_udd/ --delimiter '|' \
        --data_files train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_raw_udd_test_0/ \
        --gpu 0 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path fake_data/raw_udd/  --delimiter '|' \
        --data_files train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

   rm -fr ckpts/
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --gpu 0 --max_step 100 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path fake_data/raw_udd_1/ --delimiter ',' \
        --data_files train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_raw_udd_test_0/ \
        --gpu 0 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path fake_data/raw_udd_1/  --delimiter ',' \
        --data_files train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

   rm -fr ckpts/
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --gpu 0 --max_step 100 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path fake_data/raw_udd_2/ --delimiter ';' \
        --data_files train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_raw_udd_test_0/ \
        --gpu 0 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path fake_data/raw_udd_2/  --delimiter ';' \
        --data_files train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

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

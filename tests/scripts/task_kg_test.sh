#!/bin/bash
#. /opt/conda/etc/profile.d/conda.sh
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
    pip3 uninstall dgl
    pip3 install dgl>=0.5.0,<=0.9.1
else
    pip3 uninstall dgl
    pip3 install dgl-cu102>=0.5.0,<=0.9.1
fi

pushd $KG_DIR> /dev/null
python3 setup.py install

pushd $KG_DIR_TEST> /dev/null
echo $KG_DIR_TEST
python3 -m pytest tests/test_score.py || fail "run test_score.py on $1"
python3 -m pytest tests/test_infer.py || fail "run test_score.py on $1"
python3 -m pytest tests/test_dataset.py || fail "run test_dataset.py on $1"
popd

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

    # verify score sim
    printf '1\n2\n3\n4\n5\n' > head.list
    printf '6\n7\n8\n9\n10\n' > tail.list
    dglke_predict --model_path ckpts/DistMult_FB15k_0/ --format h_*_t --data_files head.list tail.list --topK 5 || fail "run dglke_predict DistMult"
    dglke_predict --model_path ckpts/DistMult_FB15k_0/ --format h_*_t --data_files head.list tail.list --topK 5 --exec_mode batch_head || fail "run dglke_predict DistMult with batched head"
    dglke_predict --model_path ckpts/DistMult_FB15k_0/ --format h_*_t --data_files head.list tail.list --topK 5 --exec_mode batch_rel || fail "run dglke_predict DistMult with batched rel"
    dglke_predict --model_path ckpts/DistMult_FB15k_0/ --format h_*_t --data_files head.list tail.list --topK 5 --exec_mode batch_tail || fail "run dglke_predict DistMult with batched tail"

    # verify emb sim
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --exec_mode 'batch_left' || fail "run dglke_emb_sim DistMult with cosine"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --sim_func 'l2' || fail "run dglke_emb_sim DistMult with l2"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --sim_func 'l1' || fail "run dglke_emb_sim DistMult with l1"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --sim_func 'dot' || fail "run dglke_emb_sim DistMult with dot"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format '*_r' --data_files head.list --sim_func 'ext_jaccard' --exec_mode 'batch_left' --topK 3 || fail "run dglke_emb_sim DistMult with extended jaccard"
    rm head.list
    rm tail.list

    rm -fr ckpts/
    # udd test
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --max_step 100 --dataset 'udd_test' --format 'udd_hrt' --data_path ../tests/fake_data/udd/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_udd_test_0/ \
        --dataset 'udd_test' --format 'udd_hrt' --data_path ../tests/fake_data/udd/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

    rm -fr ckpts/
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 --delimiter '|' \
        --test -adv --max_step 100 --dataset 'udd_test' --format 'udd_hrt' --data_path ../tests/fake_data/udd_1/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_udd_test_0/ --delimiter '|' \
        --dataset 'udd_test' --format 'udd_hrt' --data_path ../tests/fake_data/udd_1/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

    rm -fr ckpts/
    # udd raw test
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --max_step 100 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path ../tests/fake_data/raw_udd/ --delimiter '|' \
        --data_files train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_raw_udd_test_0/ \
        --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path ../tests/fake_data/raw_udd/ --delimiter '|' \
        --data_files train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

    rm -fr ckpts/
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --max_step 100 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path ../tests/fake_data/raw_udd_1/ --delimiter ',' \
        --data_files train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_raw_udd_test_0/ \
        --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path ../tests/fake_data/raw_udd_1/ --delimiter ',' \
        --data_files train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

    rm -fr ckpts/
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --max_step 100 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path ../tests/fake_data/raw_udd_2/ --delimiter ';' \
        --data_files train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_raw_udd_test_0/ \
        --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path ../tests/fake_data/raw_udd_2/ --delimiter ';' \
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

    # verify score sim
    printf '1\n2\n3\n4\n5\n' > head.list
    printf '1\n2\n' > rel.list
    dglke_predict --model_path ckpts/DistMult_FB15k_0/ --format h_r_* --data_files head.list rel.list --topK 5 --gpu 0 || fail "run dglke_predict DistMult"
    dglke_predict --model_path ckpts/DistMult_FB15k_0/ --format h_r_* --data_files head.list rel.list --topK 5 --exec_mode 'batch_head' --gpu 0 || fail "run dglke_predict DistMult with batched head"
    dglke_predict --model_path ckpts/DistMult_FB15k_0/ --format h_r_* --data_files head.list rel.list --topK 5 --exec_mode 'batch_rel' --gpu 0 || fail "run dglke_predict DistMult with batched rel"
    dglke_predict --model_path ckpts/DistMult_FB15k_0/ --format *_r_t --data_files head.list rel.list --topK 5 --exec_mode 'batch_tail' --gpu 0 || fail "run dglke_predict DistMult with batched tail"

    # verify emb sim
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --exec_mode 'batch_left' --gpu 0 || fail "run dglke_emb_sim DistMult with cosine"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --sim_func l2 --gpu 0 || fail "run dglke_emb_sim DistMult with l2"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --sim_func l1 --gpu 0 || fail "run dglke_emb_sim DistMult with l1"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format 'l_*' --data_files head.list --sim_func dot --gpu 0 || fail "run dglke_emb_sim DistMult with dot"
    dglke_emb_sim --mfile data/FB15k/entities.dict --emb_file ckpts/DistMult_FB15k_0/FB15k_DistMult_entity.npy --format '*_r' --data_files head.list --sim_func ext_jaccard --gpu 0 --exec_mode 'batch_left' --topK 3|| fail "run dglke_emb_sim DistMult with extended jaccard"
    rm head.list

    rm -fr ckpts/
    # udd test
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 512 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --gpu 0 --max_step 100 --dataset 'udd_test' --format 'udd_hrt' --data_path ../tests/fake_data/udd/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_udd_test_0/ \
        --gpu 0 --dataset 'udd_test' --format 'udd_hrt' --data_path ../tests/fake_data/udd/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

    rm -fr ckpts/
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 512 --gamma 20.0 --lr 0.14 --batch_size_eval 1 --delimiter '|' \
        --test -adv --gpu 0 --max_step 100 --dataset 'udd_test' --format 'udd_hrt' --data_path ../tests/fake_data/udd_1/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_udd_test_0/ --delimiter '|' \
        --gpu 0 --dataset 'udd_test' --format 'udd_hrt' --data_path ../tests/fake_data/udd_1/ \
        --data_files entity.dict relation.dict train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

    rm -fr ckpts/
    # udd raw test
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --gpu 0 --max_step 100 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path ../tests/fake_data/raw_udd/ --delimiter '|' \
        --data_files train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_raw_udd_test_0/ \
        --gpu 0 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path ../tests/fake_data/raw_udd/  --delimiter '|' \
        --data_files train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

   rm -fr ckpts/
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --gpu 0 --max_step 100 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path ../tests/fake_data/raw_udd_1/ --delimiter ',' \
        --data_files train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_raw_udd_test_0/ \
        --gpu 0 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path ../tests/fake_data/raw_udd_1/  --delimiter ',' \
        --data_files train.tsv valid.tsv test.tsv || fail "eval DistMult on $2"

   rm -fr ckpts/
    dglke_train --model_name DistMult --batch_size 2 --log_interval 1000 --neg_sample_size 2 \
        --regularization_coef 1e-06 --hidden_dim 100 --gamma 20.0 --lr 0.14 --batch_size_eval 1 \
        --test -adv --gpu 0 --max_step 100 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path ../tests/fake_data/raw_udd_2/ --delimiter ';' \
        --data_files train.tsv valid.tsv test.tsv || fail "run DistMult on $2"

    dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 100 \
        --gamma 20.0 --batch_size 2 --model_path ckpts/DistMult_raw_udd_test_0/ \
        --gpu 0 --dataset 'raw_udd_test' --format 'raw_udd_hrt' \
        --data_path ../tests/fake_data/raw_udd_2/  --delimiter ';' \
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

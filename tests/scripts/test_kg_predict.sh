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
    pip3 install dgl==0.7 -f https://data.dgl.ai/wheels/repo.html
else
    pip3 uninstall dgl
    pip3 install dgl-cu102==0.7 -f https://data.dgl.ai/wheels/repo.html
fi

pushd $KG_DIR> /dev/null
python3 setup.py install

pushd $KG_DIR_TEST> /dev/null
echo $KG_DIR_TEST
python3 -m pytest tests/test_topk.py || fail "run test_score.py on $1"
popd
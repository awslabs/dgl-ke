#!/bin/bash
. /opt/conda/etc/profile.d/conda.sh

if [ "$1" == "cpu" ]; then
    pip install --pre dgl
else
	pip install --pre dgl-cu101
fi

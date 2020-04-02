#!/bin/bash
if [ "$1" == "cpu" ]; then
    pip install --pre dgl
else
	pip install --pre dgl-cu101
fi

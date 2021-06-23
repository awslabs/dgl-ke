##Distributed Demo

First setup distributed enviroment as [https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage/experimental#distributed-training](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage/experimental#distributed-training)

To part the graph into parts needed:

```bash
python3 ~/workspace/dgl-ke/python/partion_graph.py --nump_parts NUM_PARTS
```

Run the demo:

```bash
python3 ~/workspace/dgl-ke/python/dglke/launch.py \
		--workspace ~/workspace/dgl-ke/python \
		--num_trainers 1 \
		--num_samplers 0 \
		--num_servers 1 \
		--part_config data/FB15k.json \
		--ip_config ip_config.txt "python3 ke_test.py"
```

To run in standalone mode, first construct a graph with one partition:

```bash
python3 ~/workspace/dgl-ke/python/partion_graph.py --nump_parts 1
```

To run the demo:

```bash
python3 ~/workspace/dgl-ke/python/dglke/launch.py \
		--workspace ~/workspace/dgl-ke/python \
		--num_trainers 1 \
		--num_samplers 0 \
		--num_servers 1 \
		--part_config data/FB15k.json \
		--ip_config ip_config.txt "python3 ke_test.py --standalone"
```
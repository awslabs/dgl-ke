## Built-in Datasets and Benchmark

DGL-KE provides five built-in knowledge graphs:

| Dataset | #nodes | #edges | #relations |
|---------|--------|--------|------------|
| [FB15k](https://data.dgl.ai/dataset/FB15k.zip) | 14951 | 592213 | 1345 |
| [FB15k-237](https://data.dgl.ai/dataset/FB15k-237.zip) | 14541 | 310116 | 237 |
| [wn18](https://data.dgl.ai/dataset/wn18.zip) | 40943 | 151442 | 18 |
| [wn18rr](https://data.dgl.ai/dataset/wn18rr.zip) | 40943 | 93003 | 11 |
| [Freebase](https://data.dgl.ai/dataset/Freebase.zip) | 86054151 | 338586276 | 14824 |

Users can specify one of the datasets with `--dataset` in their tasks.
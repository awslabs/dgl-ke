# <img src="https://github.com/awslabs/dgl-ke/blob/master/img/logo.png" width = "400"/>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

DGL-KE is a high performance, easy-to-use, and scalable package for learning large-scale knowledge graph embeddings. The package is implemented on the top of *[Deep Graph Library (DGL)](https://github.com/dmlc/dgl)* and developers can run DGL-KE on CPU machine, GPU machine, as well as clusters with a number of popular models, including [TransE](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf), [TransR](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9571), [RESCAL](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.2015&rep=rep1&type=pdf), [DistMult](https://arxiv.org/abs/1412.6575), [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf), and [RotatE](https://arxiv.org/pdf/1902.10197.pdf).

<p align="center">
  <img src="https://github.com/awslabs/dgl-ke/blob/master/img/dgl_ke_arch.PNG" alt="DGL-ke architecture" width="600">
  <br>
  <b>Figure</b>: DGL-KE Overall Architecture
</p>

Get started with our [tutorials](https://docs.dgl.ai)!

## Installation

DGL-KE is built in Python (version 3.6 or later) and relies on [DGL](https://github.com/dmlc/dgl) (at least version 0.4.2). The package can run with both [Pytorch](https://pytorch.org/) and [MXNet](https://mxnet.apache.org/). For Pytorch, it works with Pytorch v1.2 or newer. For MXNet, it works with MXNet 1.5 or newer.

#### Using anaconda

```
conda install -c dglteam dgl-ke
```

#### Using pip

```
pip install dgl-ke
```

#### Build from source

Refer to the guide [here](https://docs.dgl.ai/install/index.html#install-from-source).

## Performance and Scalability

DGL-KE is designed for learning at scale, and it introduces various novel optimizations that accelerate training on knowledge graphs with millions of nodes and billions of edges. Our benchmark on knowledge graphs consisting of over *86M* nodes and *338M* edges show that DGL-KE can compute embeddings in 100 minutes on a EC2 instance with 8 GPUs and 30 minutes on an EC2 cluster with 4 machines (48 cores/machine). These results represent a *2×∼5×* speedup overthe best competing approaches.

## License

This project is licensed under the Apache-2.0 License.


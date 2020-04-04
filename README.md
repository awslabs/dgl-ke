# <img src="https://github.com/awslabs/dgl-ke/blob/master/img/logo.png" width = "400"/>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

[Documentation](https://aws-dglke.readthedocs.io/en/latest/index.html)

Knowledge graphs (KGs) are data structures that store information about different entities (nodes) and their relations (edges). A common approach of using KGs in various machine learning tasks is to compute knowledge graph embeddings. DGL-KE is a high performance, easy-to-use, and scalable package for learning large-scale knowledge graph embeddings. The package is implemented on the top of *[Deep Graph Library (DGL)](https://github.com/dmlc/dgl)* and developers can run DGL-KE on CPU machine, GPU machine, as well as clusters with a set of popular models, including [TransE](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf), [TransR](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9571), [RESCAL](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.2015&rep=rep1&type=pdf), [DistMult](https://arxiv.org/abs/1412.6575), [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf), and [RotatE](https://arxiv.org/pdf/1902.10197.pdf).

<p align="center">
  <img src="https://github.com/awslabs/dgl-ke/blob/master/img/dgl_ke_arch.PNG" alt="DGL-ke architecture" width="600">
  <br>
  <b>Figure</b>: DGL-KE Overall Architecture
</p>

### A Quick Start

To install the latest version of DGL-KE run:

```
sudo pip3 install dgl
sudo pip3 install dglke
```

Train a `transE` model on `FB15k` dataset by running the following command:

```
DGLBACKEND=pytorch dglke_train --model_name TransE_l2 --dataset FB15k --batch_size 1000 \
--neg_sample_size 200 --hidden_dim 400 --gamma 19.9 --lr 0.25 --max_step 500 --log_interval 100 \
--batch_size_eval 16 -adv --regularization_coef 1.00E-09 --test --num_thread 1 --num_proc 8
```

This command will download the `FB15k` dataset, train the `transE` model on that, and save the trained embeddings into the file.

### Performance and Scalability

DGL-KE is designed for learning at scale. It introduces various novel optimizations that accelerate training on knowledge graphs with millions of nodes and billions of edges. Our benchmark on knowledge graphs consisting of over *86M* nodes and *338M* edges shows that DGL-KE can compute embeddings in 100 minutes on an EC2 instance with 8 GPUs and 30 minutes on an EC2 cluster with 4 machines (48 cores/machine). These results represent a *2×∼5×* speedup over the best competing approaches.

<p align="center">
  <img src="https://github.com/awslabs/dgl-ke/blob/master/img/vs-gv-fb15k.png" alt="vs-gv-fb15k" width="750">
  <br>
  <b>Figure</b>: DGL-KE vs Pytorch-BigGraph on Freebase
</p>

<p align="center">
  <img src="https://github.com/awslabs/dgl-ke/blob/master/img/vs-pbg-fb.png" alt="vs-pbg-fb" width="750">
  <br>
  <b>Figure</b>: DGL-KE vs Pytorch-BigGraph on Freebase
</p>

Learn more details with our [documentation](https://aws-dglke.readthedocs.io/en/latest/index.html)!

### License

This project is licensed under the Apache-2.0 License.

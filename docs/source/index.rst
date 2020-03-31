.. DGL-KE documentation master file, created by
   sphinx-quickstart on Tue Mar 31 07:38:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DGL-KE's documentation!
==================================

DGL-KE is a DGL-based package for computing node embeddings and relation embeddings
of knowledge graphs efficiently. This package is adapted from
[KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).
We enable fast and scalable training of knowledge graph embedding, while still keeping
the package as extensible as KnowledgeGraphEmbedding. On a single machine, it takes
only a few minutes for medium-size knowledge graphs, such as FB15k and wn18, and
takes a couple of hours on Freebase, which has hundreds of millions of edges.

DGL-KE includes the following knowledge graph embedding models:

* TransE (TransE_l1 with L1 distance and TransE_l2 with L2 distance)
* DistMult
* ComplEx
* RESCAL
* TransR
* RotatE

We will add other popular models in the future.

DGL-KE supports multiple training modes:

* Multiprocessing CPU training
* Multi-GPU GPU training
* Distributed training

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   /get_started
   /ec2


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

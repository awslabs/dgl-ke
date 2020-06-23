dglke_emb_sim: finding similar embeddings
-------------------------------------------
This task is to find the most similar entity/relation embeddings for some pre-defined similarity functions given a set of entities or relations. An example of the output for top5 similar entities are as follows::

    left     right    score
    0        0        0.99999
    0        18470    0.91855
    0        2105     0.89916
    0        13605    0.83187
    0        36762    0.76978

Currently we support five different similarity functions: cosine, l2 distance, l1 distance, dot product and extended jaccard.

Four arguments are required to provide basic information for finding similar embeddings:

  * ``--emb_file``, The numpy file that contains the embeddings of all entities/relations in a knowledge graph.
  * ``--format``, The format of the input objects (entities/relations).

    * ``l_r``: two list of objects are provided as left objects and right objects.
    * ``l_*``: one list of objects is provided as left objects and all objects in emb\_file are right objects. This is to find most similar objects to the ones on the left.
    * ``*_r``: one list of objects is provided as right objects list and treat all objects in emb\_file as left objects.
    * ``*``: all objects in the emb\_file are both left objects and right objects. The option finds the most similar objects in the graph.

  * ``--data_files`` A list of data file names. It provides necessary files containing the requried data according to the format, e.g., for ``l_r``, two files are required as left_data and right_data, while for ``l_*``, one file is required as left_data, and for ``*`` this argument will be omited.
  * ``--raw_data``, A flag indicates whether the data in data_files are raw IDs or KGE IDs. If True, the data are the Raw IDs and the command will map the raw IDs to KGE Ids automatically. If False, the data are KGE IDs. Default: False.

Task related arguments:

  * ``--exec_mode``, How to calculate scores for element pairs and calculate topK. Default: 'all'

    * ``pairwise``: both left and right objects are provided with the same length N, and we will calculate the similarity pair by pair: result = topK([score(l_i, r_i)]) for i in N, the result shape will be (K,).
    * ``all``: both left and right objects are provided as L and R, and we calculate all possible combinations of (l_i, r_j): result = topK([[score(l_i, rj) for l_i in L] for r_j in R]), the result shape will be (K,).
    * ``batch_left``: both left and right objects are provided as L and R,, and we calculate topK for each element in L: result = topK([score(l_i, r_j) for r_j in R]) for l_j in L, the result shape will be (sizeof(L), K).

  * ``--topk``, How many results are returned. Default: 10.
  * ``--sim_func``, What kind of distance function is used in ranking and will be output. It support five functions: 1)cosine: use cosine distance; 2) l2: use l2 distance; 3) l1: use l1 distance; 4) dot: use dot product as distance; 5) ext_jaccard: use extended jaccard as distance. Default: cosine.
  * ``--gpu``, GPU device to use in inference. Default: -1 (CPU).

Input/Output related arguments:

  * ``--output``, Where to store the result, by default it is stored in result.tsv.
  * ``--mfile``, The ID mapping file.

The following command shows how to do entity similarity using cosine distance::

    # Using PyTorch Backend
    dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_r' --data_files head.list tail.list  --topK 5

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_r' --data_files head.list tail.list --topK 5

The output is as::

    left    right   score
    6       15      0.55512
    1       12      0.33153
    7       20      0.27706
    7       19      0.25631
    7       13      0.21372

The following command shows how to do entity similarity using l2 distance with calculating topK for each element in left (--exec_mode batch_left)::

    # Using PyTorch Backend
    dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_*' --data_files head.list --sim_func l2 --topK 5 --exec_mode 'batch_left'

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'l_*' --data_files head.list --sim_func l2 --topK 5 --exec_mode 'batch_left'

The output is as::

    left    right   score
    0       0       0.0
    0       18470   3.1008
    0       24408   3.1466
    0       2105    3.3411
    0       13605   4.1587
    1       1       0.0
    1       26231   4.9025
    1       2617    5.0204
    1       12672   5.2221
    1       38633   5.3221
    ...

The following command shows how to do relation similarity using cosine distance and use Raw ID (turn on --raw_data)::

    # Using PyTorch Backend
    dglke_emb_sim --mfile data/wn18/relations.dict --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_relation.npy  --format 'l_*' --data_files raw_rel.list --topK 5 --raw_data

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_emb_sim --mfile data/wn18/relations.dict --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_relation.npy  --format 'l_*' --data_files raw_rel.list --topK 5 --raw_data

The output is as::

    left                          right                           score
    _hyponym                      _hyponym                        0.99999
    _derivationally_related_form  _derivationally_related_form    0.99999
    _hyponym                      _also_see                       0.58408
    _hyponym                      _member_of_domain_topic         0.44027
    _hyponym                      _member_of_domain_region        0.30975

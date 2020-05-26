Inference Using Pretrained Embedding
--------------------------------------

Users can use DGL-KE to do inference tasks based on pretained embeddings (We recommand using DGL-KE to generate these embedding). Here we support two kinds of inference tasks:

  * **Linkage score ranking** Given a list of (h, r, t) triplets, calculate the linkage score using the predefined score function for each triplet, sort the resulting scores and output the topk most confident triplets.
  * **Embedding similarity ranking** Given a list of (e, e) enitity pairs or (r, r) relation pairs, calculate the similarity of for each pair, sort the resulting similarity score and output the topk most similar pairs.

The Linkage Score Ranking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The task of linkage score ranking is given a list of candidate (h, r, t) triplets, calculating the edge score of each triplet based on the trained model and the pretrained embeddings and then returning the topk most relevent triplets along with their scores. An example return value of top5 linkage score likes this::

  src   relation dst   score (DistMult)
  407   5        8429  3.5953474
  3645  3        7121  3.585188
  93    10       7035  3.4557137
  93    9        7035  3.4197974
  2441  5        4833  3.3639894

DGL-KE provides dglke_score command to calculate linkage score ranking. Currently, we support six models in inference: TransE_l1, TransE_l2, RESCAL, DistMult, ComplEx, and RotatE.

Four arguments are required to provide basic information for doning the linkage score ranking task:

  * ``--data_path``, The path containing the id mapping files, include both the entity mapping file and the relation mapping file
  * ``--model_path``, The path containing the pretrained model, include the embedding files (.npy) and a config.json containing the configure information of the model.
  * ``--format``, The format of the input data, specified in ``h_r_t``. Ideally, user should provides three files, one for head entities, one for relations and one for tail entities. But we also allow users to use *** to represent *all* of the entities or relations. For example, ``h_r_*`` requires users to provide files containing head entities and relation entities and use the whole entity set as tail entities; ``*_*_t`` requires users to provide a single file containing tail entities and use the whole entity set as head entities and the whole relation set as relations. The supported formats include ``h_r_t``, ``h_r_*``, ``h_*_t``, ``*_r_t``, ``h_*_*``, ``*_r_*``, ``*_*_t``. By default, the calculation will take an N_h x N_r x N_t manner.
  * ``--data_files`` A list of data file names. This is used to provide necessary files containing the requried data according to the format, e.g., for ``h_r_t``, three files are required as h_data, r_data and t_data, while for ``h_*_t``, two files are required as h_data and t_data.
  * ``--raw_data``, A flag tells whether the data profiled in data_files is in the raw object naming space or in mapped id space. If True, the data is in the original naming space and the inference program will do the id translation according to id mapping files. If False, the data is just intergers and it is assumed that user has already done the id translation.

Task related arguments:

  * ``--bcast``, Whether to broadcast topK in a specific side. By default, an universal topK across all scores are returned. Users can specify ``head`` to broadcast at head that returns topK for each head; ``rel`` to broadcast at relation that returns topK for each relation; ``tail`` to broadcast at tail that returns topK for each tail.
  * ``--topk``, How many results are returned.
  * ``--score_func``, What kind of score is used in ranking. Currently, we support two functions: ``none`` (score = $x$) and ``logsigmoid`` ($score = log(sigmoid(x))$).
  * ``--gpu``, GPU device to use in inference, by default it uses CPU.

Input/Output related arguments:

  * ``--output``, Where to store the result, by default it is stored in result.tsv
  * ``--entity_mfile``, The entity ID mapping file. If not provided we will use the mapping file in ``--data_path`` according to the config.json under ``--model_path``.
  * ``--rel_mfile``, The relation ID mapping file. If not provided we will use the mapping file in ``--data_path`` according to the config.json under ``--model_path``.

The following command shows how to do linkage score ranking using a pretrained DistMult model::

    # Using PyTorch Backend
    dglke_score --data_path data/wn18/ --model_path ckpts/DistMult_wn18_0/ --format h_r_t --data_files head.list rel.list tail.list --score_func none --topK 5

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_score --data_path data/wn18/ --model_path ckpts/DistMult_wn18_0/ --format h_r_t --data_files head.list rel.list tail.list --score_func none --topK 5

The output is as::

    src  rel  dst  score
    6    0    15   -2.39380
    8    0    14   -2.65297
    2    0    14   -2.67331
    9    0    18   -2.86985
    8    0    20   -2.89651

The following command shows how to do linkage score ranking while broadcasting at head using a pretrained TransE_l2 model::

    # Using PyTorch Backend
    dglke_score --data_path data/wn18/ --model_path ckpts/TransE_l2_wn18_0/ --format h_r_t --data_files head.list rel.list tail.list --score_func logsigmoid --topK 5 --bcast head

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_score --data_path data/wn18/ --model_path ckpts/TransE_l2_wn18_0/ --format h_r_t --data_files head.list rel.list tail.list --score_func logsigmoid --topK 5 --bcast head

The output is as::

    src  rel  dst  score
    1    0    12   -5.11393
    1    0    18   -6.10925
    1    0    13   -6.66778
    1    0    17   -6.81532
    1    0    19   -6.83329
    2    0    17   -5.09325
    2    0    18   -5.42972
    2    0    20   -5.61894
    2    0    12   -5.75848
    2    0    14   -5.94183
    ...

The following command shows how to do linkage score ranking using a pretrained TransE_l2 model and use raw ID space (turn on --raw_data)::

    # Using PyTorch Backend
    dglke_score --data_path data/wn18/ --model_path ckpts/TransE_l2_wn18_0/ --format h_r_t --data_files raw_head.list raw_rel.list raw_tail.list --topK 5 --raw_data

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_score --data_path data/wn18/ --model_path ckpts/TransE_l2_wn18_0/ --format h_r_t --data_files raw_head.list raw_rel.list raw_tail.list --topK 5 --raw_data

The output is as::

    head      rel                           tail      score
    08847694  _derivationally_related_form  09440400  -7.41088
    08847694  _hyponym                      09440400  -8.99562
    02537319  _derivationally_related_form  01490112  -9.08666
    02537319  _hyponym                      01490112  -9.44877
    00083809  _derivationally_related_form  05940414  -9.88155

The Embedding Similarity Ranking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The task of embedding similarity ranking is given a list of entity (e1, e2) pairs or relation (r1, r2) pairs, calculating the similarity between their corresponding embeddings and returning the topk most similar pairs. An example of return value of top5 similar entities likes this::

    head  tail  score
    0     0     0.99999
    0     18470 0.91855
    0     2105  0.89916
    0     13605 0.83187
    0     36762 0.76978

DGL-KE provides dglke_emb_sim command to calculate the embedding similarity ranking between entity pairs or relation pairs. Currently we support five different similarity functions: cosine, l2 distance, l1 distance, dot and extended jaccard.

Four arguments are required to provide basic information for doning the embedding similarity ranking task:

  * ``--emb_file``, The numpy file containing the embeddings.
  * ``--format``, The format of the input data, specified in ``e_e``. Ideally, user should provides two files, one for heads and one for tails. But we also allow users to use *** to represent *all* of the embeddings. For exmpale, ``e_*`` only requires users to provide a file containing heads and use the whole embedding set as tails; ``*_e`` only requires users to provide a file containing tails and use the whole embedding set as heads; even users can specify a single *** to treat the whole embedding set as both heads and tails. By default, the calculation will take an N_head x N_tail manner, but user can use ``e_e_pw`` to give two files with same length and the similarity is calcuated pair by pair.
  * ``--data_files`` A list of data file names. This is used to provide necessary files containing the requried data according to the format, e.g., for ``e_e``, two files are required as h_data and t_data, while for ``e_*``, one file is required as t_data, and for ``*`` this argument can be omited.
  * ``--raw_data``, A flag tells whether the data profiled in data_files is in the raw object naming space or in mapped id space. If True, the data is in the original naming space and the inference program will do the id translation according to id mapping files. If False, the data is just intergers and it is assumed that user has already done the id translation.

Task related arguments:

 * ``--bcast``, Whether to broadcast topK or not (boolean flag). By default, an universal topK across all pairs are returned. Users can turn it on that topK for each head will be returned.
 * ``--topk``, How many results are returned.
 * ``--sim_func``, What kind of distance function is used in ranking and will be output. It support five functions: 1)cosine: use cosine distance; 2) l2: use l2 distance; 3) l1: use l1 distance; 4) dot: use dot product as distance; 5) ext_jaccard: use extended jaccard as distance.
 * ``--gpu``, GPU device to use in inference, by default it uses CPU.

Input/Output related arguments:

  * ``--output``, Where to store the result, by default it is stored in result.tsv
  * ``--mfile``, The ID mapping file.

The following command shows how to do entity similarity using cosine distance::

    # Using PyTorch Backend
    dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'e_e' --data_files head.list tail.list  --topK 5

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'e_e' --data_files head.list tail.list --topK 5

The output is as::

    head    tail    score
    6       15      0.55512
    1       12      0.33153
    7       20      0.27706
    7       19      0.25631
    7       13      0.21372

The following command shows how to do entity similarity using l2 distance with broadcast::

    # Using PyTorch Backend
    dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'e_*' --data_files head.list --sim_func l2 --topK 5 --bcast

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_emb_sim --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_entity.npy --format 'e_*' --data_files head.list --sim_func l2 --topK 5 --bcast

The output is as::

    head    tail    score
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

The following command shows how to do relation similarity using cosine distance and use raw ID space (turn on --raw_data)::

    # Using PyTorch Backend
    dglke_emb_sim --mfile data/wn18/relations.dict --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_relation.npy  --format e_* --data_files raw_rel.list --topK 5 --raw_data

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_emb_sim --mfile data/wn18/relations.dict --emb_file ckpts/TransE_l2_wn18_0/wn18_TransE_l2_relation.npy  --format e_* --data_files raw_rel.list --topK 5 --raw_data

The output is as::

    head                          tail                            score
    _hyponym                      _hyponym                        0.99999
    _derivationally_related_form  _derivationally_related_form    0.99999
    _hyponym                      _also_see                       0.58408
    _hyponym                      _member_of_domain_topic         0.44027
    _hyponym                      _member_of_domain_region        0.30975

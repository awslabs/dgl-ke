dglke_predict: predicting entities/relations in a triplet
-------------------------------------------
The task is mainly used to predict missing entities or relations in a triplet. An example return value of top5 linkage score likes this::

  src   rel  dst   score (DistMult)
  407   5    8429  3.5953474
  3645  3    7121  3.585188
  93    10   7035  3.4557137
  93    9    7035  3.4197974
  2441  5    4833  3.3639894

Currently, it supports six models: TransE_l1, TransE_l2, RESCAL, DistMult, ComplEx, and RotatE.

Four arguments are required to provide basic information for predicting missing entities or relations:

  * ``--data_path``, The path containing the id mapping files, including both the entity ID mapping file and the relation ID mapping file. Default: ./data.
  * ``--model_path``, The path containing the pretrained model, including the embedding files (.npy) and a config.json containing the configure information of the model.
  * ``--format``, The format of the input data, specified in ``h_r_t``. Ideally, user should provides three files, one for head entities, one for relations and one for tail entities. But we also allow users to use *\** to represent *all* of the entities or relations. For example, ``h_r_*`` requires users to provide files containing head entities and relation entities and use the whole entity set as tail entities; ``*_*_t`` requires users to provide a single file containing tail entities and use the whole entity set as head entities and the whole relation set as relations. The supported formats include ``h_r_t``, ``h_r_*``, ``h_*_t``, ``*_r_t``, ``h_*_*``, ``*_r_*``, ``*_*_t``. By default, the calculation will take an N\_h x N\_r x N\_t manner.
  * ``--data_files`` A list of data file names. This is used to provide necessary files containing the requried data according to the format, e.g., for ``h_r_t``, three files are required as h_data, r_data and t_data, while for ``h_*_t``, two files are required as h_data and t_data.
  * ``--raw_data``, A flag tells whether the data profiled in data_files is in the raw object naming space or in mapped id space. If True, the data is using the Raw ID and the inference program will do the ID translation according to ID mapping files. If False, the data is using the KGE ID and it is assumed that user has already done the ID translation. Default False.

Task related arguments:

  * ``--exec_mode``, How to calculate scores for triplets and calculate topK. Default 'all'.

    * ``triplet_wise``: head, relation and tail lists have the same length N, and we calculate the similarity triplet by triplet: result = topK([score(h_i, r_i, t_i) for i in N]), the result shape will be (K,).
    * ``all``: three lists of head, relation and tail ids are provided as H, R and T, and we calculate all possible combinations of all triplets (h_i, r_j, t_k): result = topK([[[score(h_i, r_j, t_k) for each h_i in H] for each r_j in R] for each t_k in T]), the result shape will be (K,).
    * ``batch_head``: three lists of head, relation and tail ids are provided as H, R and T, and we calculate topK for each element in head: result = topK([[score(h_i, r_j, t_k) for each r_j in R] for each t_k in T]) for each h_i in H, the result shape will be (sizeof(H), K).
    * ``batch_rel``: three lists of head, relation and tail ids are provided as H, R and T, and we calculate topK for each element in relation: result = topK([[score(h_i, r_j, t_k) for each h_i in H] for each t_k in T]) for each r_j in R, the result shape will be (sizeof(R), K).
    * ``batch_tail``: three lists of head, relation and tail ids are provided as H, R and T, and we calculate topK for each element in tail: result = topK([[score(h_i, r_j, t_k) for each h_i in H] for each r_j in R]) for each t_k in T, the result shape will be (sizeof(T), K).

  * ``--topk``, How many results are returned. Default: 10.
  * ``--score_func``, What kind of score is used in ranking. Currently, we support two functions: ``none`` (score = $x$) and ``logsigmoid`` ($score = log(sigmoid(x))$). Default: 'none'.
  * ``--gpu``, GPU device to use in inference. Default: -1 (CPU)

Input/Output related arguments:

  * ``--output``, Where to store the result, by default it is stored in result.tsv
  * ``--entity_mfile``, The entity ID mapping file. If not provided we will use the mapping file in ``--data_path`` according to the config.json under ``--model_path``, otherwise we will search the mapping file under ``--data_path``.
  * ``--rel_mfile``, The relation ID mapping file. If not provided we will use the mapping file in ``--data_path`` according to the config.json under ``--model_path``,  otherwise we will search the mapping file under ``--data_path``.

The following command shows how to do entities/relations linkage prediction and ranking using a pretrained DistMult model::

    # Using PyTorch Backend
    dglke_predict --data_path data/wn18/ --model_path ckpts/DistMult_wn18_0/ --format 'h_r_t' --data_files head.list rel.list tail.list --score_func none --topK 5

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_predict --data_path data/wn18/ --model_path ckpts/DistMult_wn18_0/ --format 'h_r_t' --data_files head.list rel.list tail.list --score_func none --topK 5

The output is as::

    src  rel  dst  score
    6    0    15   -2.39380
    8    0    14   -2.65297
    2    0    14   -2.67331
    9    0    18   -2.86985
    8    0    20   -2.89651

The following command shows how to do entities/relations linkage prediction and ranking while calculate topK for each element in head using a pretrained TransE_l2 model (--exec_mode ‘batch_head’)::

    # Using PyTorch Backend
    dglke_predict --data_path data/wn18/ --model_path ckpts/TransE_l2_wn18_0/ --format 'h_r_t' --data_files head.list rel.list tail.list --score_func logsigmoid --topK 5 --exec_mode 'batch_head'

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_predict --data_path data/wn18/ --model_path ckpts/TransE_l2_wn18_0/ --format 'h_r_t' --data_files head.list rel.list tail.list --score_func logsigmoid --topK 5  --exec_mode 'batch_head'

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

The following command shows how to do entities/relations linkage prediction and ranking using a pretrained TransE_l2 model and use Raw ID (turn on --raw_data)::

    # Using PyTorch Backend
    dglke_predict --data_path data/wn18/ --model_path ckpts/TransE_l2_wn18_0/ --format 'h_r_t' --data_files raw_head.list raw_rel.list raw_tail.list --topK 5 --raw_data

    # Using MXNet Backend
    MXNET_ENGINE_TYPE=NaiveEngine DGLBACKEND=mxnet dglke_predict --data_path data/wn18/ --model_path ckpts/TransE_l2_wn18_0/ --format 'h_r_t' --data_files raw_head.list raw_rel.list raw_tail.list --topK 5 --raw_data

The output is as::

    head      rel                           tail      score
    08847694  _derivationally_related_form  09440400  -7.41088
    08847694  _hyponym                      09440400  -8.99562
    02537319  _derivationally_related_form  01490112  -9.08666
    02537319  _hyponym                      01490112  -9.44877
    00083809  _derivationally_related_form  05940414  -9.88155

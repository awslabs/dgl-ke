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
  * ``--format``, The format of the input data, specified in ``h_r_t``. Ideally, user should provides three files, one for head entities, one for relations and one for tail entities. But we also allow users to use *** to represent *all* of the entities or relations. For example, ``h_r_*`` requires users to provide files containing head entities and relation entities and use the whole entity set as tail entities; ``*_*_t`` requires users to provide a single file containing tail entities and use the whole entity set as head entities and the whole relation set as relations. The supported formats include ``h_r_t``, ``h_r_*``, ``h_*_t``, ``*_r_t``, ``h_*_*``, ``*_r_*``, ``*_*_t``.
  * ``--data_files`` A list of data file names. This is used to provide necessary files containing the requried data according to the format, e.g., for ``h_r_t``, three files are required as h_data, r_data and t_data, while for ``h_*_t``, two files are required as h_data and t_data.
  * ``--raw_data``, A flag tells whether the data profiled in data_files is in the raw object naming space or in mapped id space. If True, the data is in the original naming space and the inference program will do the id translation according to id mapping files. If False, the data is just intergers and it is assumed that user has already done the id translation.

Task related arguments:

  * ``--bcast``, Whether to broadcast topK in a specific side. By default, an universal topK across all scores are returned. Users can specify ``head`` to broadcast at head that returns topK for each head; ``rel`` to broadcast at relation that returns topK for each relation; ``tail`` to broadcast at tail that returns topK for each tail.
  * ``--topk``, How many results are returned.
  * ``--score_func``, What kind of score is used in ranking. Currently, we support two functions: ``none`` (score = $x$) and ``logsigmoid`` ($score = log(sigmoid(x))$).
  * ``--gpu``, GPU device to use in inference, by default it uses CPU.

Input/Output related arguments:


  * ``--output``, Where to store the result, by default it is stored in result.tsv
  * ``--entity_mfile``, The entity id mapping file. If not provided we will use the mapping file in ``--data_path`` according to the config.json under ``--model_path``.
  * ``--rel_mfile``, The relation id mapping file. If not provided we will use the mapping file in ``--data_path`` according to the config.json under ``--model_path``.

The following command shows how to do linkage score ranking using a pretrained DistMult model::

    dglke_score --data_path data/wn18/ --model_path ckpts/DistMult_wn18_0/ --format h_r_t --data_files head.list rel.list tail.list --score_func none --topK 5

The output is as::

    src  rel  dst  score
    6    0    15   -2.393801
    8    0    14   -2.6529696
    2    0    14   -2.6733077
    9    0    18   -2.8698525
    8    0    20   -2.8965101

The following command shows how to do linkage score ranking while broadcasting at head using a pretrained TransE_l2 model::

    dglke_score --data_path data/wn18/ --model_path ckpts/TransE_l2_wn18_0/ --format h_r_t --data_files head.list rel.list tail.list --score_func logsigmoid --topK 5 --bcast head

The output is as::

    src  rel  dst  score
    1    0    12   -5.113936
    1    0    18   -6.10925
    1    0    13   -6.667781
    1    0    17   -6.8153195
    1    0    19   -6.833286
    2    0    17   -5.093254
    2    0    18   -5.4297166
    2    0    20   -5.618936
    2    0    12   -5.7584825
    2    0    14   -5.941834
    ...

The Embedding Similarity Ranking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
DGL-KE provides dglke_sim command to embedding similarity score ranking.
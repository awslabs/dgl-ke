Partition a Knowledge Graph
===========================

For distributed training, a user needs to partition a graph beforehand. DGL-KE provides a partition tool ``dglke_partition``, which partitions a given knowledge graph into ``N`` parts with `the METIS partition algorithm`__. This partition algorithm minimizes edge cuts between partitions, which results in low network communication during the distributed training. For a cluster of ``P`` machines, we split the graph into ``P`` partitions using the METIS partition algorithm as shown in the following Figure.

.. __: http://glaros.dtc.umn.edu/gkhome/metis/metis/overview


.. image:: ../images/metis.png
    :width: 400

The majority of the triplets are in the diagonal blocks. We co-locate the embeddings of the entities with the triplets in the diagonal block by specifying a proper data partitioning in the distributed KVStore. When a trainer process samples triplets in the local partition, most of the entity embeddings accessed by the batch fall in the local partition and, thus, there is little network communication to access entity embeddings from other machines.

Arguments
---------

  - ``--data_path DATA_PATH``
    The name of the knowledge graph stored under data_path. If it is one ofthe builtin knowledge grpahs such as FB15k, DGL-KE will automatically download the knowledge graph and keep it under data_path.

  - ``--dataset DATA_SET``
    The name of the knowledge graph stored under data_path. If it is one ofthe builtin knowledge grpahs such as FB15k, DGL-KE will automatically download the knowledge graph and keep it under data_path.

  - ``--data_files [DATA_FILES ...]``
    A list of data file names. This is used if users want to train KGE on their own datasets. If the format is *raw_udd_{htr}*, users need to provide *train_file* [*valid_file*] [*test_file*]. If the format is *udd_{htr}*, users need to provide *entity_file* *relation_file* *train_file* [*valid_file*] [*test_file*]. In both cases, *valid_file* and *test_file* are optional.

  - ``--delimiter DELIMITER``
    Delimiter used in data files. Note all files should use the same delimiter.

  - ``--format FORMAT``
    The format of the dataset. For builtin knowledge graphs,the foramt should be *built_in*. For users own knowledge graphs,it needs to be *raw_udd_{htr}* or *udd_{htr}*.

  - ``-k NUM_PARTS`` or ``--num-parts NUM_PARTS``
    The number of partitions.

dglke_predict
^^^^^^^^^^^^^^^
  - ``--model_path MODEL_PATH``
    The place where to load the model. Default 'ckpts'.

  - ``--format FORMAT``
    The format of the input data, specified in ``h_r_t``. Ideally, user should provides three files, one for head entities, one for relations and one for tail entities. But we also allow users to use *\** to represent *all* of the entities or relations. For example, ``h_r_*`` requires users to provide files containing head entities and relation entities and use the whole entity set as tail entities; ``*_*_t`` requires users to provide a single file containing tail entities and use the whole entity set as head entities and the whole relation set as relations. The supported formats include ``h_r_t``, ``h_r_*``, ``h_*_t``, ``*_r_t``, ``h_*_*``, ``*_r_*``, ``*_*_t``. By default, the calculation will take an N\_h x N\_r x N\_t manner.

  - ``--data_files [DATA_FILES ...]`` 
    A list of data file names. This is used to provide necessary files containing the requried data according to the format, e.g., for ``h_r_t``, three files are required as h_data, r_data and t_data, while for ``h_*_t``, two files are required as h_data and t_data.

  - ``--raw_data``
    A flag tells whether the data profiled in data_files is in the raw object naming space or in mapped id space. If True, the data is in the original naming space and the inference program will do the id translation according to id mapping files. If False, the data is just intergers and it is assumed that user has already done the id translation. Default: False.

  - ``--exec_mode``
    How to calculate scores for triplets and calculate topK. Possible candidates include: ``triplet_wise``, ``all``, ``batch_head``, ``batch_rel``, ``batch_tail``.

  - ``--topk NUM_OF_K``
    How many results are returned. Default:10.

  - ``--score_func SCORE_FUNC_NAME or None``
    What kind of score is used in ranking. Currently, we support two functions: ``none`` (score = $x$) and ``logsigmoid`` ($score = log(sigmoid(x))$). Default: none.

  - ``--gpu GPU_ID``
    GPU device to use in inference, by default it uses CPU. Default: -1.(CPU)

  - ``--output FILE_PATH``
    Where to store the result. Default: result.tsv

  - ``--entity_mfile`` (Optional)
    The entity ID mapping file. Required if Raw ID is used.

  - ``--rel_mfile`` (Optional)
    The relation ID mapping file. Required if Raw ID is used.

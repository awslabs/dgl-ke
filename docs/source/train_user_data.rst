Train User-Defined Knowledage Graphs
--------------------------------------

Users can use DGL-KE to train embeddings on their own knowledge graphs. In this case, users need to use ``--data_path`` to specify the path to the knowledge graph dataset, ``--data_files`` to specify the triplets of a knowledge graph as well as node/relation Id mapping, ``--format`` to specify the input format of the knowledge graph.

The input format of users' knowledge graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users need to store all the data associated with a knowledge graph in the same directory. DGL-KE supports two knowledge graph input formats:

``raw_udd_[h|r|t]``, raw user defined dataset. In this format, users only need to provide triplets and the dataloader generates the id mappings for entities and relations in the triplets. The dataloader outputs two files: entities.tsv for entity id mapping and relations.tsv for relation id mapping while loading data. The order of head, relation and tail entities are described in ``[h|r|t]``, for example, raw_udd_trh means the triplets are stored in the order of tail, relation and head. The directory contains three files:


  * *train* stores the triplets in the training set. The format of a triplet, e.g., ``[src_name, rel_name, dst_name]``, should follow the order specified in ``[h|r|t]``
  * *valid* stores the triplets in the validation set. The format of a triplet, e.g., ``[src_name, rel_name, dst_name]``, should follow the order specified in ``[h|r|t]``. This is optional.
  * *test* stores the triplets in the test set. The format of a triplet, e.g., ``[src_name, rel_name, dst_name]``, should follow the order specified in ``[h|r|t]``. This is optional.

``udd_[h|r|t]``, user defined dataset. In this format, user should provide the id mapping for entities and relations. The order of head, relation and tail entities are described in ``[h|r|t]``, for example, raw_udd_trh means the triplets are stored in the order of tail, relation and head. The directory should contains five files:


  * *entities* stores the mapping between entity name and entity Id
  * *relations* stores the mapping between relation name relation Id
  * *train* stores the triplets in the training set. The format of a triplet, e.g., ``[src_id, rel_id, dst_id]``, should follow the order specified in ``[h|r|t]``
  * *valid* stores the triplets in the validation set. The format of a triplet, e.g., ``[src_id, rel_id, dst_id]``, should follow the order specified in ``[h|r|t]``
  * *test* stores the triplets in the test set. The format of a triplet, e.g., ``[src_id, rel_id, dst_id]``, should follow the order specified in ``[h|r|t]``

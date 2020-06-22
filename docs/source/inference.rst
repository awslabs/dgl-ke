Inference Using Pretrained Embedding
--------------------------------------

Users can use DGL-KE to do inference tasks based on pretained embeddings (We recommand using DGL-KE to generate these embedding). Here we support two kinds of inference tasks:

  * **Predicting entities/relations in a triplet** Given entities and/or relations, predict which entities or relations are likely to connect with the existing entities for given relations. For example, given a head entity and a relation, predict which entities are likely to connect to the head entity via the given relation.
  * **Finding similar embeddings** Given entity/relation embeddings, find the most similar entity/relation embeddings for some pre-defined similarity functions.

The ranking result will be automatically stored in the output file (result.tsv by default) using the tsv format.

.. toctree::
   :glob:
   :maxdepth: 1

   predict
   emb_sim
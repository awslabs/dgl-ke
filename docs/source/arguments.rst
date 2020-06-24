DGL-KE Command Line Arguments
----------------------------------


dglke_train
^^^^^^^^^^^^

  - ``--model_name {TransE, TransE_l1, TransE_l2, TransR, RESCAL, DistMult, ComplEx, RotatE}``
    The models provided by DGL-KE.

  - ``--data_path DATA_PATH``
    The path of the directory where DGL-KE loads knowledge graph data.

  - ``--dataset DATA_SET``
    The name of the knowledge graph stored under data_path. If it is one ofthe builtin knowledge grpahs such as FB15k, DGL-KE will automatically download the knowledge graph and keep it under data_path.

  - ``--format FORMAT``
    The format of the dataset. For builtin knowledge graphs,the foramt should be *built_in*. For users own knowledge graphs,it needs to be *raw_udd_{htr}* or *udd_{htr}*.

  - ``--data_files [DATA_FILES ...]``
    A list of data file names. This is used if users want to train KGE on their own datasets. If the format is *raw_udd_{htr}*, users need to provide *train_file* [*valid_file*] [*test_file*]. If the format is *udd_{htr}*, users need to provide *entity_file* *relation_file* *train_file* [*valid_file*] [*test_file*]. In both cases, *valid_file* and *test_file* are optional.

  - ``--delimiter DELIMITER``
    Delimiter used in data files. Note all files should use the same delimiter.

  - ``--save_path SAVE_PATH``
    The path of the directory where models and logs are saved.

  - ``--no_save_emb``         
    Disable saving the embeddings under save_path.

  - ``--max_step MAX_STEP``   
    The maximal number of steps to train the model. A step trains the model with a batch of data.

  - ``--batch_size BATCH_SIZE``
    The batch size for training.

  - ``--batch_size_eval BATCH_SIZE_EVAL``
    The batch size used for validation and test.

  - ``--neg_sample_size NEG_SAMPLE_SIZE``
    The number of negative samples we use for each positive sample in the training.

  - ``--neg_deg_sample``
    Construct negative samples proportional to vertex degree in the training. When this option is turned on, the number of negative samples per positive edge will be doubled. Half of the negative samples are generated uniformly whilethe other half are generated proportional to vertex degree.

  - ``--neg_deg_sample_eval``
    Construct negative samples proportional to vertex degree in the evaluation.

  - ``--neg_sample_size_eval NEG_SAMPLE_SIZE_EVAL``
    The number of negative samples we use to evaluate a positive sample.

  - ``--eval_percent EVAL_PERCENT``
    Randomly sample some percentage of edges for evaluation.

  - ``--no_eval_filter`` 
    Disable filter positive edges from randomly constructed negative edges for evaluation.

  - ``-log LOG_INTERVAL``
    Print runtime of different components every *x* steps.

  - ``--eval_interval EVAL_INTERVAL``
    Print evaluation results on the validation dataset every *x* stepsif validation is turned on.

  - ``--test``
    Evaluate the model on the test set after the model is trained.

  - ``--num_proc NUM_PROC`` 
    The number of processes to train the model in parallel.In multi-GPU training, the number of processes by default is set to match the number of GPUs. If set explicitly, the number of processes needs to be divisible by the number of GPUs.

  - ``--num_thread NUM_THREAD``
    The number of CPU threads to train the model in each process. This argument is used for multi-processing training.

  - ``--force_sync_interval FORCE_SYNC_INTERVAL``
    We force a synchronization between processes every *x* steps formultiprocessing training. This potentially stablizes the training processto get a better performance. For multiprocessing training, it is set to 1000 by default.

  - ``--hidden_dim HIDDEN_DIM``
    The embedding size of relation and entity.

  - ``--lr LR``          
    The learning rate. DGL-KE uses Adagrad to optimize the model parameters.

  - ``-g GAMMA`` or ``--gamma GAMMA``
    The margin value in the score function. It is used by *TransX* and *RotatE*.

  - ``-de`` or ``--double_ent``
    Double entitiy dim for complex number It is used by *RotatE*.

  - ``-dr`` or ``--double_rel``
    Double relation dim for complex number.

  - ``-adv`` or ``--neg_adversarial_sampling``
    Indicate whether to use negative adversarial sampling.It will weight negative samples with higher scores more.

  - ``-a ADVERSARIAL_TEMPERATURE`` or ``--adversarial_temperature ADVERSARIAL_TEMPERATURE``
    The temperature used for negative adversarial sampling.

  - ``-rc REGULARIZATION_COEF`` or ``--regularization_coef REGULARIZATION_COEF``
    The coefficient for regularization.

  - ``-rn REGULARIZATION_NORM`` or ``--regularization_norm REGULARIZATION_NORM``
    norm used in regularization.

  - ``--gpu [GPU ...]``
    A list of gpu ids, e.g. 0 1 2 4

  - ``--mix_cpu_gpu``         
    Training a knowledge graph embedding model with both CPUs and GPUs.The embeddings are stored in CPU memory and the training is performed in GPUs.This is usually used for training a large knowledge graph embeddings.

  - ``--valid``               
    Evaluate the model on the validation set in the training.

  - ``--rel_part``         
    Enable relation partitioning for multi-GPU training.

  - ``--async_update``
    Allow asynchronous update on node embedding for multi-GPU training. This overlaps CPU and GPU computation to speed up.

dglke_eval
^^^^^^^^^^^^

  - ``--model_name {TransE, TransE_l1, TransE_l2, TransR, RESCAL, DistMult, ComplEx, RotatE}``
    The models provided by DGL-KE.

  - ``--data_path DATA_PATH``
    The name of the knowledge graph stored under data_path. If it is one ofthe builtin knowledge grpahs such as FB15k, DGL-KE will automatically download the knowledge graph and keep it under data_path.

  - ``--dataset DATASET``     
    dataset name, under data_path

  - ``--format FORMAT``
    The format of the dataset. For builtin knowledge graphs,the foramt should be *built_in*. For users own knowledge graphs,it needs to be *raw_udd_{htr}* or *udd_{htr}*.

  - ``--data_files [DATA_FILES ...]``
    A list of data file names. This is used if users want to train KGE on their own datasets. If the format is *raw_udd_{htr}*, users need to provide *train_file* [*valid_file*] [*test_file*]. If the format is *udd_{htr}*, users need to provide *entity_file* *relation_file* *train_file* [*valid_file*] [*test_file*]. In both cases, *valid_file* and *test_file* are optional.

  - ``--delimiter DELIMITER``
    Delimiter used in data files. Note all files should use the same delimiter.

  - ``--model_path MODEL_PATH``
    The place where models are saved.

  - ``--batch_size_eval BATCH_SIZE_EVAL``
    Batch size used for eval and test

  - ``--neg_sample_size_eval NEG_SAMPLE_SIZE_EVAL``
    Negative sampling size for testing

  - ``--neg_deg_sample_eval``
    Negative sampling proportional to vertex degree for testing.

  - ``--hidden_dim HIDDEN_DIM``
    Hidden dim used by relation and entity

  - ``-g GAMMA`` or ``--gamma GAMMA``
    The margin value in the score function. It is used by *TransX* and *RotatE*.

  - ``--eval_percent EVAL_PERCENT``
    Randomly sample some percentage of edges for evaluation.

  - ``--no_eval_filter`` 
    Disable filter positive edges from randomly constructed negative edges for evaluation.

  - ``--gpu [GPU ...]``
    A list of gpu ids, e.g. 0 1 2 4

  - ``--mix_cpu_gpu``         
    Training a knowledge graph embedding model with both CPUs and GPUs.The embeddings are stored in CPU memory and the training is performed in GPUs.This is usually used for training a large knowledge graph embeddings. 

  - ``-de`` or ``--double_ent``
    Double entitiy dim for complex number It is used by *RotatE*.

  - ``-dr`` or ``--double_rel``
    Double relation dim for complex number.

  - ``--num_proc NUM_PROC`` 
    The number of processes to train the model in parallel.In multi-GPU training, the number of processes by default is set to match the number of GPUs. If set explicitly, the number of processes needs to be divisible by the number of GPUs.

  - ``--num_thread NUM_THREAD``
    The number of CPU threads to train the model in each process. This argument is used for multi-processing training.


dglke_dist_train
^^^^^^^^^^^^^^^^^

  - ``--model_name {TransE, TransE_l1, TransE_l2, TransR, RESCAL, DistMult, ComplEx, RotatE}``
    The models provided by DGL-KE.

  - ``--data_path DATA_PATH``
    The path of the directory where DGL-KE loads knowledge graph data.

  - ``--dataset DATA_SET``
    The name of the knowledge graph stored under data_path. If it is one ofthe builtin knowledge grpahs such as FB15k, DGL-KE will automatically download the knowledge graph and keep it under data_path.

  - ``--format FORMAT``
    The format of the dataset. For builtin knowledge graphs,the foramt should be *built_in*. For users own knowledge graphs,it needs to be *raw_udd_{htr}* or *udd_{htr}*.

  - ``--data_files [DATA_FILES ...]``
    A list of data file names. This is used if users want to train KGE on their own datasets. If the format is *raw_udd_{htr}*, users need to provide *train_file* [*valid_file*] [*test_file*]. If the format is *udd_{htr}*, users need to provide *entity_file* *relation_file* *train_file* [*valid_file*] [*test_file*]. In both cases, *valid_file* and *test_file* are optional.

  - ``--save_path SAVE_PATH``
    The path of the directory where models and logs are saved.

  - ``--no_save_emb``         
    Disable saving the embeddings under save_path.

  - ``--max_step MAX_STEP``   
    The maximal number of steps to train the model. A step trains the model with a batch of data.

  - ``--batch_size BATCH_SIZE``
    The batch size for training.

  - ``--batch_size_eval BATCH_SIZE_EVAL``
    The batch size used for validation and test.

  - ``--neg_sample_size NEG_SAMPLE_SIZE``
    The number of negative samples we use for each positive sample in the training.

  - ``--neg_deg_sample``
    Construct negative samples proportional to vertex degree in the training. When this option is turned on, the number of negative samples per positive edge will be doubled. Half of the negative samples are generated uniformly whilethe other half are generated proportional to vertex degree.

  - ``--neg_deg_sample_eval``
    Construct negative samples proportional to vertex degree in the evaluation.

  - ``--neg_sample_size_eval NEG_SAMPLE_SIZE_EVAL``
    The number of negative samples we use to evaluate a positive sample.

  - ``--eval_percent EVAL_PERCENT``
    Randomly sample some percentage of edges for evaluation.

  - ``--no_eval_filter`` 
    Disable filter positive edges from randomly constructed negative edges for evaluation.

  - ``-log LOG_INTERVAL``
    Print runtime of different components every *x* steps.

  - ``--eval_interval EVAL_INTERVAL``
    Print evaluation results on the validation dataset every *x* stepsif validation is turned on.

  - ``--test``
    Evaluate the model on the test set after the model is trained.

  - ``--num_proc NUM_PROC`` 
    The number of processes to train the model in parallel.In multi-GPU training, the number of processes by default is set to match the number of GPUs. If set explicitly, the number of processes needs to be divisible by the number of GPUs.

  - ``--num_thread NUM_THREAD``
    The number of CPU threads to train the model in each process. This argument is used for multi-processing training.

  - ``--force_sync_interval FORCE_SYNC_INTERVAL``
    We force a synchronization between processes every *x* steps formultiprocessing training. This potentially stablizes the training processto get a better performance. For multiprocessing training, it is set to 1000 by default.

  - ``--hidden_dim HIDDEN_DIM``
    The embedding size of relation and entity.

  - ``--lr LR``          
    The learning rate. DGL-KE uses Adagrad to optimize the model parameters.

  - ``-g GAMMA`` or ``--gamma GAMMA``
    The margin value in the score function. It is used by *TransX* and *RotatE*.

  - ``-de`` or ``--double_ent``
    Double entitiy dim for complex number It is used by *RotatE*.

  - ``-dr`` or ``--double_rel``
    Double relation dim for complex number.

  - ``-adv`` or ``--neg_adversarial_sampling``
    Indicate whether to use negative adversarial sampling.It will weight negative samples with higher scores more.

  - ``-a ADVERSARIAL_TEMPERATURE`` or ``--adversarial_temperature ADVERSARIAL_TEMPERATURE``
    The temperature used for negative adversarial sampling.

  - ``-rc REGULARIZATION_COEF`` or ``--regularization_coef REGULARIZATION_COEF``
    The coefficient for regularization.

  - ``-rn REGULARIZATION_NORM`` or ``--regularization_norm REGULARIZATION_NORM``
    norm used in regularization.

  - ``--path PATH``
    Path of distributed workspace.

  - ``--ssh_key SSH_KEY``     
    ssh private key.

  - ``--ip_config IP_CONFIG``
    Path of IP configuration file.

  - ``--num_client_proc NUM_CLIENT_PROC``
    Number of worker processes on each machine.


dglke_partition
^^^^^^^^^^^^^^^

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
  - ``--data_path DATA_PATH``
    The root path of all dataset including id mapping files. Default: 'data'

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
    The entity ID mapping file. If not provided we will use the mapping file in ``--data_path`` according to the config.json under ``--model_path``.

  - ``--rel_mfile`` (Optional)
    The relation ID mapping file. If not provided we will use the mapping file in ``--data_path`` according to the config.json under ``--model_path``.

dglke_emb_sim
^^^^^^^^^^^^^^^
  - ``--emb_file EMB_DATA_PATH``
    The numpy file containing the embeddings.

  - ``--format FORMAT``
    The format of the input data, specified in ``l_r``. Ideally, user should provides two files, one for left objects and one for right objects. But we also allow users to use *\** to represent *all* of the embeddings. For exmpale, ``l_*`` only requires users to provide a file containing left objects and use the whole embedding set as right; ``*_r`` only requires users to provide a file containing right objects and use the whole embedding set as left; even users can specify a single *\** to treat the whole embedding set as both left and right. 

  - ``--data_files [DATA_FILES ...]``
    A list of data file names. This is used to provide necessary files containing the requried data according to the format, e.g., for ``e_e``, two files are required as h_data and t_data, while for ``e_*``, one file is required as t_data, and for ``*`` this argument can be omited.

  - ``--raw_data``
    A flag tells whether the data profiled in data_files is in the raw object naming space or in mapped id space. If True, the data is in the original naming space and the inference program will do the id translation according to id mapping files. If False, the data is just intergers and it is assumed that user has already done the id translation. Default: False.

  - ``--exec_mode``
    How to calculate scores for element pairs and calculate topK. Possible candidates include: ``pairwise``， ``all``, ``batch_left`` 

  - ``--topk NUM_OF_K``
    How many results are returned. Default:10.

  - ``--sim_func SIM_FUNC_NAME``
    What kind of distance function is used in ranking and will be output. It support five functions: ``cosine`` (score = $\\frac{x \\cdot y}{||x||_2||y||_2}$), ``l2`` (score = $-||x - y||_2$), ``l1`` (score = $-||x - y||_1$), ``dot`` (score = $x \\cdot y$)) and ``ext_jaccard`` (score = $\\frac{x \\cdot y}{||x||_{2}^{2} + ||y||_{2}^{2} - x \\cdot y}$).

  - ``--gpu GPU_ID``
    GPU device to use in inference, by default it uses CPU. Default: -1.(CPU)

  - ``--mfile`` (Optional)
    ID mapping file. 

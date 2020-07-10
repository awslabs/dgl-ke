DGL-KE Command Line Arguments
----------------------------------

dglke_eval
^^^^^^^^^^^^


dglke_dist_train
^^^^^^^^^^^^^^^^^


dglke_partition
^^^^^^^^^^^^^^^

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

DGL-KE Command Line
----------------------------------

DGL-KE provides four commands to users:

``dglke_train`` trains KG embeddings on CPUs or GPUs in a single machine and saves the trained node embeddings and relation embeddings into the file. 

``dglke_eval`` reads the pre-trained embeddings and evaluates the embeddings with a link prediction task on the test set. This is a common task used for evaluating the quality of pre-trained KG embeddings.

``dglke_partition`` partitions the given knowledge graph into ``N`` parts by the METIS partition algorithm. Different partitions will be stored on different machines in distributed training. You can find more details about the METIS partition algorithm in this `link`__.

.. __: http://glaros.dtc.umn.edu/gkhome/metis/metis/overview

``dglke_dist_train`` launches a set of processes in the cluster for distributed training.


Training on Multi-Core
^^^^^^^^^^^^^^^^^^^^^^^

Multi-core processors are very common and widely used in modern computer architecture. DGL-KE is optimized on multi-core processors for the best system performance. In DGL-KE, we use multi-processes instead of multi-threads for parallel training. In this design, the entity embeddings and relation embeddings will be stored in a global shared-memory and all the trainer processes can read and write it. All the processes will train the global model in a *Hogwild* style.

.. image:: ../images/multi-core.png
    :width: 400

The following command trains the ``transE`` model on ``FB15k`` dataset on a multi-core machine::

  dglke_train --model_name TransE_l2 --dataset FB15k --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 \
  --gamma 19.9 --lr 0.25 --max_step 3000 --log_interval 100 --batch_size_eval 16 --test -adv \
  --regularization_coef 1.00E-09 --num_thread 1 --num_proc 8

After training, you will see the following messages::

    -------------- Test result --------------
    Test average MRR : 0.6520483281422476
    Test average MR : 43.725415178344704
    Test average HITS@1 : 0.5257063533713666
    Test average HITS@3 : 0.7524081190431853
    Test average HITS@10 : 0.8479202993008413
    -----------------------------------------


``--num_proc`` indicates that we will launch ``8`` processes in parallel for the training task, and ``--num_thread`` indicates that each process will use ``1`` thread. Typically, ``num_proc * num_thread`` is set to ``<=`` the ``number_of _cores`` of the current machine for the best performance. For example, when the number of processes is the same as the number of CPU cores, a user should use one thread in each process.

``--model_name`` is used to specify our model, including ``TransE_l2``, ``TransE_l1``, ``DistMult``, ``ComplEx``, ``TransR``, ``RESCAL``, and ``RotatE``.

``--dataset`` is used to choose a built-in dataset, including ``FB15k``, ``FB15k-237``, ``wn18``, ``wn18rr``, and ``Freebase``. See more details about the built-in KG on this `page`__.

.. __: ./train_built_in.html

``--batch_size``, ``--neg_batch_size`` is the hyper-parameter used for training sampler, and ``--batch_size_eval`` is the hyper-parameter used for the test.

``--hidden_dim`` defines the dimension size of the KG embeddings. ``--gamma`` is a hyper-parameter to initialize embeddings. ``--regularization_coef`` is the hyper-parameter for regularization.

``--lr`` is used to set the learning rate for our optimization algorithm. ``--max_step`` defines the maximal learning steps for the training task. Note that, the total steps in our training is ``max_step * num_proc``. With multi-processing, we need to adjust the number of ``--max_step`` in each process. Usually, we only need the total number of steps performed by all processes equal to the number of steps performed in the single-process training.

``-adv`` indicates whether to use negative adversarial sampling. It will weight negative samples with higher scores more.

``--log_interval`` indicates that on every ``100`` steps we print the training loss on the screen like this::

   [proc 7][Train](100/500) average pos_loss: 0.7686050720512867
   [proc 7][Train](100/500) average neg_loss: 0.6058262066915632
   [proc 7][Train](100/500) average loss: 0.6872156363725662
   [proc 7][Train](100/500) average regularization: 8.930973201586312e-06
   [proc 7][Train] 100 steps take 22.813 seconds
   [proc 7]sample: 0.226, forward: 13.125, backward:

As we can see, every 100 steps will take ``22.8`` seconds on each process.

``--test`` indicates that we will do an evaluation after training. It could print the following outputs to the screen::

    training takes 37.735950231552124 seconds
    -------------- Test result --------------
    Test average MRR : 0.47615999491724303
    Test average MR : 58.97734929153053
    Test average HITS@1 : 0.28428501295051717
    Test average HITS@3 : 0.6277276497773865
    Test average HITS@10 : 0.775862944592101
    -----------------------------------------
    testing takes 110.887 seconds

After training, we can see a new folder ``ckpts/TransE_l2_FB15k_0``, which stores our training log and trained KG embeddings. Users can set ``--no_save_emb`` to stop saving embedding to the file. 


Training on single GPU
^^^^^^^^^^^^^^^^^^^^^^^

Training knowledge graph embedding contains large numbers of tensor computation, which can be accelerated by GPU. DGL-KE can run on single-GPU, as well as the multi-GPU machine. Also, it can run in a *mix-gpu-cpu* environment, where the embedding data cannot be fit into GPU memory.

The following command trains the ``transE`` model on ``FB15k`` on a single GPU::

    dglke_train --model_name TransE_l2 --dataset FB15k --batch_size 1000 --log_interval 1000 \
    --neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 400 --gamma 19.9 \
    --lr 0.25 --batch_size_eval 16 --test -adv --gpu 0 --max_step 24000

Most of the options here we have already seen in the previous section. The only difference is that we add ``--gpu 0`` here to indicate that we will use 1 GPU to train our model. Compared to the cpu training, every 100 steps only takes ``0.68`` seconds on each Nvidia v100 GPU, which is much faster ``22.8`` second in CPU training::

    [proc 0][Train](24000/24000) average pos_loss: 0.2704171320796013
    [proc 0][Train](24000/24000) average neg_loss: 0.39646861135959627
    [proc 0][Train](24000/24000) average loss: 0.33344287276268003
    [proc 0][Train](24000/24000) average regularization: 0.0017754920991137624
    [proc 0][Train] 100 steps take 0.680 seconds


Mix CPU-GPU training
^^^^^^^^^^^^^^^^^^^^^

By default, DGL-KE keeps all node and relation embeddings in GPU memory for single-GPU training. Therefore, it cannot train embeddings of large knowledge graphs because the capacity of GPU memory typically is much smaller than the CPU memory. So if your KG embedding is too large to fit into the GPU memory, you can use ``--mix_cpu_gpu`` training::

    dglke_train --model_name TransE_l2 --dataset FB15k --batch_size 1000 --log_interval 1000 \
    --neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 400 --gamma 19.9 \
    --lr 0.25 --batch_size_eval 16 --test -adv --gpu 0 --max_step 24000 --mix_cpu_gpu

The ``--mix_cpu_gpu`` training will keep node and relation embeddings in CPU memory and perform batch computation in GPU. In this way, you can train very large KG embeddings as long as your cpu memory can handle it. While the training speed of *mix_cpu_gpu* training will be slower than pure GPU training::

    [proc 0][Train](24000/24000) average pos_loss: 0.2693914473056793
    [proc 0][Train](24000/24000) average neg_loss: 0.39576649993658064
    [proc 0][Train](24000/24000) average loss: 0.3325789734721184
    [proc 0][Train](24000/24000) average regularization: 0.0017816077976021915
    [proc 0][Train] 100 steps take 1.073 seconds
    [proc 0]sample: 0.158, forward: 0.383, backward: 0.214, update: 0.316

As we can see, the *mix_cpu_gpu* training takes ``1.07`` seconds on every 100 steps.


Training on Multi-GPU
^^^^^^^^^^^^^^^^^^^^^^^

DGL-KE also supports multi-GPU training, which can increase performance by distributing training across multiple GPUs. The following figure depicts 4 GPUs on a single machine and connected to the CPU through a PCIe switch. Multi-GPU training automatically keeps node and relation embeddings on CPUs and dispatch batches to different GPUs.

.. image:: ../images/multi-gpu.svg
    :width: 200


The following command shows how to training our ``transE`` model using 4 Nvidia v100 GPUs jointly::

    dglke_train --model_name TransE_l2 --dataset FB15k --batch_size 1000 --log_interval 1000 \
    --neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 400 --gamma 19.9 \
    --lr 0.25 --batch_size_eval 16 --test -adv --gpu 0 1 2 3 --max_step 6000

Compared to single-GPU training, we change ``--gpu 0`` to ``--gpu 0 1 2 3``, and also we change ``--max_step`` from ``24000`` to ``6000``.

Users can add ``--async_update`` option for multi-GPU training. This optimization overlaps batch computation in GPU with gradient updates on CPU to speed up the overall training::

    dglke_train --model_name TransE_l2 --dataset FB15k --batch_size 1000 --log_interval 1000 \
    --neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 400 --gamma 19.9 \
    --lr 0.25 --batch_size_eval 16 --test -adv --gpu 0 1 2 3 --async_update --max_step 6000


``--async_update`` can increase system performance but it could slow down the model convergence. So DGL-KE provides another option called ``--force_sync_interval`` that forces all GPU sync their model on every ``N`` steps. For example, the following command will sync model across GPUs on every 1000 steps::

    dglke_train --model_name TransE_l2 --dataset FB15k --batch_size 1000 --log_interval 1000 \
    --neg_sample_size 200 --regularization_coef=1e-9 --hidden_dim 400 --gamma 19.9 \
    --lr 0.25 --batch_size_eval 16 --test -adv --gpu 0 1 2 3 --async_update --max_step 6000 --force_sync_interval 1000


Evaluation on Pre-Trained Embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, ``dglke_train`` saves the embeddings in the ``ckpts`` folder. Each runs creates a new folder in ``ckpts`` to store the training results. The new folder is named after ``xxxx_yyyy_zz``\ , where ``xxxx`` is the model name, ``yyyy`` is the dataset name, ``zz`` is a sequence number that ensures a unique name for each run. 

The saved embeddings are stored as numpy ndarrays. The node embedding is saved as ``XXX_YYY_entity.npy``.
The relation embedding is saved as ``XXX_YYY_relation.npy``. ``XXX`` is the dataset name and ``YYY`` is the model name.

A user can disable saving embeddings with ``--no_save_emb``. This might be useful for some cases, such as hyperparameter tuning.

``dglke_eval`` reads the pre-trained embeddings and evaluates the embeddings with a link prediction task on the test set. This is a common task used for evaluating the quality of pre-trained KG embeddings. The following command evaluates the pre-trained KG embedding on multi-cores::

    dglke_eval --model_name TransE_l2 --dataset FB15k --hidden_dim 400 --gamma 19.9 --batch_size_eval 16 \
    --num_thread 1 --num_proc 8 --model_path ~/my_task/ckpts/TransE_l2_FB15k_0/

We can also use GPUs in our evaluation tasks::

    dglke_eval --model_name TransE_l2 --dataset FB15k --hidden_dim 400 --gamma 19.9 --batch_size_eval 16 \
    --gpu 0 1 2 3 4 5 6 7 --model_path ~/my_task/ckpts/TransE_l2_FB15k_0/




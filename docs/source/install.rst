Installation Guide
----------------------------------

DGL-KE works with both Linux and MacOS, and it requires Python version 3.5 or later (Python 3.4 or earlier is not tested). DGL-KE can run on both pytorch and mxnet, please refer the following pages to install pytorch or mxnet.

`Pytorch installation`__

`MXNet installation`__

.. __: https://pytorch.org/
.. __: https://mxnet.apache.org/


Install DGL
^^^^^^^^^^^^^^^^^^^^^^^^

DGL-KE is implemented on DGL. You can install DGL using pip::

    pip install dgl

or you can install dgl from source::

    git clone --recursive https://github.com/dmlc/dgl.git
    cd dgl
    mkdir build
    cd build
    cmake ../
    amke -j4


Install DGL-KE 
^^^^^^^^^^^^^^^^

The fastest way to install DGL-KE is using pip::

    pip install dglke

or you can install DGL-KE from source::

    git clone https://github.com/awslabs/dgl-ke.git
    cd dgl-ke/python
    sudo python3 setup.py install


Test
^^^^^^^^

Once you install DGL-KE successfully, you can test it by the following command::

    dglke_train --model_name TransE_l2 --dataset FB15k --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 \
    --gamma 19.9 --lr 0.25 --max_step 500 --log_interval 100 --batch_size_eval 16 --test -adv \
    --regularization_coef 1.00E-09 --num_thread 1 --num_proc 8

You can see the following output::

    training takes 37.735950231552124 seconds
    -------------- Test result --------------
    Test average MRR : 0.47615999491724303
    Test average MR : 58.97734929153053
    Test average HITS@1 : 0.28428501295051717
    Test average HITS@3 : 0.6277276497773865
    Test average HITS@10 : 0.775862944592101
    -----------------------------------------
    testing takes 110.887 seconds

import argparse
class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        self.add_argument('--score_func', default='TransE',
                          choices=['TransE', 'TransE_l1', 'TransE_l2', 'TransR',
                                   'RESCAL', 'DistMult', 'ComplEx', 'RotatE',
                                   'SimplE', 'AttH'],
                          help='The score function provided by DGL-KE.')
        self.add_argument('--model', default='BaseModel',
                          help='the name of model. Default is BaseModel.')
        self.add_argument('--encoder', default='KGE',
                          help='Which encoder is used to encode graph data.')
        self.add_argument('--decoder', type=str, default='KGE',
                          help='The decoders are used to decode graph data.')
        self.add_argument('--data_path', type=str, default='data',
                          help='The path of the directory where DGL-KE loads knowledge graph data.')
        self.add_argument('--dataset', type=str, default='FB15k',
                          help='The name of the builtin knowledge graph. Currently, the builtin knowledge ' \
                               'graphs include FB15k, FB15k-237, wn18, wn18rr and Freebase. ' \
                               'DGL-KE automatically downloads the knowledge graph and keep it under data_path.')
        self.add_argument('--format', type=str, default='built_in',
                          help='The format of the dataset. For builtin knowledge graphs,' \
                               'the foramt should be built_in. For users own knowledge graphs,' \
                               'it needs to be raw_udd_{htr} or udd_{htr}.')
        self.add_argument('--data_files', type=str, default=None, nargs='+',
                          help='A list of data file names. This is used if users want to train KGE' \
                               'on their own datasets. If the format is raw_udd_{htr},' \
                               'users need to provide train_file [valid_file] [test_file].' \
                               'If the format is udd_{htr}, users need to provide' \
                               'entity_file relation_file train_file [valid_file] [test_file].' \
                               'In both cases, valid_file and test_file are optional.')
        self.add_argument('--delimiter', type=str, default='\t',
                          help='Delimiter used in data files. Note all files should use the same delimiter.')
        self.add_argument('--save_path', type=str, default='ckpts',
                          help='the path of the directory where models and logs are saved.')
        self.add_argument('--no_save_model', action='store_false', dest='save_model',
                          help='Disable saving model parameters under save_path.')
        self.add_argument('--max_step', type=int, default=80000,
                          help='The maximal number of steps to train the model.' \
                               'A step trains the model with a batch of data.')
        self.add_argument('--batch_size_eval', type=int, default=8,
                          help='The batch size used for validation and test.')
        self.add_argument('--neg_sample_size', type=int, default=256,
                          help='The number of negative samples we use for each positive sample in the training.')
        self.add_argument('--neg_deg_sample', action='store_true',
                          help='Construct negative samples proportional to vertex degree in the training.' \
                               'When this option is turned on, the number of negative samples per positive edge' \
                               'will be doubled. Half of the negative samples are generated uniformly while' \
                               'the other half are generated proportional to vertex degree.')
        self.add_argument('--neg_deg_sample_eval', action='store_true',
                          help='Construct negative samples proportional to vertex degree in the evaluation.')
        self.add_argument('--neg_sample_size_eval', type=int, default=-1,
                          help='The number of negative samples we use to evaluate a positive sample.')
        self.add_argument('--eval_percent', type=float, default=1,
                          help='Randomly sample some percentage of edges for evaluation.')
        self.add_argument('--no_eval_filter', action='store_false', dest='eval_filter',
                          help='Disable filter positive edges from randomly constructed negative edges for evaluation')
        self.add_argument('--self_loop_filter', action='store_true', dest='self_loop_filter',
                          help='Disable filter triple like (head - relation - head) score for evaluation')
        self.add_argument('-log', '--log_interval', type=int, default=1000,
                          help='Print runtime of different components every x steps.')
        self.add_argument('--fit', action='store_true',
                          help='Train the model on the training set.')
        self.add_argument('--test', action='store_true',
                          help='Evaluate the model on the test set after the model is trained.')
        self.add_argument('--num_proc', type=int, default=1,
                          help='The number of processes to train the model in parallel.' \
                               'In multi-GPU training, the number of processes by default is set to match the number of GPUs.' \
                               'If set explicitly, the number of processes needs to be divisible by the number of GPUs.')
        self.add_argument('--num_thread', type=int, default=1,
                          help='The number of CPU threads to train the model in each process.' \
                               'This argument is used for multiprocessing training.')
        self.add_argument('--hidden_dim', type=int, default=400,
                          help='The embedding size of relation and entity')
        self.add_argument('--lr', type=float, default=0.01,
                          help='The learning rate. DGL-KE uses Adagrad to optimize the model parameters.')
        self.add_argument('-g', '--gamma', type=float, default=12.0,
                          help='The margin value in the score function. It is used by TransX and RotatE.')
        self.add_argument('-de', '--double_ent', action='store_true',
                          help='Double entity dim for complex number or canonical polyadic. It is used by RotatE and SimplE.')
        self.add_argument('-dr', '--double_rel', action='store_true',
                          help='Double relation dim for complex number or canonical polyadic. It is used by RotatE and SimplE')
        self.add_argument('-adv', '--neg_adversarial_sampling', action='store_true',
                          help='Indicate whether to use negative adversarial sampling.' \
                               'It will weight negative samples with higher scores more.')
        self.add_argument('-a', '--adversarial_temperature', default=1.0, type=float,
                          help='The temperature used for negative adversarial sampling.')
        self.add_argument('-rc', '--regularization_coef', type=float, default=0.000002,
                          help='The coefficient for regularization.')
        self.add_argument('-rn', '--regularization_norm', type=int, default=3,
                          help='norm used in regularization.')
        self.add_argument('-pw', '--pairwise', action='store_true',
                          help='Indicate whether to use pairwise loss function. '
                               'It compares the scores of a positive triple and a negative triple')
        self.add_argument('--loss_genre', default='Logsigmoid',
                          choices=['Hinge', 'Logistic', 'Logsigmoid', 'BCE'],
                          help='The loss function used to train KGEM.')
        self.add_argument('-m', '--margin', type=float, default=1.0,
                          help='hyper-parameter for hinge loss.')
        self.add_argument('--label_smooth', type=float, default=.0,
                          help='use label smoothing for training.')
        self.add_argument('--num_node', type=int, default=1,
                          help='Number of node used for distributed training')
        self.add_argument('--node_rank', type=int, default=0,
                          help='The rank of node, ranged from [0, num_node - 1]')
        self.add_argument('--init', type=str, default='uniform',
                          choices=['uniform', 'xavier', 'constant'],
                          help='Initial strategy for embeddings.')
        # dataloader
        self.add_argument('--batch_size', type=int, default=1024,
                          help='The batch size for training.')
        self.add_argument('--num_workers', type=int, default=0,
                          help='Number of process to fetch data for training/validation dataset.')
        self.add_argument('--shuffle_data', type=bool, default=False,
                          help='Whether to shuffle data for training.')
        # hyper-parameter for hyperbolic embeddings
        self.add_argument('--init_scale', type=float, default=0.001,
                          help='Initialization scale for entity embedding, relation embedding, curvature, attention in hyperbolic embeddings')
        self.add_argument('--optimizer', type=str, default='Adagrad',
                          choices=['Adagrad', 'Adam'],
                          help='Optimizer for kg embeddings')
        self.add_argument('--no_save_log', action='store_false', dest='save_log',
                          help='If specified, dglke will not save log and result file to save path.')
        self.add_argument('--tqdm', action='store_true', dest='tqdm',
                          help='Use tqdm to visualize training and evaluation process. Note this might drag speed of process 0 for multi-GPU training.')
        self.add_argument('--profile', action='store_true', dest='profile',
                          help='Profile the process to test training speed. Used for debug.')
        self.add_argument('--inverse_rel', action='store_true', dest='inverse_rel',
                          help='If specified, create a->inv_rel->b  based on b->rel->a.')

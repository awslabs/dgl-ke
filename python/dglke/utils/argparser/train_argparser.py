from .common_argparser import CommonArgParser
class TrainArgParser(CommonArgParser):
    def __init__(self):
        super(TrainArgParser, self).__init__()

        self.add_argument('--gpu', type=int, default=[-1], nargs='+',
                          help='A list of gpu ids, e.g. 0 1 2 4')
        self.add_argument('--mix_cpu_gpu', action='store_true',
                          help='Training a knowledge graph embedding model with both CPUs and GPUs.' \
                               'The embeddings are stored in CPU memory and the training is performed in GPUs.' \
                               'This is usually used for training a large knowledge graph embeddings.')
        self.add_argument('--valid', action='store_true',
                          help='Evaluate the model on the validation set in the training.')
        self.add_argument('--eval_interval', type=int, default=1,
                          help='Print evaluation results on the validation dataset every x steps' \
                               'if validation is turned on')
        self.add_argument('--rel_part', action='store_true',
                          help='Enable relation partitioning for multi-GPU training.')
        self.add_argument('--has_edge_importance', action='store_true',
                          help='Allow providing edge importance score for each edge during training.' \
                               'The positive score will be adjusted ' \
                               'as pos_score = pos_score * edge_importance')


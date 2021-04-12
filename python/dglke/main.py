from .utils.argparser import TrainArgParser
from .utils.misc import prepare_args
from .utils.logging import Logger
from data import get_dataset, TrainDataset, TestDataset, ValidDataset
from data.dataloader import KGETrainDataLoaderGenerator, KGEEvalDataLoaderGenerator
from .utils import EMB_INIT_EPS
from .nn.modules import KGEDecoder
from .nn.loss import sLCWAKGELossGenerator
from .nn.loss import BCELoss, HingeLoss, LogisticLoss, LogsigmoidLoss
from .regularizer import Regularizer
from .nn.modules import TransEScore
from .nn.metrics import RankingMetricsEvaluator
from functools import partial
import torch as th
import time
from .nn.modules import Model

def create_dataset_graph(args):
    g = None
    dataset = get_dataset(data_path=args.data_path,
                          data_name=args.dataset,
                          format_str=args.format,
                          delimiter=args.delimiter,
                          files=args.data_files,
                          has_edge_importance=args.has_edge_importance)
    train_dataset, eval_dataset, test_dataset = None, None, None
    # create training dataset needed parameters
    train_dataset = TrainDataset(dataset, args)
    g = train_dataset.g
    args.strict_rel_part = args.mix_cpu_gpu and (train_dataset.cross_part is False)
    args.rel_parts = train_dataset.rel_parts if args.strict_rel_part else None
    args.n_entities = dataset.n_entities
    args.n_relations = dataset.n_relations

    if args.neg_sample_size_eval < 0:
        args.neg_sample_size_eval = dataset.n_entities

    if args.valid or args.test:
        if args.valid:
            assert dataset.valid is not None, 'validation set is not provided'
            eval_dataset = ValidDataset(dataset, args)
            g = eval_dataset.g
        if args.test:
            assert dataset.test is not None, 'test set is not provided'
            # no sampler is needed
            test_dataset = TestDataset(dataset, args)
            g = test_dataset.g
    return [train_dataset, eval_dataset, test_dataset], g

def create_dataloader_generator(args):
    train_dataloader, eval_dataloader = None, None
    if args.model == 'KGE':
        train_metadata = {'neg_sample_size': args.neg_sample_size,
                          'chunk_size': args.neg_sample_size,
                          'chunk': True}
        train_dataloader = KGETrainDataLoaderGenerator(shuffle=args.shuffle_data,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              metadata=train_metadata)
        eval_metadata = {'num_nodes': args.n_entities,
                         'chunk_size': 1,
                         'neg_sample_size': args.n_entities,
                         'chunk': True}
        eval_dataloader = KGEEvalDataLoaderGenerator(batch_size=args.batch_size_eval,
                                            num_workers=args.num_workers,
                                            metadata=eval_metadata)
    else:
        raise NotImplementedError(f'Dataloader for {args.model} is not supported yet.')
    return train_dataloader, eval_dataloader

# ! dataloader is associated with how the encoder will be created.
def create_encoder(args):
    if args.encoder == 'KGE':
        from .nn.modules import KGEEncoder
        if args.init == 'uniform':
            emb_init = (args.gamma + EMB_INIT_EPS) / args.hidden_dim
            init_func = [partial(th.nn.init.uniform_, a=-emb_init, b=emb_init), partial(th.nn.init.uniform_, a=-emb_init, b=emb_init)]
        else:
            raise NotImplementedError(f'init {args.init} is not implemented yet.')
        encoder = KGEEncoder(hidden_dim=args.hidden_dim,
                             n_entity=args.n_entities,
                             n_relation=args.n_relations,
                             init_func=init_func)
        return encoder
    else:
        raise NotImplementedError(f'encoder {args.encoder} is not supported yet.')

def create_decoder(args):
    if args.decoder == 'KGE':
        decoder = KGEDecoder(args.decoder)
        # add score function
        if 'TransE' in args.score_func:
            dist = args.score_func.split('_')[-1]
            score_func = TransEScore(args.gamma, dist_func=dist if dist != '' else 'l1')
        else:
            raise NotImplementedError(f'score func {args.score_func} is not implemented yet.')
        decoder.attach_score_func(score_func)

        # add loss generator for each decoder
        loss_gen = sLCWAKGELossGenerator(neg_adversarial_sampling=args.neg_adversarial_sampling,
                                         adversarial_temperature=args.adversarial_temperature,
                                         pairwise=args.pairwise,
                                         label_smooth=args.label_smooth)

        # set criterion for loss generator
        if args.loss_genre == 'Logsigmoid':
            criterion = LogsigmoidLoss()
        elif args.loss_genre == 'Hinge':
            criterion = HingeLoss(margin=args.margin)
        elif args.loss_genre == 'Logistic':
            criterion = LogisticLoss()
        elif args.loss_genre == 'BCE':
            criterion = BCELoss()
        else:
            raise ValueError(f'criterion {args.loss_genre} is not supported.')
        loss_gen.set_criterion(criterion)
        decoder.attach_loss_generator(loss_gen)

        # add metrics evaluator for decoder
        metrics_evaluator = RankingMetricsEvaluator(args.eval_filter)
        decoder.attach_metrics_evaluator(metrics_evaluator)
        return decoder
    else:
        raise NotImplementedError(f'decoder {args.decoder} is not implemented yet.')

def create_optimizer(args, model):
    if args.optimizer == 'Adagrad':
        optimizer = th.optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    return optimizer


def main():
    args = TrainArgParser().parse_args()
    prepare_args(args)

    # print configuration
    print('-' * 50)
    for k, v in vars(args).items():
        print('{} : {}'.format(k, v))
    print('-' * 50)

    # setup logger
    logger = Logger(args.save_path, save_log=args.save_log)

    init_time_start = time.time()
    ############# create model #######################

    # get dataset
    dataset, g = create_dataset_graph(args)
    # create encoder
    encoder = create_encoder(args)
    # create decoders
    decoder = create_decoder(args)

    model = Model(args.gpu, encoder, decoder, args.model)

    # attach graph, dataset, dataloader_generator
    model.attach_graph(g)
    model.attach_dataset(dataset)
    dataloader = create_dataloader_generator(args)
    model.attach_dataloader(dataloader)

    print('Total initialize time {:.3f} seconds'.format(time.time() - init_time_start))
    # train
    if args.fit:
        # create regularizer
        regularizer = Regularizer(coef=args.regularization_coef, norm=args.regularization_norm)
        #  set training parameters
        model.set_training_params(args)
        if args.gpu[0] != -1:
            device = th.device(f'cuda:{args.gpu[0]}')
            model = model.to(device)
        optimizer = create_optimizer(args, model)
        model.fit(max_step=args.max_step,
                  regularizer=regularizer,
                  world_size=args.num_node * args.num_proc,
                  use_tqdm=args.tqdm,
                  log_interval=args.log_interval,
                  val=args.valid,
                  val_interval=args.eval_interval,
                  optimizer=optimizer,
                  profile=args.profile,
                  logger=logger)

    if args.save_model:
        model.save(args.save_path)

    if args.test:
        model.set_test_params(args)
        model.test(world_size=args.num_node * args.num_proc,
                   use_tqdm=args.tqdm,
                   profile=args.profile,
                   logger=logger)

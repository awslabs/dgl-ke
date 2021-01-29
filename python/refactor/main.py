from utils.argparser import TrainArgParser
from utils.misc import set_seed, get_compatible_batch_size
from utils.logging import Logger
from utils.data.dataset import get_dataset, TrainDataset, EvalDataset, NegDataset, TestDataset
from utils.data.sampler import SequentialEpochSampler, SequentialTotalStepSampler
from utils.data.dataloader import DataLoaderFactory, LCWADataLoaderWrapper, NegChunkDataLoaderWrapper
from utils import EMB_INIT_EPS
from .kge_model import KGEModel
from nn.modules.dglke_modules import KGEDecoder, KGEEncoder
from nn.loss import sLCWAKGELossGenerator, LCWAKGELossGenerator
from nn.loss import BCELoss, HingeLoss, LogisticLoss, LogsigmoidLoss
from .regularizer import Regularizer
from nn.modules import TransEScore
from nn.metrics import KGEMetricsEvaluator
from .optim import Adagrad
from functools import partial
import torch as th
import time

def main():
    args = TrainArgParser().parse_args()
    set_seed(args.seed)
    # set up logger
    world_size = args.num_proc * args.num_node
    init_time_start = time.time()

    ############# create model #######################
    model = KGEModel(max_step=args.max_step,
                     tqdm=args.tqdm,
                     world_size=world_size,
                     gpu=args.gpu,
                     force_sync_interval=args.force_sync_interval,
                     num_proc=args.num_proc,
                     log_interval=args.log_interval,
                     valid=args.valid)
    # setup logger
    log_file = 'log.txt'
    result_file = 'result.json'
    logger = Logger(args.save_path, log_file, result_file, save_log=args.save_log)
    model.attach_logger(logger)

    # get dataset
    dataset = get_dataset(data_path=args.data_path,
                          data_name=args.dataset,
                          format_str=args.format,
                          delimiter=args.delimiter,
                          files=args.data_files,
                          has_edge_importance=args.has_edge_importance)

    if args.neg_sample_size_eval < 0:
        args.neg_sample_size_eval = dataset.n_entities

    args.batch_size = get_compatible_batch_size(args.batch_size, args.neg_sample_size)
    args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval, args.neg_sample_size_eval)
    # We should turn on mix CPU-GPU training for multi-GPU training.
    if len(args.gpu) > 1:
        args.mix_cpu_gpu = True
        if args.num_proc < len(args.gpu):
            args.num_proc = len(args.gpu)
    # We need to ensure that the number of processes should match the number of GPUs.
    if len(args.gpu) > 1 and args.num_proc > 1:
        assert args.num_proc % len(args.gpu) == 0, \
            'The number of processes needs to be divisible by the number of GPUs'

    train_samplers = []
    train_datasets = []
    train_wrappers = []

    # create train dataset
    train_dataloader_factory = DataLoaderFactory(batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 pin_memory=args.pin_memory,)

    train_datasets += [TrainDataset(dataset, args, ranks=args.num_proc, has_importance=args.has_edge_importance, has_label=not args.sLCWA)]
    # NFC - use partial here because we can only know the size of dataset when we partition dataset
    train_samplers += [partial(SequentialEpochSampler, batch_size=args.batch_size, max_step=args.max_step)]
    if args.sLCWA:
        train_datasets += [NegDataset(num_of_nodes=dataset.n_entities, batch_size=args.batch_size, max_step=args.max_step)]
        train_samplers += [partial(SequentialTotalStepSampler, batch_size=args.batch_size, max_step=args.max_step)]
        train_wrappers += [NegChunkDataLoaderWrapper(chunk_size=args.neg_sample_size, neg_sample_size=args.neg_sample_size, self_neg=args.self_neg)]
    else:
        train_wrappers += [LCWADataLoaderWrapper(num_of_nodes=dataset.n_entities, lcwa=not args.sLCWA)]

    for d, s in zip(train_datasets, train_samplers):
        train_dataloader_factory.append_dataset_sampler(d, s)
    for wrapper in train_wrappers:
        train_dataloader_factory.append_dataset_wrapper(wrapper)

    model.attach_dataloader_factory(train_dataloader_factory, 'train')

    if args.valid or args.test:
        if len(args.gpu) > 1:
            args.num_test_proc = args.num_proc if args.num_proc < len(args.gpu) else len(args.gpu)
        else:
            args.num_test_proc = args.num_proc

    if args.valid:
        valid_datasets = []
        valid_samplers = []
        valid_wrappers = []
        assert dataset.valid is not None, 'validation set is not provided'
        valid_dataloader_factory = DataLoaderFactory(batch_size=args.batch_size_eval,
                                                     shuffle=False,
                                                     num_workers=args.num_workers,
                                                     pin_memory=args.pin_memory,
                                                     drop_last=False)
        # no sampler is needed
        valid_datasets += [EvalDataset(dataset, args, has_label=True)]
        valid_samplers += [None]
        valid_wrappers += [LCWADataLoaderWrapper(dataset.n_entities, lcwa=args.lcwa)]
        for dataset, sampler in zip(valid_datasets, valid_samplers):
            valid_dataloader_factory.append_dataset_sampler(dataset, sampler)
        for wrapper in valid_wrappers:
            valid_dataloader_factory.append_dataset_wrapper(wrapper)
        model.attach_dataloader_factory(valid_dataloader_factory, 'valid')

    if args.test:
        test_datasets = []
        test_samplers = []
        test_wrappers = []
        assert dataset.test is not None, 'test set is not provided'
        test_dataloader_factory = DataLoaderFactory(batch_size=args.batch_size_eval,
                                                    shuffle=False,
                                                    num_workers=args.num_workers,
                                                    pin_memory=args.pin_memory,
                                                    drop_last=False)
        # no sampler is needed
        test_datasets += [TestDataset(dataset, args, has_label=False)]
        test_samplers += [None]
        test_wrappers += [LCWADataLoaderWrapper(dataset.n_entities, lcwa=not args.sLCWA)]
        for d, s in zip(test_datasets, test_samplers):
            test_dataloader_factory.append_dataset_sampler(d, s)
        for wrapper in test_wrappers:
            test_dataloader_factory.append_dataset_wrapper(wrapper)
        model.attach_dataloader_factory(test_dataloader_factory, 'test')


    # create encoder
    if args.encoder == 'KGE':
        if args.init == 'uniform':
            emb_init = (args.gamma + EMB_INIT_EPS) / args.hidden_dim
            init_func = [partial(th.nn.init.uniform_, a=-emb_init, b=emb_init), partial(th.nn.init.uniform_, a=-emb_init, b=emb_init)]
        else:
            raise NotImplementedError(f'init {args.init} is not implemented yet.')
        encoder = KGEEncoder(hidden_dim=args.hidden_dim,
                             n_entity=dataset.n_entities,
                             n_relation=dataset.n_relations,
                             init_func=init_func)
    else:
        raise NotImplementedError(f'encoder {args.encoder} is not supported yet.')

    model.attach_encoder(encoder)

    # create decoders
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
        if args.sLCWA:
            loss_gen = sLCWAKGELossGenerator(neg_adversarial_sampling=args.neg_adversarial_sampling,
                                             adversarial_temperature=args.adversarial_temperature,
                                             pairwise=args.pairwise,
                                             label_smooth=args.label_smooth)
        else:
            loss_gen = LCWAKGELossGenerator(label_smooth=args.label_smooth)

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
        metrics_evaluator = KGEMetricsEvaluator(args.eval_filter)
        decoder.attach_metrics_evaluator(metrics_evaluator)
    else:
        raise NotImplementedError(f'decoder {args.decoder} is not implemented yet.')
    model.attach_decoder(decoder)

    # set regularizer
    regularizer = Regularizer(coef=args.regularization_coef, norm=args.regularization_norm)

    model.attach_regularizer(regularizer)

    # set optimizer
    optimizer = Adagrad(model.dense_parameters(), model.sparse_parameters(), lr=args.lr)
    model.set_optimizer(optimizer)

    print('Total initialize time {:.3f} seconds'.format(time.time() - init_time_start))
    # print configuration
    print('-' * 50)
    for k, v in vars(args).items():
        print('{} : {}'.format(k, v))
    print('-' * 50)
    # train
    # rel_parts = train_dataset.rel_parts if args.strict_rel_part or args.soft_rel_part else None
    # cross_rels = train_dataset.cross_rels if args.soft_rel_part else None
    if args.fit:
        #  set training parameters
        model.fit()

    if args.test:
        model.test()

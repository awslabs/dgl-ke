import torch as th
import torch.nn as nn
import os
from itertools import chain
from tqdm import trange, tqdm
from dglke.utils import get_scalar
import time

class KEModel(nn.Module):
    r"""Generic module for all encoders & decoders.

    KEModel is a subclass of nn.Module that contains one encoder and one/multiple decoder(s).

    Encoder and decoder are subclass of nn.Module just as KEModel that might contains parameters of embeddings.

    Encoder uses indices(data) provided by dataloader to slice embeddings. Decoder uses both indices(data) and
    embeddings from encoder to generate errors of downstream task for self-supervised signals/supervised training.

    The encoder, decoder are defined once the KEModel is initialized. By attaching graph, dataset and dataloader, the
    model can perform different downstream tasks.
    """
    def __init__(self,
                 gpu=(-1),
                 encoder=None,
                 decoder=None,
                 model_name='BaseModel',):
        super(KEModel, self).__init__()
        self._name = model_name
        self.encoder = encoder
        self.decoder = decoder
        self.gpu = gpu

    def save(self, save_path):
        # save in CPU
        state_dict = {}
        for k, v in self.state_dict().items():
            state_dict[k] = v.cpu()
        file_path = os.path.join(save_path, 'model.pth')
        th.save(state_dict, file_path)

    def load(self, load_path):
        device = th.device('cpu') if -1 in self.gpu else th.device(f'cuda:{self.gpu[0]}')
        file_path = os.path.join(load_path, 'model.pth')
        self.load_state_dict(th.load(file_path, map_location=device))
        self.to(device)

    def set_training_params(self, args):
        self.encoder.set_training_params(args)
        self.decoder.set_training_params(args)

    def set_test_params(self, args):
        self.encoder.set_test_params(args)
        self.decoder.set_test_params(args)

    def test(self,
             world_size=1,
             use_tqdm=False,
             logger=None,
             profile=False):
        self.eval()
        start = time.time()

        self.eval_proc(rank=0, world_size=world_size, use_tqdm=use_tqdm, logger=logger, profile=profile, mode='test')
        logger.print_log(f'test takes {time.time() - start} seconds')

    def sparse_parameters(self):
        # TODO: to be fixed for dgl embedding
        return chain(self.encoder.sparse_parameters(), self.decoder.sparse_parameters())

    def fit(self,
            max_step,
            regularizer,
            world_size=1,
            use_tqdm=False,
            log_interval=1000,
            val=False,
            val_interval=1000,
            optimizer=None,
            profile=False,
            logger=None):

        self.train()
        start = time.time()
        self.train_proc(rank=0,
                        max_step=max_step,
                        regularizer=regularizer,
                        use_tqdm=use_tqdm,
                        world_size=world_size,
                        log_interval=log_interval,
                        val=val,
                        val_interval=val_interval,
                        profile=profile,
                        optimizer=optimizer,
                        logger=logger)

        print(f'training takes {time.time() - start} seconds')



    def prepare_data(self, data):
        pass

    def train_proc(self,
                   rank,
                   max_step,
                   regularizer=None,
                   use_tqdm=False,
                   world_size=1,
                   log_interval=1000,
                   val=False,
                   val_interval=1000,
                   profile=False,
                   optimizer=None,
                   logger=None):
        if profile and rank == 0:
            from pyinstrument import Profiler
            profiler = Profiler()
            profiler.start()
        # if it's multi-GPU training, the adapter will only sample data partition for gpu where gpu_id=gpu_id
        dataset = self.train_dataset.partition(rank, world_size)
        dataloader = self.train_dataloader.generate_dataloader(dataset)
        iter_data = iter(dataloader)
        if len(self.gpu) > 0:
            gpu_id = self.gpu[rank % len(self.gpu)]
        else:
            gpu_id = -1
        logs = []
        train_start = start =  time.time()
        sample_time = 0
        update_time = 0
        forward_time = 0
        backward_time = 0

        iter_range = trange(0, max_step, desc='train') if (rank == 0 and use_tqdm) else range(0, max_step)
        for step in iter_range:
            start1 = time.time()
            loss = {}
            data = next(iter_data)
            sample_time += time.time() - start1
            start1 = time.time()
            encode_results = self.encoder.forward(data, gpu_id)
            decode_results = self.decoder.forward(encode_results, data, gpu_id)
            loss['decode'] = self.decoder.get_loss(decode_results)
            if regularizer is not None:
                loss['reg'] = regularizer.compute_regularization(encode_results)
            total_loss = 0
            for k, v in loss.items():
                total_loss += v
            logs.append({k: get_scalar(v) for k, v in loss.items()})
            forward_time += time.time() - start1
            if rank == 0 and use_tqdm:
                iter_range.set_postfix(loss=f'{total_loss.item(): .4f}')
            start1 = time.time()
            if optimizer is not None:
                optimizer.zero_grad()
            total_loss.backward()
            backward_time += time.time() - start1
            start1 = time.time()
            if optimizer is not None:
                optimizer.step()
            update_time += time.time() - start1

            if (step + 1) % log_interval == 0:
                for k in logs[0].keys():
                    v = sum(l[k] for l in logs) / len(logs)
                    logger.print_log('[proc {}][Train]({}/{}) average {}: {}'.format(rank, (step + 1), max_step, k, v))
                logs = []
                logger.print_log('[proc {}][Train] {} steps take {:.3f} seconds'.format(rank, log_interval,
                                                                             time.time() - start))
                logger.print_log('[proc {}]sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                    rank, sample_time, forward_time, backward_time, update_time))
                sample_time = 0
                update_time = 0
                forward_time = 0
                backward_time = 0

            if val and (step + 1) % val_interval == 0:
                valid_start = time.time()
                self.eval()
                self.eval_proc(rank=rank, world_size=world_size, use_tqdm=use_tqdm, logger=logger, profile=profile, mode='valid')
                self.train()
                logger.print_log('[proc {}]validation take {:.3f} seconds.'.format(rank, time.time() - valid_start))

        print('proc {} takes {:.3f} seconds'.format(rank, time.time() - train_start))
        if rank == 0 and profile:
            profiler.stop()
            print(profiler.output_text(unicode=False, color=False))


    def eval_proc(self, rank, world_size, use_tqdm, logger, profile=False, mode='test'):
        if rank == 0 and mode == 'test' and profile:
            from pyinstrument import Profiler
            profiler = Profiler()
            profiler.start()
        dataset = self.test_dataset if mode == 'test' else self.eval_dataset
        dataloader = self.eval_dataloader.generate_dataloader(dataset.partition(rank, world_size))
        iter_data = iter(dataloader)
        if len(self.gpu) > 0:
            gpu_id = self.gpu[rank % len(self.gpu)]
        else:
            gpu_id = -1
        if mode == 'test':
            self.eval()
        with th.no_grad():
            logs = []
            iter_data = tqdm(iter_data, desc='evaluation') if (rank == 0 and use_tqdm) else iter_data
            for data in iter_data:
                '''
                data here should have:
                head, rel, tail, neg: sampled data (rel_id if using TransR)
                chunk_size: chunk size of test data, equals to batch_size_eval except the last batch
                neg_sample_size: neg sample size for test, default is number of nodes in test set 
                neg_type: head/tail/head_tail specifying neg sample type
                '''
                data['chunk_size'] = data['head'].shape[0]
                encode_results = self.encoder.forward(data, gpu_id)
                results = self.decoder.forward(encode_results, data, gpu_id)
                logs += self.decoder.evaluate(results, data, self.g)
            metrics = {}
            if len(logs) > 0:
                for metric in logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
            for k, v in metrics.items():
                logger.print_log('[{}]{} average {}: {}'.format(rank, mode, k, v))
            logger.save_result(mode, metrics)
        if rank == 0 and mode == 'test' and profile:
            profiler.stop()
            print(profiler.output_text(unicode=False, color=False))

    def attach_graph(self, graph):
        self.g = graph

    def attach_dataset(self, dataset):
        self.train_dataset, self.eval_dataset, self.test_dataset = dataset

    def attach_dataloader(self, dataloader):
        self.train_dataloader, self.eval_dataloader = dataloader

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

import itertools
import functools
import torch.multiprocessing as mp
import torch as th
from itertools import chain
from tqdm import trange, tqdm
from utils import get_scalar
import time

# We use template design pattern here - which is divide the whole code chunk into partitions
# so that user can define their own method if necessary.
class KGEModel:
    def __init__(self,
                 max_step,
                 tqdm=False,
                 world_size=1,
                 gpu=[-1],
                 force_sync_interval=-1,
                 num_proc=1,
                 log_interval=1000,
                 valid=False,
                 ):
        # NFC - optim is correlated with parameters of encoder and decoder.
        # If we put optim in GNNModel, client who wants to customize their own
        # decoder need to modify self.optim to involve trainable parameters of
        # decoder. We need a way to prevent client from excluding parameters of encoder.
        self.max_step = max_step
        self.tqdm = tqdm
        self.gpu = gpu
        self.world_size = world_size
        self.force_sync_interval = force_sync_interval
        self.num_proc = num_proc
        self.log_interval = log_interval
        self.valid = valid

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def save(self, save_path):
        self.encoder.save(save_path)
        self.decoder.save(save_path)

    def load(self, load_path):
        self.encoder.load(load_path)
        for k, v in self.decoder.items():
            v.load(load_path)

    def update(self, gpu_id):
        self.encoder.update(gpu_id)
        self.decoder.update(gpu_id)

    def prepare_distribued_training(self, gpu_id=-1, rank=0, world_size=1):
        self.encoder.prepare_distribued_training(gpu_id, rank, world_size)
        self.decoder.prepare_distribued_training(gpu_id, rank, world_size)

    def share_memory(self):
        self.encoder.share_memory()
        self.decoder.share_memory()

    def dense_parameters(self):
        return list(chain(self.encoder.dense_parameters(), self.decoder.dense_parameters()))

    def sparse_parameters(self):
        return list(chain(self.encoder.sparse_parameters(), self.decoder.sparse_parameters()))

    def fit(self):
        self.train()
        start = time.time()
        if self.world_size > 1:
            self.share_memory()
        if self.num_proc > 1:
            processes = []
            barrier = mp.Barrier(self.num_proc)
            for rank in range(self.num_proc):
                p = mp.Process(target=self.train_mp,
                               args=(rank, barrier))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            self.train_proc(rank=0,)

        print(f'training takes {time.time() - start} seconds')

        if not self.no_save:
            self.save(self.save_path)

    def test(self):
        if num_proc > 1:
            for i in range(num_proc):
                Process(test_proc(i))
        else:
            test_proc(0)

    def train_proc(self, rank, barrier=None):
        # if it's multi-GPU training, the adapter will only sample data partition for gpu where gpu_id=gpu_id
        dataloader = self.train_factory.create_dataloader(rank=rank, world_size=self.world_size)
        iter_data = iter(dataloader)
        if len(self.gpu) > 0:
            gpu_id = self.gpu[rank % len(self.gpu)]
        else:
            gpu_id = -1
        if self.world_size > 1:
            self.prepare_distribued_training(gpu_id=gpu_id, rank=rank, world_size=self.world_size)
        logs = []
        train_start = time.time()
        sample_time = 0
        update_time = 0
        forward_time = 0
        backward_time = 0

        iter_range = trange(0, self.max_step, desc='train') if (rank == 0 and self.tqdm) else range(0, self.max_step)
        for step in iter_range:
            start = time.time()
            loss = {}
            data = next(iter_data)
            data['neg_type'] = 'head' if step % 2 == 0 else 'tail'
            sample_time += time.time() - start
            start = time.time()
            encode_results = self.encoder.encode(data)
            # normally it's 0
            loss['encode'] = self.encoder.get_loss(encode_results)
            decode_results = self.decoder.decode(encode_results, data) + [None]
            loss['decode'] = self.decoder.get_loss(decode_results)
            # add regularization
            loss['reg'] = self.regularizer.compute_regularization(list(itertools.chain(self.dense_parameters(), self.sparse_parameters())))
            total_loss = 0
            for k, v in loss.items():
                total_loss += v
                logs.append({k: get_scalar(v)})
            forward_time += time.time() - start
            if rank == 0 and self.tqdm:
                iter_range.set_postfix(loss=f'{total_loss.item(): .4f}')
            start = time.time()
            # problem for DDP
            self.optim.zero_grad()
            total_loss.backward()
            backward_time += time.time() - start
            start = time.time()
            self.optim.step(gpu_id)
            update_time += time.time() - start

            if self.force_sync_interval != -1 and (step + 1) % self.force_sync_interval == 0:
                barrier.wait()
            if (step + 1) % self.log_interval == 0:
                for k in logs[0].keys():
                    # use get_scalar here to reduce .item() overhead
                    v = sum(l[k] for l in logs) / len(logs)
                    print('[proc {}][Train]({}/{}) average {}: {}'.format(rank, (step + 1), self.max_step, k, v))
                logs = []
                print('[proc {}][Train] {} steps take {:.3f} seconds'.format(rank, self.log_interval,
                                                                             time.time() - start))
                print('[proc {}]sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                    rank, sample_time, forward_time, backward_time, update_time))
                sample_time = 0
                update_time = 0
                forward_time = 0
                backward_time = 0

            if self.valid and (step + 1) % self.valid_interval == 0:
                valid_start = time.time()
                self.eval()
                # TODO: lingfei - figure out how to carry rel_parts
                if self.strict_rel_part or self.soft_rel_part:
                    self.writeback_relation(rank, rel_parts)
                if self.world_size > 0:
                    barrier.wait()
                self.eval_proc(rank, mode='valid')
                self.train()
                print('[proc {}]validation take {:.3f} seconds.'.format(rank, time.time() - valid_start))
                # TODO: lingfei - figure out how to carry cross_rels
                if self.soft_rel_part:
                    self.prepare_cross_rels(cross_rels)
                if self.world_size > 0:
                    barrier.wait()

        print('proc {} takes {:.3f} seconds'.format(rank, time.time() - train_start))
        if self.async_update:
            self.finish_async_update()
        if self.strict_rel_part or args.soft_rel_part:
            self.writeback_relation(rank, rel_parts)
        self.cleanup()


    def eval_proc(self, rank, mode='test', queue=None):
        dataloader = (self.valid_factory if mode == 'valid' else self.test_factory).create_dataloader(rank=rank, world_size=self.world_size)
        iter_data = iter(dataloader)
        if len(self.gpu) > 0:
            gpu_id = self.gpu[rank % len(self.gpu)]
        else:
            gpu_id = -1
        if mode == 'test':
            self.prepare_distribued_training(gpu_id=gpu_id, rank=rank, world_size=self.world_size)
            self.eval()
        with th.no_grad():
            logs = []
            iter_data = tqdm(iter_data, desc='evaluation') if (rank == 0 and self.tqdm) else iter_data
            for data in iter_data:
                encode_results = self.encoder.encode(data)
                data = next(iter_data)
                # normally does nothing
                log = self.encoder.metrics_evaluator.evaluate(encode_results, data)
                if log is not None:
                    logs += log
                decode_results = self.decoder.decode(encode_results, data)
                logs += self.decoder.metrics_evaluator.evaluate(decode_results, data)
                metrics = {}
                if len(logs) > 0:
                    for metric in logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
                if queue is not None:
                    queue.put(logs)
                else:
                    for k, v in metrics.items():
                        print('[{}]{} average {}: {}'.format(rank, mode, k, v))
                    self.logger.save_result(mode, metrics)

        if mode == 'test':
            self.cleanup()

    def cleanup(self):
        pass

    def attach_encoder(self, encoder):
        self.encoder = encoder

    def attach_decoder(self, decoder):
        self.decoder = decoder

    def attach_regularizer(self, regularizer):
        self.regularizer = regularizer

    def attach_dataloader_factory(self, factory, genre='train'):
        if genre == 'train':
            self.train_factory = factory
        elif genre == 'valid':
            self.valid_factory = factory
        elif genre == 'test':
            self.test_factory = factory
        else:
            raise ValueError(f'{genre} factory is not supported.')

    def attach_logger(self, logger):
        self.logger = logger

    def set_optimizer(self, optimizer: functools.partial):
        self.optim = optimizer
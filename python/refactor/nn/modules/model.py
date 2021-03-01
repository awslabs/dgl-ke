import torch.multiprocessing as mp
import torch as th
from itertools import chain
from tqdm import trange, tqdm
from utils import get_scalar
from .module import Module
from utils.multiprocess import thread_wrapped_func
import time

# We use template design pattern here - which is divide the whole code chunk into partitions
# so that user can define their own method if necessary.
class Model(Module):
    def __init__(self,
                 gpu=(-1),
                 model_name='BaseModel',):
        super(Model, self).__init__(model_name)
        self.gpu = gpu
        self.optim = None

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
        self.decoder.load(load_path)

    def update(self, gpu_id):
        if self.optim is not None:
            self.optim.backward()
        self.encoder.update(gpu_id)
        self.decoder.update(gpu_id)

    def set_training_params(self, args):
        self.encoder.set_training_params(args)
        self.decoder.set_training_params(args)

    def set_test_params(self, args):
        self.encoder.set_test_params(args)
        self.decoder.set_test_params(args)

    def prepare_model(self, gpu_id, rank, world_size):
        # TODO: lingfei - remain to be done
        self.encoder.prepare_model(gpu_id, rank, world_size)
        self.decoder.prepare_model(gpu_id, rank, world_size)

    def sync_model(self, gpu_id, rank, world_size):
        self.encoder.sync_model(gpu_id, rank, world_size)
        self.decoder.sync_model(gpu_id, rank, world_size)

    def postprocess_model(self, gpu_id=-1, rank=0, world_size=1):
        self.encoder.postprocess_model(gpu_id, rank, world_size)
        self.decoder.postprocess_model(gpu_id, rank, world_size)

    def share_memory(self):
        self.encoder.share_memory()
        self.decoder.share_memory()

    def test(self,
             world_size=1,
             use_tqdm=False,
             num_test_proc=1,
             profile=False):
        self.eval()
        start = time.time()
        if world_size > 1:
            self.share_memory()

        if num_test_proc > 1:
            queue = mp.Queue(num_test_proc)
            procs = []
            for i in range(num_test_proc):
                proc = mp.Process(target=self.eval_mp, kwargs={'rank': i,
                                                               'world_size': world_size,
                                                               'use_tqdm': use_tqdm,
                                                               'profile': profile,
                                                               'mode': 'test',
                                                               'queue': queue})
                procs.append(proc)
                proc.start()
            metrics = {}
            logs = []
            for i in range(num_test_proc):
                log = queue.get()
                logs = logs + log

            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
            print("-------------- Test result --------------")
            for k, v in metrics.items():
                print('Test average {} : {}'.format(k, v))
            print("-----------------------------------------")

            for proc in procs:
                proc.join()
        else:
            self.eval_proc(rank=0, world_size=world_size, use_tqdm=use_tqdm, profile=profile, mode='test')
        print(f'test takes {time.time() - start} seconds')

    def dense_parameters(self):
        return chain(self.encoder.dense_parameters(), self.decoder.dense_parameters())

    def sparse_parameters(self):
        return chain(self.encoder.sparse_parameters(), self.decoder.sparse_parameters())

    def parameters(self):
        for param in self.dense_parameters():
            yield param
        for param in self.sparse_parameters():
            yield param

    def fit(self,
            max_step,
            world_size=1,
            num_proc=1,
            use_tqdm=False,
            force_sync_interval=-1,
            log_interval=1000,
            val=False,
            val_interval=1000,
            optim='Adagrad',
            profile=False):

        self.train()
        start = time.time()
        if world_size > 1:
            self.share_memory()
        if num_proc > 1:
            processes = []
            barrier = mp.Barrier(num_proc)
            for rank in range(num_proc):
                p = mp.Process(target=self.train_mp,
                               args=(rank, barrier, num_proc))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            self.train_proc(rank=0,
                            max_step=max_step,
                            use_tqdm=use_tqdm,
                            world_size=world_size,
                            force_sync_interval=force_sync_interval,
                            log_interval=log_interval,
                            val=val,
                            val_interval=val_interval,
                            profile=profile,
                            optim=optim,)

        print(f'training takes {time.time() - start} seconds')



    def prepare_data(self, data):
        pass

    def train_proc(self,
                   rank,
                   max_step,
                   use_tqdm=False,
                   world_size=1,
                   force_sync_interval=-1,
                   log_interval=1000,
                   val=False,
                   val_interval=1000,
                   profile=False,
                   optim='Adagrad',
                   barrier=None):
        if profile and rank == 0:
            from pyinstrument import Profiler
            profiler = Profiler()
            profiler.start()
        # if it's multi-GPU training, the adapter will only sample data partition for gpu where gpu_id=gpu_id
        dataset = self.train_dataset.partition(rank, world_size)
        dataloader = self.train_dataloader.create_dataloader(dataset)
        iter_data = iter(dataloader)
        if len(self.gpu) > 0:
            gpu_id = self.gpu[rank % len(self.gpu)]
        else:
            gpu_id = -1
        self.prepare_model(gpu_id=gpu_id, rank=rank, world_size=world_size)
        logs = []
        train_start = time.time()
        sample_time = 0
        update_time = 0
        forward_time = 0
        backward_time = 0

        iter_range = trange(0, max_step, desc='train') if (rank == 0 and use_tqdm) else range(0, max_step)
        for step in iter_range:
            start = time.time()
            loss = {}
            data = next(iter_data)
            self.prepare_data(data)
            sample_time += time.time() - start
            start = time.time()
            encode_results = self.encoder.forward(data, gpu_id)
            # normally it's 0
            loss['encode'] = self.encoder.get_loss(encode_results)
            decode_results = self.decoder.forward(encode_results, data, gpu_id)
            # None for edge importance
            loss['decode'] = self.decoder.get_loss(decode_results)
            # add regularization
            loss['reg'] = self.regularizer.compute_regularization(self.parameters())
            total_loss = 0
            for k, v in loss.items():
                total_loss += v
            logs.append({k: get_scalar(v) for k, v in loss.items()})
            forward_time += time.time() - start
            if rank == 0 and use_tqdm:
                iter_range.set_postfix(loss=f'{total_loss.item(): .4f}')
            start = time.time()
            if self.optim is not None:
                self.optim.zero_grad()
            total_loss.backward()
            backward_time += time.time() - start
            start = time.time()
            self.update(gpu_id)
            update_time += time.time() - start

            if force_sync_interval != -1 and (step + 1) % force_sync_interval == 0:
                barrier.wait()
            if (step + 1) % log_interval == 0:
                for k in logs[0].keys():
                    # use get_scalar here to reduce .item() overhead
                    v = sum(l[k] for l in logs) / len(logs)
                    print('[proc {}][Train]({}/{}) average {}: {}'.format(rank, (step + 1), max_step, k, v))
                logs = []
                print('[proc {}][Train] {} steps take {:.3f} seconds'.format(rank, log_interval,
                                                                             time.time() - start))
                print('[proc {}]sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                    rank, sample_time, forward_time, backward_time, update_time))
                sample_time = 0
                update_time = 0
                forward_time = 0
                backward_time = 0

            if val and (step + 1) % val_interval == 0:
                valid_start = time.time()
                self.eval()
                self.sync_model(gpu_id, rank, world_size)
                if world_size > 1:
                    barrier.wait()
                self.eval_proc(rank=rank, world_size=world_size, use_tqdm=use_tqdm, profile=profile, mode='valid')
                self.train()
                print('[proc {}]validation take {:.3f} seconds.'.format(rank, time.time() - valid_start))
                # TODO: lingfei - figure out how to carry cross_rels
                self.prepare_model(gpu_id, rank, self.world_size)
                if world_size > 0:
                    barrier.wait()

        self.postprocess_model(gpu_id, rank, world_size)
        print('proc {} takes {:.3f} seconds'.format(rank, time.time() - train_start))
        if rank == 0 and profile:
            profiler.stop()
            print(profiler.output_text(unicode=False, color=False))


    def eval_proc(self, rank, world_size, use_tqdm, profile=False, mode='test', queue=None):
        if rank == 0 and mode == 'test' and profile:
            from pyinstrument import Profiler
            profiler = Profiler()
            profiler.start()
        dataset = self.test_dataset if mode == 'test' else self.eval_dataset
        dataloader = self.eval_dataloader.create_dataloader(dataset.partition(rank, world_size))
        iter_data = iter(dataloader)
        if len(self.gpu) > 0:
            gpu_id = self.gpu[rank % len(self.gpu)]
        else:
            gpu_id = -1
        if mode == 'test':
            self.prepare_model(gpu_id=gpu_id, rank=rank, world_size=world_size)
            self.eval()
        with th.no_grad():
            logs = []
            iter_data = tqdm(iter_data, desc='evaluation') if (rank == 0 and use_tqdm) else iter_data
            for data in iter_data:
                self.prepare_data(data)
                encode_results = self.encoder.forward(data, gpu_id)
                log = self.encoder.evaluate(encode_results, data, self.g)
                if log is not None:
                    logs += log
                results = self.decoder.forward(encode_results, data, gpu_id)
                logs += self.decoder.evaluate(results, data, self.g)
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
            self.postprocess_model(gpu_id, rank, world_size)
        if rank == 0 and mode == 'test' and profile:
            profiler.stop()
            print(profiler.output_text(unicode=False, color=False))

    @thread_wrapped_func
    def train_mp(self, rank, barrier=None, num_proc=1):
        if num_proc > 1:
            th.set_num_threads(num_proc)
        self.train_proc(rank, barrier)

    @thread_wrapped_func
    def eval_mp(self, rank, mode='valid', queue=None, num_proc=1):
        if num_proc > 1:
            th.set_num_threads(num_proc)
        self.eval_proc(rank, mode, queue)


    def cleanup(self):
        pass

    def attach_encoder(self, encoder):
        self.encoder = encoder

    def attach_decoder(self, decoder):
        self.decoder = decoder

    def attach_regularizer(self, regularizer):
        self.regularizer = regularizer

    def attach_dataset(self, dataset):
        self.train_dataset, self.eval_dataset, self.test_dataset = dataset

    def attach_graph(self, g):
        self.g = g

    def attach_dataloader(self, dataloader):
        self.train_dataloader, self.eval_dataloader = dataloader

    def attach_logger(self, logger):
        self.logger = logger

    @property
    def name(self):
        return super(Model, self).name()


class KGEModel(Model):
    def __init__(self,
                 gpu=(-1)):
        super(KGEModel, self).__init__(gpu, 'KGE_MODEL')

    def prepare_data(self, data):
        for k, v in data.items():
            if type(data[k]) == th.Tensor:
                data[k] = v.view(-1)
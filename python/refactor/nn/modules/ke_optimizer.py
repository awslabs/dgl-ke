import torch as th
import math
from collections import defaultdict
import copy


class Optimizer(object):
    def __init__(self, defaults, device=th.device('cpu')):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.device = device
        self.state['step'] = 0

    def step(self, idx, param, grad, gpu_id=-1):
        raise NotImplementedError

    def share_memory(self):
        for s in self.state.values():
            if type(s) is th.Tensor:
                s.share_memory_()

    def to(self, device: th.device):
        if device == self.device:
            return self
        else:
            optim = copy.deepcopy(self)
            for k, s in optim.state.items():
                if type(s) is th.Tensor:
                    optim.state[k] = s.to(device)
            optim.device = device
            return optim


class Adagrad(Optimizer):
    def __init__(self, emb, device=th.device('cpu'), lr=1e-3, epsilon=1e-10, unique_indices=True, mean_sum=False):
        defaults = dict(lr=lr, epsilon=epsilon)
        super(Adagrad, self).__init__(defaults, device)
        self.unique_indices = unique_indices
        self.mean_sum = mean_sum
        if mean_sum:
            self.state['sum'] = emb.new().resize_(emb.shape[0], 1).zero_()
        else:
            self.state['sum'] = th.zeros_like(emb, device=device)

    @th.no_grad()
    def step(self, idx, param, grad, gpu_id=-1):
        self.state['step'] += 1
        # hyper-parameters
        clr = self.defaults['lr']
        epsilon = self.defaults['epsilon']

        # device to save accumulated result and to calculate gradient
        device_cal = th.device('cpu' if gpu_id == -1 else f'cuda:{gpu_id}')
        device_sv = self.device

        grad_values = grad
        grad_indices = idx.to(device_sv)

        # average gradient if duplicate found
        if self.unique_indices:
            grad_indices, inv_indicies, cnt = th.unique(grad_indices, return_inverse=True, return_counts=True)
            grad_avg = th.zeros(grad_indices.shape[0], grad.shape[1], device=device_sv)
            grad_avg.index_add_(0, inv_indicies, grad.to(device_sv))
            grad_avg = grad_avg / cnt.unsqueeze(1)
            grad_values = grad_avg

        if self.mean_sum:
            grad_sq = th.mean(grad_values ** 2, 1, keepdim=True)
        else:
            grad_sq = grad_values ** 2

        # save it back to state before doing inplace operation
        self.state['sum'].index_add_(0, grad_indices, grad_sq.to(device_sv))
        std = self.state['sum'][grad_indices].to(device_cal)

        std_values = std.sqrt_().add_(epsilon)

        update_val = (-clr * grad_values.to(device_cal) / std_values)

        param.index_add_(0, grad_indices, update_val.to(device_sv))


class Adam(Optimizer):
    def __init__(self, emb, device=th.device('cpu'), lr=1e-3, eps=1e-8, betas=(0.9, 0.999)):
        defaults = dict(lr=lr, eps=eps, betas=betas)
        super(Adam, self).__init__(defaults)
        self.state['exp_avg'] = th.zeros_like(emb, device=device)
        self.state['exp_avg_sq'] = th.zeros_like(emb, device=device)

    @th.no_grad()
    def step(self, idx, param, grad, gpu_id=-1):
        beta1, beta2 = self.defaults['betas']
        lr, eps = self.defaults['lr'], self.defaults['eps']
        self.state['step'] += 1
        device = self.state['exp_avg'].device
        if idx.device != device:
            idx = idx.to(device)
        if grad.device != device:
            grad = grad.to(device)
        grad_indices, inv_indicies, cnt = th.unique(idx, return_inverse=True, return_counts=True)
        grad_values = th.zeros(grad_indices.shape[0], grad.shape[1], device=device)
        grad_values.index_add_(0, inv_indicies, grad)
        grad_values = grad_values / cnt.unsqueeze(1)
        old_exp_avg_values = self.state['exp_avg'][grad_indices]
        exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
        self.state['exp_avg'].index_add_(0, grad_indices, exp_avg_update_values)
        old_exp_avg_sq_values = self.state['exp_avg_sq'][grad_indices]
        exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
        self.state['exp_avg_sq'].index_add_(0, grad_indices, exp_avg_sq_update_values)
        bias_correction1 = 1 - beta1 ** self.state['step']
        bias_correction2 = 1 - beta2 ** self.state['step']
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        numer = exp_avg_update_values.add_(old_exp_avg_values)
        demon = exp_avg_sq_update_values.add_(old_exp_avg_sq_values).sqrt_().add_(eps)
        del exp_avg_update_values, exp_avg_sq_update_values

        if gpu_id >= 0:
            numer = numer.cuda(gpu_id)
            demon = demon.cuda(gpu_id)
        update_val = - step_size * numer.div_(demon)
        if update_val.device != device:
            update_val = update_val.to(device)
        param.index_add_(0, grad_indices, update_val)

import torch as th
import math
from collections import defaultdict
import copy


class Optimizer(object):
    def __init__(self, defaults):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.state['step'] = 0

    def step(self, idx, param, grad, gpu_id=-1):
        raise NotImplementedError

    def share_memory(self):
        for s in self.state.values():
            if type(s) is th.Tensor:
                s.share_memory_()

    def to(self, device: th.device):
        for k, s in self.state.items():
            if type(s) is th.Tensor:
                self.state[k] = s.to(device)
        return self



class Adagrad(Optimizer):
    def __init__(self, emb, device=th.device('cpu'), lr=1e-3, epsilon=1e-10, unique_indices=False, mean_sum=True):
        defaults = dict(lr=lr, epsilon=epsilon)
        super(Adagrad, self).__init__(defaults)
        self.unique_indices = unique_indices
        self.mean_sum = mean_sum
        if mean_sum:
            self.state['sum'] = emb.new().resize_(emb.shape[0], 1).zero_()
        else:
            self.state['sum'] = th.zeros_like(emb, device=device)

    @th.no_grad()
    def step(self, idx, param, grad, gpu_id=-1):
        clr = self.defaults['lr']
        epsilon = self.defaults['epsilon']
        device = self.state['sum'].device
        self.state['step'] += 1
        grad_values = grad
        grad_indices = idx
        if grad_indices.device != device:
            grad_indices = grad_indices.to(device)
        if self.unique_indices:
            grad_indices, inv_indicies, cnt = th.unique(grad_indices, return_inverse=True, return_counts=True)
            grad_values = th.zeros(grad_indices.shape[0], grad.shape[1], device=device)
            grad_values.index_add_(0, inv_indicies, grad.to(device))
            grad_values = grad_values / cnt.unsqueeze(1)
        if self.mean_sum:
            grad_sq = th.mean(grad_values ** 2, 1, keepdim=True)
        else:
            grad_sq = grad_values ** 2
        if grad_sq.device != device:
            grad_sq = grad_sq.to(device)
        self.state['sum'].index_add_(0, grad_indices, grad_sq)
        std = self.state['sum'][grad_indices]
        if gpu_id >= 0:
            std = std.cuda(gpu_id)
        std_values = std.sqrt_().add_(epsilon)
        update_val = (-clr * grad_values / std_values)
        if update_val.device != device:
            update_val = update_val.to(device)
        param.index_add_(0, grad_indices, update_val)


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

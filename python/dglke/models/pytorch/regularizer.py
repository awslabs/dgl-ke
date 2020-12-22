import torch as th

class Regularizer(object):
    def __init__(self, coef, norm):
        self.coef = coef
        self.norm = norm

    def __call__(self, params: list):
        log = {}
        reg = 0
        if self.coef != 0:
            for param in params:
                reg += param.norm(p=self.norm) ** self.norm
            reg *= self.coef
            log['regularization'] = reg.detach().item()
            return reg, log
        else:
            log['regularization'] = 0
            return reg, log
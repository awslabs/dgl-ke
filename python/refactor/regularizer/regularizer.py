from utils import norm
class Regularizer:
    def __init__(self, coef=1e-07, norm=3):
        self.coef = coef
        self.norm = norm

    def compute_regularization(self, params_list):
        reg = 0
        for params in params_list:
            reg += self.coef * norm(params, p=self.norm)
        return reg
from dglke.utils import norm

class Regularizer:
    def __init__(self, coef=1e-07, norm=3):
        self.coef = coef
        self.norm = norm

    def compute_regularization(self, params):
        """ compute sparse regularization

        Parameters
        ----------
        params: torch.nn.module.parameter()
            iterable parameters that should be regularized

        Returns
        -------
        torch.Tensor
            regularization
        """
        reg = 0
        if type(params) == dict:
            for k, v in params.items():
                if k == 'rel_id':
                    continue
                reg += self.coef * norm(v, p=self.norm)
        elif type(params) == list:
            reg += self.coef * norm(params, p=self.norm)
        else:
            raise NotImplementedError(f'regularization for {type(params)} is not supported.')
        return reg
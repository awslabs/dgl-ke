import torch as th
import torch.nn.functional as functional

logsigmoid = functional.logsigmoid
softplus = functional.softplus
sigmoid = th.sigmoid


class BaseLoss(object):
    def __call__(self, score, label):
        raise NotImplementedError


class HingeLoss(BaseLoss):
    r""" Hinge Loss
    \max(0, \lambda - l_i \cdot f(t_i))
    \lambda : margin value (hyper-parameter)
    l_i : label i
    f : score function
    t_i : triple i
    """
    def __init__(self, margin):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def __call__(self, score: th.Tensor, label):
        loss = self.margin - label * score
        loss[loss < 0] = 0
        return loss


class LogisticLoss(BaseLoss):
    r""" Logistic Loss
    \log(1 + \exp(-l_i \cdot f(t_i)))
    l_i : label i from {-1, 1}
    f : score function
    t_i : triple i
    """
    def __init__(self):
        super(LogisticLoss, self).__init__()

    def __call__(self, score: th.Tensor, label):
        return softplus(-label * score)


class BCELoss(BaseLoss):
    r""" Binary Cross Entropy Loss
    -(l_i \cdot log(\sigma(f(t_i))) + (1 - l_i) \cdot \log(1 - \sigma(f(t_i))))
    l_i : label i from {0, 1}
    f : score function
    \sigma : logistic sigmoid function
    t_i : triple i
    """
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss = th.nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, score: th.Tensor, label):
        if type(label) is int or type(label) is float:
            label = th.full_like(score, label)
        return self.loss(score, label)


class LogsigmoidLoss(BaseLoss):
    r""" Logsigmoid Loss
    -\log(\frac{1}{1 + \exp(-l_i \cdot f(t_i))})
    l_i : label i from {-1, 1}
    f : score
    t_i : triple i
    """
    def __init__(self):
        super(LogsigmoidLoss, self).__init__()

    def __call__(self, score: th.Tensor, label):
        return - logsigmoid(label * score)

from ..base_loss import *
from .tensor_models import logsigmoid

class HingeLoss(BaseHingeLoss):
    def __init__(self, margin):
        assert False, 'HingeLoss is not implemented'

    def __call__(self, score, label):
        pass

class LogisticLoss(BaseLogisticLoss):
    def __init__(self):
        assert False, 'LogisticLoss is not implemented'

    def __call__(self, score, label):
        pass

class BCELoss(BaseBCELoss):
    def __init__(self):
        assert False, 'BCELoss is not implemented'

    def __call__(self, score, label):
        pass

class LogsigmoidLoss(BaseLogsigmoidLoss):
    def __init__(self):
        super(LogsigmoidLoss, self).__init__()

    def __call__(self, score, label):
        return - logsigmoid(label * score)

class LossGenerator(BaseLossGenerator):
    def __init__(self, args, loss_genre='Logsigmoid', neg_adversarial_sampling=False, adversarial_temperature=1.0,
                 pairwise=False, label_smooth=.0):
        assert False, 'LossGenerator is not implemented'


    def _get_pos_loss(self, pos_score, pos_label):
        pass

    def _get_neg_loss(self, neg_score, neg_label):
        pass

    def get_total_loss(self, pos_score, neg_score, edge_weight):
        pass

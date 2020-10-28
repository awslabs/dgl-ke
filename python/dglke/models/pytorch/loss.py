from ..base_loss import *
from .tensor_models import *
import torch as th
import torch.nn.functional as functional

logsigmoid = functional.logsigmoid
softplus  = functional.softplus

class HingeLoss(BaseHingeLoss):
    def __init__(self, args):
        super(HingeLoss, self).__init__(args)

    def __call__(self, score: th.Tensor, label):
        loss = self.margin - label * score
        loss[loss < 0] = 0
        return loss

class LogisticLoss(BaseLogisticLoss):
    def __init__(self, args):
        super(LogisticLoss, self).__init__(args)

    def __call__(self, score: th.Tensor, label):
        return softplus(-label * score)

class BCELoss(BaseBCELoss):
    def __init__(self, args):
        super(BCELoss, self).__init__(args)

    def __call__(self, score: th.Tensor, label):
        return -(label * th.log(th.sigmoid(score))) + \
               (1 - label) * th.log(1 - th.sigmoid(score))

class LogsigmoidLoss(BaseLogsigmoidLoss):
    def __init__(self, args):
        super(LogsigmoidLoss, self).__init__(args)

    def __call__(self, score: th.Tensor, label):
        return - logsigmoid(label * score)


class LossGenerator(BaseLossGenerator):
    def __init__(self, args):
        super(LossGenerator, self).__init__(args)
        if self.loss_genre == 'Hinge':
            self.neg_label = -1
            self.loss_criterion = HingeLoss(args)
        elif self.loss_genre == 'Logistic' or self.loss_genre == 'Softplus':
            self.neg_label = -1
            self.loss_criterion = LogisticLoss(args)
        elif self.loss_genre == 'Logsigmoid':
            self.neg_label = -1
            self.loss_criterion = LogsigmoidLoss(args)
        elif self.loss_genre == 'BCE':
            self.neg_label = 0
            self.loss_criterion = BCELoss(args)
        else:
            raise ValueError('loss genre %s is not support' % self.loss_genre)

    def _get_pos_loss(self, pos_score):
        return self.loss_criterion(pos_score, 1)

    def _get_neg_loss(self, neg_score):
        return self.loss_criterion(neg_score, self.neg_label)

    def get_total_loss(self, pos_score, neg_score):
        log = {}
        if self.pairwise:
            pos_score = pos_score.unsqueeze(-1)
            loss = th.mean(self.loss_criterion(pos_score - neg_score, 1))
            log['loss'] = get_scalar(loss)
            return loss, log
        pos_loss = self._get_pos_loss(pos_score)
        neg_loss = self._get_neg_loss(neg_score)
        # MARK - would average twice make loss function lose precision?
        # do mean over neg_sample
        if self.neg_adversarial_sampling:
            neg_loss = th.sum(th.softmax(neg_score * self.adversarial_temperature, dim=-1).detach() * neg_loss, dim=-1)
        else:
            neg_loss = th.mean(neg_loss, dim=-1)
        # do mean over chunk
        neg_loss = th.mean(neg_loss)
        pos_loss = th.mean(pos_loss)
        loss = (neg_loss + pos_loss) / 2
        log['pos_loss'] = get_scalar(pos_loss)
        log['neg_loss'] = get_scalar(neg_loss)
        log['loss'] = get_scalar(loss)
        return loss, log








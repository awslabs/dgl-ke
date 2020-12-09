from ..base_loss import *
from .tensor_models import *
import torch as th
import torch.nn.functional as functional

logsigmoid = functional.logsigmoid
softplus = functional.softplus
sigmoid = th.sigmoid

class HingeLoss(BaseHingeLoss):
    def __init__(self, margin):
        super(HingeLoss, self).__init__(margin)

    def __call__(self, score: th.Tensor, label):
        loss = self.margin - label * score
        loss[loss < 0] = 0
        return loss

class LogisticLoss(BaseLogisticLoss):
    def __init__(self):
        super(LogisticLoss, self).__init__()

    def __call__(self, score: th.Tensor, label):
        return softplus(-label * score)

class BCELoss(BaseBCELoss):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss = th.nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, score: th.Tensor, label):
        if type(label) is int or type(label) is float:
            label = th.full_like(score, label)
        return self.loss(score, label)

class LogsigmoidLoss(BaseLogsigmoidLoss):
    def __init__(self):
        super(LogsigmoidLoss, self).__init__()

    def __call__(self, score: th.Tensor, label):
        return - logsigmoid(label * score)


class LossGenerator(BaseLossGenerator):
    def __init__(self, args, loss_genre='Logsigmoid', neg_adversarial_sampling=False, adversarial_temperature=1.0,
                 pairwise=False, label_smooth=.0):
        super(LossGenerator, self).__init__(neg_adversarial_sampling, adversarial_temperature, pairwise, label_smooth)
        self.pos_label = 1
        if loss_genre == 'Hinge':
            self.neg_label = -1
            self.loss_criterion = HingeLoss(args.margin)
        elif loss_genre == 'Logistic':
            self.neg_label = -1
            self.loss_criterion = LogisticLoss()
        elif loss_genre == 'Logsigmoid':
            self.neg_label = -1
            self.loss_criterion = LogsigmoidLoss()
        elif loss_genre == 'BCE':
            self.neg_label = 0
            self.loss_criterion = BCELoss()
        else:
            raise ValueError('loss genre %s is not support' % loss_genre)

        if self.pairwise and loss_genre not in ['Logistic', 'Hinge']:
            raise ValueError('{} loss cannot be applied to pairwise loss function'.format(loss_genre))

    def _get_pos_loss(self, pos_score, pos_label):
        return self.loss_criterion(pos_score, pos_label)

    def _get_neg_loss(self, neg_score, neg_label):
        return self.loss_criterion(neg_score, neg_label)

    def get_total_loss(self, pos_score, neg_score, edge_weight=None):
        log = {}
        if edge_weight is None:
            edge_weight = 1
        if self.pairwise:
            pos_score = pos_score.unsqueeze(-1)
            loss = th.mean(self.loss_criterion((pos_score - neg_score), 1) * edge_weight)
            log['loss'] = get_scalar(loss)
            return loss, log
        pos_label = (1 - self.label_smooth) * self.pos_label + (self.label_smooth / (neg_score.shape[-1] + 1))
        neg_label = (1 - self.label_smooth) * self.neg_label + (self.label_smooth / (neg_score.shape[-1] + 1))
        pos_loss = self._get_pos_loss(pos_score, pos_label) * edge_weight
        neg_loss = self._get_neg_loss(neg_score, neg_label) * edge_weight
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

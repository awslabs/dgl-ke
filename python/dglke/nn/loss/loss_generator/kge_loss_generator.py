from .loss_generator import LossGenerator
from dglke.nn.loss.loss_criterion import BCELoss
import torch as th


class KGELossGenerator(LossGenerator):
    def __init__(self,
                 label_smooth=.0,):
        super(KGELossGenerator, self).__init__()
        self.label_smooth = label_smooth


class LCWAKGELossGenerator(KGELossGenerator):
    """
    Local Closed World Assumption Knowledge Graph Embedding Loss Generator
    """
    def __init__(self,
                 label_smooth=.0):
        super(LCWAKGELossGenerator, self).__init__(label_smooth)

# TODO: lingfei - change to loss
class sLCWAKGELossGenerator(KGELossGenerator):
    """
    stochastic Local Closed World Assumption Knowledge Graph Embedding Loss Generator
    """
    def __init__(self,
                 neg_adversarial_sampling=False,
                 adversarial_temperature=1.0,
                 pairwise=False,
                 label_smooth=.0,
                 ):
        super(sLCWAKGELossGenerator, self).__init__(label_smooth)
        self.neg_adversarial_sampling = neg_adversarial_sampling
        self.adversarial_temperature = adversarial_temperature
        self.pairwise = pairwise
        self.pos_label = 1
        self.neg_label = -1

    def __get_pos_loss(self, pos_score, pos_label):
        return self.loss_criterion(pos_score, pos_label)

    def __get_neg_loss(self, neg_score, neg_label):
        return self.loss_criterion(neg_score, neg_label)

    def set_criterion(self, loss_criterion):
        # additional code fragment to set neg_label to 0
        if type(loss_criterion) == BCELoss:
            self.neg_label = 0
        super(sLCWAKGELossGenerator, self).set_criterion(loss_criterion)

    def get_total_loss(self, results):
        pos_score, neg_score = results['pos_score'], results['neg_score']
        edge_weight = results['edge_weight'] if 'edge_weight' in results.keys() else None
        if edge_weight is None:
            edge_weight = 1
        if self.pairwise:
            pos_score = pos_score.unsqueeze(-1)
            loss = th.mean(self.loss_criterion((pos_score - neg_score), 1) * edge_weight)
            return loss
        pos_label = (1 - self.label_smooth) * self.pos_label + (self.label_smooth / (neg_score.shape[-1] + 1))
        neg_label = (1 - self.label_smooth) * self.neg_label + (self.label_smooth / (neg_score.shape[-1] + 1))
        pos_loss = self.__get_pos_loss(pos_score, pos_label) * edge_weight
        neg_loss = self.__get_neg_loss(neg_score, neg_label) * edge_weight
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
        return loss


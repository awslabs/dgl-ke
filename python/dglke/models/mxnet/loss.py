from ..base_loss import *

class HingeLoss(BaseHingeLoss):
    def __init__(self, args):
        assert False, 'HingeLoss is not implemented'

    def __call__(self, score, label):
        pass

class LogisticLoss(BaseLogisticLoss):
    def __init__(self, args):
        assert False, 'LogisticLoss is not implemented'

    def __call__(self, score, label):
        pass

class LogsigmoidLoss(BaseLogsigmoidLoss):
    def __init__(self, args):
        assert False, 'Logsigmoid is not implemented'


    def __call__(self, score, label):
        pass

class RankingLoss(BaseRankingLoss):
    def __init__(self, args):
        assert False, 'Logsigmoid is not implemented'

    def __call__(self, score, label):
        pass

class LossGenerator(BaseLossGenerator):
    def __init__(self, args):
        assert False, 'LossGenerator is not implemented'


    def _get_pos_loss(self, pos_score):
        """ Predict loss for positive labels

        Parameters
        ----------
        pos_score : tensor
                    Score calculated from positive triples

        Returns
        -------
        tensor
                positive loss calculated with specific loss criterion
        """
        pass

    def _get_neg_loss(self, neg_score):
        """ Predict loss for negative triples

        Parameters
        ----------
        neg_score: tensor
                   Score calculated from positive triples

        Returns
        -------
        tensor
                Negative loss calculated with specific loss criterion
        """
        pass

    def get_total_loss(self, pos_score, neg_score):
        """ Calculate total loss for a batch of positive triples and negative triples. The total loss can be
            point-wise and pairwise. For pairwise, it is average of the relative loss from positive score to negative
            score. For point-wise, it can be average of the positive loss and negative loss or negative loss
            weighted by its negative score and adversarial_temperature.

        Parameters
        ----------
        pos_score : tensor
                    Score calculated from positive triples
        neg_score : tensor
                    Score calculated from negative triples

        Returns
        -------
        tensor
            Total loss by aggregate positive score and negative score.
        log
            log to record scalar value of negative loss, positive loss and/or total loss
        """
        pass

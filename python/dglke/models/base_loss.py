class BaseLoss(object):
    def __call__(self, score, label):
        pass

class BaseLogisticLoss(BaseLoss):
    def __init__(self, args):
        super(BaseLogisticLoss, self).__init__()
        self.neg_label = args.neg_label

    def __call__(self, score, label):
        pass

class BaseBCELoss(BaseLoss):
    def __init__(self, args):
        super(BaseBCELoss, self).__init__()

    def __call__(self, score, label):
        pass

class BaseHingeLoss(BaseLoss):
    def __init__(self, args):
        super(BaseHingeLoss, self).__init__()
        self.margin = args.margin

    def __call__(self, score, label):
        pass

class BaseRankingLoss(BaseLoss):
    def __init__(self, args):
        super(BaseRankingLoss, self).__init__()
        self.margin = args.margin

    def __call__(self, score, label):
        pass

class BaseLogsigmoidLoss(BaseLoss):
    def __init__(self, args):
        super(BaseLogsigmoidLoss, self).__init__()

    def __call__(self, score, label):
        pass

class BaseLossGenerator(object):
    def __init__(self, args):
        self.pairwise = args.pairwise
        self.neg_adversarial_sampling = args.neg_adversarial_sampling
        if self.neg_adversarial_sampling:
            self.adversarial_temperature = args.adversarial_temperature
        else:
            self.adversarial_temperature = 0
        self.loss_genre = args.loss_genre
        self.neg_label = args.neg_label
        if self.pairwise == self.neg_adversarial_sampling == True:
            raise ValueError('loss cannot be pairwise and adversarial sampled')
        if self.pairwise and self.loss_genre == 'Ranking':
            raise ValueError('Ranking loss cannot be applied to pairwise loss function')

    def get_pos_loss(self, pos_score):
        pass

    def get_neg_loss(self, neg_score):
        pass

    def get_total_loss(self, pos_loss, neg_loss):
        pass
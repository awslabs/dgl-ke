# We split LossGenerator from criterion(Logsigmoid, BCE, etc) so that user can
# dynamically choose which criterion to use for different supervision task.
# However, in this way, the hyper-paramters of LossGenerator will not change dynamically
# according to different supervision tasks.
class LossGenerator:
    def __init__(self):
        pass

    def set_criterion(self, loss_criterion):
        # loss_criterion should be callable, it can be user-defined or from
        # pytorch library
        self.loss_criterion = loss_criterion

    def get_total_loss(self, results):
        # this is left for subclass to implement
        return 0

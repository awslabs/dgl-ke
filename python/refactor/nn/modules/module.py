from refactor.nn.loss.loss_generator import LossGenerator
from refactor.nn.metrics import MetricsEvaluator
from collections import OrderedDict
from torch import nn

# encoder, decoder are both inherited from Module. The default behavior of get_loss(), evaluate()
# does nothing as sometimes encoder does not need them. While for decoder, it's mandatory for it
# to overload these two functions.


class Module(nn.Module):
    def __init__(self, module_name):
        super(Module, self).__init__()
        self.module_name = module_name
        self.metrics_evaluator = MetricsEvaluator()
        self.loss_generator = LossGenerator()
        self._sparse_parameters = OrderedDict()

    def sparse_parameters(self):
        for name, ele in self._sparse_parameters.items():
            yield ele

    def attach_loss_generator(self, loss_generator: LossGenerator):
        # default does nothing
        self.loss_generator = loss_generator

    def attach_metrics_evaluator(self, metrics_evaluator: MetricsEvaluator):
        # default does nothing
        self.metrics_evaluator = metrics_evaluator

    def get_loss(self, results):
        return self.loss_generator.get_total_loss(results)

    def set_training_params(self, args):
        raise NotImplementedError

    def set_test_params(self, args):
        raise NotImplementedError

    def evaluate(self, results: list, data, graph):
        raise NotImplementedError

    @property
    def name(self):
        return self.module_name

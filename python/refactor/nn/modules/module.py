from refactor.nn.loss.loss_generator import LossGenerator
from refactor.nn.metrics import MetricsEvaluator
from collections import OrderedDict

# encoder, decoder are both inherited from Module. The default behavior of get_loss(), evaluate()
# does nothing as sometimes encoder does not need them. While for decoder, it's mandatory for it
# to overload these two functions.


class Module(object):
    def __init__(self, module_name):
        self.module_name = module_name
        # default metrics_evaluator and loss_generator do nothing
        self.metrics_evaluator = MetricsEvaluator()
        self.loss_generator = LossGenerator()
        self._sparse_parameters = OrderedDict()
        self._dense_parameters = OrderedDict()

    def sparse_parameters(self):
        for name, elem in self._sparse_parameters.items():
            yield elem

    def dense_parameters(self):
        for name, elem in self._dense_parameters.items():
            yield elem

    def attach_loss_generator(self, loss_generator: LossGenerator):
        # default does nothing
        self.loss_generator = loss_generator

    def attach_metrics_evaluator(self, metrics_evaluator: MetricsEvaluator):
        # default does nothing
        self.metrics_evaluator = metrics_evaluator

    def get_loss(self, results):
        return self.loss_generator.get_total_loss(results)

    def save(self, save_path: str):
        raise NotImplementedError()

    def load(self, load_path: str):
        raise NotImplementedError()

    def share_memory(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def update(self, gpu_id):
        raise NotImplementedError()

    def set_training_params(self, args):
        raise NotImplementedError

    def set_test_params(self, args):
        raise NotImplementedError

    def prepare_model(self, gpu_id, rank, world_size):
        raise NotImplementedError

    def sync_model(self, gpu_id, rank, world_size):
        raise NotImplementedError

    def postprocess_model(self, gpu_id, rank, world_size):
        raise NotImplementedError

    def evaluate(self, results: list, data, graph):
        raise NotImplementedError


    @property
    def name(self):
        return self.module_name

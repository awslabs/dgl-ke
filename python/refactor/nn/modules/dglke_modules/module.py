import torch as th
from nn.loss.loss_generator import LossGenerator
from nn.metrics import MetricsEvaluator

# encoder, decoder are both inherited from Module. The default behavior of get_loss(), evaluate()
# does nothing as sometimes encoder does not need them. While for decoder, it's mandatory for it
# to overload these two functions.

class Module:
    def __init__(self, module_name):
        self.module_name = module_name
        # default metrics_evaluator and loss_generator do nothing
        self.metrics_evaluator = MetricsEvaluator()
        self.loss_generator = LossGenerator()

    def train(self):
        pass

    def eval(self):
        pass

    def save(self, save_path: str):
        raise NotImplementedError

    def load(self, load_path: str):
        raise NotImplementedError

    def share_memory(self):
        raise NotImplementedError

    def dense_parameters(self) -> list:
        # return dense trainable paramters of encoder
        return [th.nn.Parameter(th.zeros(1), requires_grad=False)]

    def sparse_parameters(self) -> list:
        # return sparse parameters for optimization
        return [th.nn.Parameter(th.zeros(1), requires_grad=False)]

    def prepare_distributed_training(self, gpu_id=-1, rank=0, world_size=-1):
        # default does nothing
        pass

    def attach_loss_generator(self, loss_generator: LossGenerator):
        # default does nothing
        self.loss_generator = loss_generator

    def attach_metrics_evaluator(self, metrics_evaluator: MetricsEvaluator):
        # default does nothing
        self.metrics_evaluator = metrics_evaluator

    def get_loss(self, results: list):
        # hooker func that can do nothing if no loss is needed. or it could be self-supervised loss or
        # supervised signal provided by data
        # used for training
        return self.loss_generator.get_total_loss(results)

    def evaluate(self, results: list):
        # default does nothing
        # used for validation/test.
        pass

    @property
    def name(self):
        return self.module_name

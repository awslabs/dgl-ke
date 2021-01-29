class Optimizer:
    def __init__(self):
        self.dense_optim = None
        self.sparse_optim = None

    def step(self, gpu_id):
        if self.dense_optim is not None:
            self.dense_optim.step()
        if self.sparse_optim is not None:
            self.sparse_optim.step(gpu_id)

    def zero_grad(self):
        self.dense_optim.zero_grad()
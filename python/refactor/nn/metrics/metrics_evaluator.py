# we hide the implementation of evaluator from KGEModel
# as one class should do one thing and fit() in KGEModel should not take responsibility
# to aggregate metric results
class MetricsEvaluator:
    def __init__(self):
        pass

    def evaluate(self, results: list, data):
        # different models might have different results. Thus results might contain different number
        # of elements to evaluate performance
        pass


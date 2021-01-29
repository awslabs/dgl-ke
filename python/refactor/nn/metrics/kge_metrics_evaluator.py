import torch as th
from .metrics_evaluator import MetricsEvaluator
from utils import get_scalar


class KGEMetricsEvaluator(MetricsEvaluator):
    def __init__(self, eval_filter=True):
        self.eval_filter = eval_filter
        super(MetricsEvaluator, self).__init__()

    def evaluate(self, results, data):
        # MARK - do we need lock here to prevent multi-thread from changing self.result
        # or we use subresult to store local result then sum them up for final aggregation?
        # pos_score: batch x 1 neg_score: batch x nodes mask: batch x nodes
        pos_score, neg_score, mask = results
        batch_size, _ = pos_score.shape[0]
        log = []
        for i in range(batch_size):
            ranking = get_scalar(self.__compute_ranking(pos_score[i], neg_score[i], mask[i]))
            log.append({
                'MRR': 1.0 / ranking,
                'MR': float(ranking),
                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                'HITS@10': 1.0 if ranking <= 10 else 0.0
            })
        return log

    def __compute_ranking(self, pos_score_i, neg_score_i, mask_i=None):
        if self.eval_filter:
            # mask_i will mask neg_score_i where the negative sample head/tail = positive sample head/tail
            ranking = th.sum(th.masked_select(neg_score_i >= pos_score_i, mask_i)) + 1
        else:
            ranking = th.sum(neg_score_i > pos_score_i) + 1
        return ranking



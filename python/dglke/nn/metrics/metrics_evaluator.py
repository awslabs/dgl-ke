import torch as th
from dglke.utils import get_scalar

# we hide the implementation of evaluator from KGEModel
# as one class should do one thing and fit() in KGEModel should not take responsibility
# to aggregate metric results
class MetricsEvaluator:
    def __init__(self):
        pass

    def evaluate(self, results, data, graph):
        # different models might have different results. Thus results might contain different number
        # of elements to evaluate performance
        pass

class RankingMetricsEvaluator(MetricsEvaluator):
    def __init__(self, eval_filter=True):
        self.eval_filter = eval_filter
        super(MetricsEvaluator, self).__init__()

    def evaluate(self, results, data, graph):
        # pos_score: batch x 1 neg_score: batch x nodes mask: batch x nodes
        logs = []
        pos_score, neg_score_head, neg_score_tail = results['pos_score'], results['neg_score_head'], results['neg_score_tail']
        logs += self.evaluate_ranking(pos_score, neg_score_head, data, graph, mode='head')
        logs += self.evaluate_ranking(pos_score, neg_score_tail, data, graph, mode='tail')
        return logs

    def evaluate_ranking(self, pos_score, neg_score, data, graph, mode):
        """ compute ranking for each triple

        This function firstly count all the negative samples that has larger scores than the positive samples. Then it
        filters out all the false negative sample containing in the count (including positive triple).

        Parameters
        ----------
        pos_score: torch.Tensor
            scores of positive sample, shape (B x 1)
        neg_score: torch.Tensor
            scores of negative sample, shape (B x num_entity)
        data: torch.Tensor
            training data indicies and metadata
        graph: DGLGraph
            graph that containing all the triples
        mode: str
            choices ['tail', 'head'], which type of negative score is

        Returns
        -------
        dcit
            log dict containing MRR, MR, HITS@1, HITS@3, HITS10
        """
        head, rel, tail, neg = data['head'], data['rel'], data['tail'], data['neg']
        b_size = data['head'].shape[0]
        pos_score = pos_score.view(b_size, -1)
        neg_score = neg_score.view(b_size, -1)
        ranking = th.zeros(b_size, 1, device=th.device('cpu'))
        for i in range(b_size):
            cand_idx = (neg_score[i] >= pos_score[i]).nonzero(as_tuple=False).cpu()
            # there might be precision error where pos_score[i] actually equals neg_score[i, pos_entity[i]]
            # we explicitly add this index to cand_idx to overcome this issue
            if mode == 'tail':
                if tail[i] not in cand_idx:
                    cand_idx = th.cat([cand_idx, tail[i].detach().cpu().view(-1, 1)], dim=0)
            else:
                if head[i] not in cand_idx:
                    cand_idx = th.cat([cand_idx, head[i].detach().cpu().view(-1, 1)], dim=0)
            cand_num = len(cand_idx)
            if not self.eval_filter:
                ranking[i] = cand_num
                continue
            if mode is 'tail':
                select = graph.has_edges_between(head[i], neg[cand_idx[:, 0]]).nonzero(as_tuple=False)[:, 0]
                if len(select) > 0:
                    select_idx = cand_idx[select].view(-1)
                    uid, vid, eid = graph.edge_ids(head[i], select_idx, return_uv=True)
                else:
                    raise ValueError('at least one element should have the same score with pos_score. That is itself!')
            else:
                select = graph.has_edges_between(neg[cand_idx[:, 0]], tail[i]).nonzero(as_tuple=False)[:, 0]
                if len(select) > 0:
                    select_idx = cand_idx[select].view(-1)
                    uid, vid, eid = graph.edge_ids(select_idx, tail[i], return_uv=True)
                else:
                    raise ValueError('at least one element should have the same score with pos_score. That is itself!')

            rid = graph.edata['tid'][eid]
            #  - 1 to exclude rank for positive score itself
            cand_num -= th.sum(rid == rel[i]) - 1
            ranking[i] = cand_num

        batch_size = pos_score.shape[0]
        log = []
        for i in range(batch_size):
            rank_i = get_scalar(ranking[i])
            log.append({
                'MRR': 1.0 / rank_i,
                'MR': float(rank_i),
                'HITS@1': 1.0 if rank_i <= 1 else 0.0,
                'HITS@3': 1.0 if rank_i <= 3 else 0.0,
                'HITS@10': 1.0 if rank_i <= 10 else 0.0
            })
        return log

# -*- coding: utf-8 -*-
#
# score_fun.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch as th
import torch.nn as nn
from dglke.utils.math import expmap0, project, mobius_add, tanh, artanh, MIN_NORM
import numpy as np
import os


def batched_l2_dist(a, b):
    a_squared = a.norm(dim=-1).pow(2)
    b_squared = b.norm(dim=-1).pow(2)

    squared_res = th.baddbmm(
        b_squared.unsqueeze(-2), a, b.transpose(-2, -1), alpha=-2
    ).add_(a_squared.unsqueeze(-1))
    res = squared_res.clamp_min_(1e-30).sqrt_()
    return res

def batched_l1_dist(a, b):
    res = th.cdist(a, b, p=1)
    return res

class TransEScore(nn.Module):
    """TransE score function
    Paper link: https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
    """
    def __init__(self, gamma, dist_func='l2'):
        super(TransEScore, self).__init__()
        self.gamma = gamma
        if dist_func == 'l1':
            self.neg_dist_func = batched_l1_dist
            self.dist_ord = 1
        else:  # default use l2
            self.neg_dist_func = batched_l2_dist
            self.dist_ord = 2

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=self.dist_ord, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_emb = head_emb.unsqueeze(1)
        rel_emb = rel_emb.unsqueeze(0)
        score = (head_emb + rel_emb).unsqueeze(2) - tail_emb.unsqueeze(0).unsqueeze(0)

        return self.gamma - th.norm(score, p=self.dist_ord, dim=-1)

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    # ! NFD -> put dict here or upper level? put dict here is beneficial for extension but
    # wrong dict key will cause trouble
    def predict(self, head, rel, tail):
        score = head + rel - tail
        return self.gamma - th.norm(score, p=self.dist_ord, dim=-1)

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        state_dict = self.cpu().state_dict()
        file_path = os.path.join(path, name)
        th.save(state_dict, file_path)

    def load(self, path, name):
        pass

    def create_neg(self, neg_head):
        gamma = self.gamma
        if neg_head:
            def fn(heads, relations, tails, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(-1, neg_sample_size, hidden_dim)
                tails = tails - relations
                tails = tails.reshape(-1, chunk_size, hidden_dim)
                return gamma - self.neg_dist_func(tails, heads)
            return fn
        else:
            def fn(heads, relations, tails, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads + relations
                heads = heads.reshape(-1, chunk_size, hidden_dim)
                tails = tails.reshape(-1, neg_sample_size, hidden_dim)
                return gamma - self.neg_dist_func(heads, tails)
            return fn

class TransRScore(nn.Module):
    """TransR score function
    Paper link: https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523
    """
    def __init__(self, gamma, projection_emb, relation_dim, entity_dim):
        super(TransRScore, self).__init__()
        self.gamma = gamma
        self.projection_emb = projection_emb
        self.relation_dim = relation_dim
        self.entity_dim = entity_dim

    def edge_func(self, edges):
        head = edges.data['head_emb']
        tail = edges.data['tail_emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        pass

    def predict(self, head, rel, tail, rel_id):
        projection = self.projection_emb(rel_id)
        projection = projection.reshape(-1, self.entity_dim, self.relation_dim)
        head = th.einsum('ab,abc->ac', head, projection)
        tail = th.einsum('ab,abc->ac', tail, projection)
        score  = head + rel - tail
        return self.gamma - th.norm(score, p=1, dim=-1)


    def prepare(self, g, gpu_id, trace=False):
        head_ids, tail_ids = g.all_edges(order='eid')
        projection = self.projection_emb(g.edata['id'], gpu_id, trace)
        projection = projection.reshape(-1, self.entity_dim, self.relation_dim)
        g.edata['head_emb'] = th.einsum('ab,abc->ac', g.ndata['emb'][head_ids], projection)
        g.edata['tail_emb'] = th.einsum('ab,abc->ac', g.ndata['emb'][tail_ids], projection)

    def create_neg_prepare(self, neg_head):
        if neg_head:
            def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
                # pos node, project to its relation
                projection = self.projection_emb(rel_id, gpu_id, trace)
                projection = projection.reshape(num_chunks, -1, self.entity_dim, self.relation_dim)
                tail = tail.reshape(num_chunks, -1, 1, self.entity_dim)
                tail = th.matmul(tail, projection)
                tail = tail.reshape(num_chunks, -1, self.relation_dim)

                # neg node, each project to all relations
                head = head.reshape(num_chunks, 1, -1, self.entity_dim)
                # (num_chunks, num_rel, num_neg_nodes, rel_dim)
                head = th.matmul(head, projection)
                return head, tail
            return fn
        else:
            def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
                # pos node, project to its relation
                projection = self.projection_emb(rel_id, gpu_id, trace)
                projection = projection.reshape(num_chunks, -1, self.entity_dim, self.relation_dim)
                head = head.reshape(num_chunks, -1, 1, self.entity_dim)
                head = th.matmul(head, projection)
                head = head.reshape(num_chunks, -1, self.relation_dim)

                # neg node, each project to all relations
                tail = tail.reshape(num_chunks, 1, -1, self.entity_dim)
                # (num_chunks, num_rel, num_neg_nodes, rel_dim)
                tail = th.matmul(tail, projection)
                return head, tail
            return fn

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def reset_parameters(self):
        self.projection_emb.init(1.0)

    def update(self, gpu_id=-1):
        pass

    def save(self, path, name):
        self.projection_emb.save(path, name + 'projection')

    def load(self, path, name):
        self.projection_emb.load(path, name + 'projection')

    def prepare_local_emb(self, projection_emb):
        self.global_projection_emb = self.projection_emb
        self.projection_emb = projection_emb

    def writeback_local_emb(self, idx):
        self.global_projection_emb.emb[idx] = self.projection_emb.emb.cpu()[idx]

    def load_local_emb(self, projection_emb):
        device = projection_emb.emb.device
        projection_emb.emb = self.projection_emb.emb.to(device)
        self.projection_emb = projection_emb

    def create_neg(self, neg_head):
        gamma = self.gamma
        if neg_head:
            def fn(heads, relations, tails, rel_ids, chunk_size, neg_sample_size):
                projection = self.projection_emb(rel_ids)
                projection = projection.reshape(-1, chunk_size, self.entity_dim, self.relation_dim)
                tails = tails.reshape(-1, chunk_size, 1, self.entity_dim)
                tails = th.matmul(tails, projection)
                tails = tails.reshape(-1, chunk_size, self.relation_dim)
                heads = heads.reshape(-1, 1, neg_sample_size, self.entity_dim)
                heads = th.matmul(heads, projection)

                relations = relations.reshape(-1, chunk_size, self.relation_dim)
                tails = tails - relations
                tails = tails.reshape(-1, chunk_size, 1, self.relation_dim)
                score = heads - tails
                return gamma - th.norm(score, p=1, dim=-1)
            return fn
        else:
            def fn(heads, relations, tails, rel_ids, chunk_size, neg_sample_size):
                projection = self.projection_emb(rel_ids)
                projection = projection.reshape(-1, chunk_size, self.entity_dim, self.relation_dim)
                heads = heads.reshape(-1, chunk_size, 1, self.entity_dim)
                heads = th.matmul(heads, projection)
                heads = heads.reshape(-1, chunk_size, self.relation_dim)
                tails = tails.reshape(-1, 1, neg_sample_size, self.entity_dim)
                tails = th.matmul(tails, projection)

                relations = relations.reshape(-1, chunk_size, self.relation_dim)
                heads = heads + relations
                heads = heads.reshape(-1, chunk_size, 1, self.relation_dim)
                score = heads - tails
                return gamma - th.norm(score, p=1, dim=-1)
            return fn

class DistMultScore(nn.Module):
    """DistMult score function
    Paper link: https://arxiv.org/abs/1412.6575
    """
    def __init__(self):
        super(DistMultScore, self).__init__()

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head * rel * tail
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': th.sum(score, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_emb = head_emb.unsqueeze(1)
        rel_emb = rel_emb.unsqueeze(0)
        score = (head_emb * rel_emb).unsqueeze(2) * tail_emb.unsqueeze(0).unsqueeze(0)

        return th.sum(score, dim=-1)

    def predict(self, head, rel, tail):
        score = (head * rel) * tail
        return th.sum(score, dim=-1)

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail

        return fn

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(-1, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                tmp = (tails * relations).reshape(-1, chunk_size, hidden_dim)
                return th.bmm(tmp, heads)

            return fn
        else:
            def fn(heads, relations, tails, chunk_size, neg_sample_size):
                hidden_dim = tails.shape[1]
                tails = tails.reshape(-1, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                tmp = (heads * relations).reshape(-1, chunk_size, hidden_dim)
                return th.bmm(tmp, tails)

            return fn


class ComplExScore(nn.Module):
    """ComplEx score function
    Paper link: https://arxiv.org/abs/1606.06357
    """

    def __init__(self):
        super(ComplExScore, self).__init__()

    def edge_func(self, edges):
        real_head, img_head = th.chunk(edges.src['emb'], 2, dim=-1)
        real_tail, img_tail = th.chunk(edges.dst['emb'], 2, dim=-1)
        real_rel, img_rel = th.chunk(edges.data['emb'], 2, dim=-1)

        score = real_head * real_tail * real_rel \
                + img_head * img_tail * real_rel \
                + real_head * img_tail * img_rel \
                - img_head * real_tail * img_rel
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': th.sum(score, -1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        real_head, img_head = th.chunk(head_emb, 2, dim=-1)
        real_tail, img_tail = th.chunk(tail_emb, 2, dim=-1)
        real_rel, img_rel = th.chunk(rel_emb, 2, dim=-1)

        score = (real_head.unsqueeze(1) * real_rel.unsqueeze(0)).unsqueeze(2) * real_tail.unsqueeze(0).unsqueeze(0) \
                + (img_head.unsqueeze(1) * real_rel.unsqueeze(0)).unsqueeze(2) * img_tail.unsqueeze(0).unsqueeze(0) \
                + (real_head.unsqueeze(1) * img_rel.unsqueeze(0)).unsqueeze(2) * img_tail.unsqueeze(0).unsqueeze(0) \
                - (img_head.unsqueeze(1) * img_rel.unsqueeze(0)).unsqueeze(2) * real_tail.unsqueeze(0).unsqueeze(0)

        return th.sum(score, dim=-1)

    def predict(self, head, rel, tail):
        real_head, img_head = th.chunk(head, 2, dim=-1)
        real_tail, img_tail = th.chunk(tail, 2, dim=-1)
        real_rel, img_rel = th.chunk(rel, 2, dim=-1)

        score = (real_head * real_rel) * real_tail \
                + (img_head * real_rel) * img_tail \
                + (real_head * img_rel) * img_tail \
                - (img_head * img_rel) * real_tail

        return th.sum(score, dim=-1)


    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail

        return fn

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = tails[..., :hidden_dim // 2]
                emb_imag = tails[..., hidden_dim // 2:]
                rel_real = relations[..., :hidden_dim // 2]
                rel_imag = relations[..., hidden_dim // 2:]
                real = emb_real * rel_real + emb_imag * rel_imag
                imag = -emb_real * rel_imag + emb_imag * rel_real
                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(-1, chunk_size, hidden_dim)
                heads = heads.reshape(-1, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                return th.bmm(tmp, heads)

            return fn
        else:
            def fn(heads, relations, tails, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = heads[..., :hidden_dim // 2]
                emb_imag = heads[..., hidden_dim // 2:]
                rel_real = relations[..., :hidden_dim // 2]
                rel_imag = relations[..., hidden_dim // 2:]
                real = emb_real * rel_real - emb_imag * rel_imag
                imag = emb_real * rel_imag + emb_imag * rel_real
                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(-1, chunk_size, hidden_dim)
                tails = tails.reshape(-1, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                return th.bmm(tmp, tails)
            return fn

class RESCALScore(nn.Module):
    """RESCAL score function
    Paper link: http://www.icml-2011.org/papers/438_icmlpaper.pdf
    """
    def __init__(self, relation_dim, entity_dim):
        super(RESCALScore, self).__init__()
        self.relation_dim = relation_dim
        self.entity_dim = entity_dim

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb'].unsqueeze(-1)
        rel = edges.data['emb']
        rel = rel.view(-1, self.relation_dim, self.entity_dim)
        score = head * th.matmul(rel, tail).squeeze(-1)
        # TODO: check if use self.gamma
        return {'score': th.sum(score, dim=-1)}
        # return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_emb = head_emb.unsqueeze(1).unsqueeze(1)
        rel_emb = rel_emb.view(-1, self.relation_dim, self.entity_dim)
        score = head_emb * th.einsum('abc,dc->adb', rel_emb, tail_emb).unsqueeze(0)

        return th.sum(score, dim=-1)

    def predict(self, head, rel, tail):
        rel = rel.view(-1, self.relation_dim, self.entity_dim)
        tail = tail.unsqueeze(-1)
        score = head * th.matmul(rel, tail).squeeze(-1).reshape(-1, self.entity_dim)
        return th.sum(score, dim=-1)

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(-1, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                tails = tails.unsqueeze(-1)
                relations = relations.view(-1, self.relation_dim, self.entity_dim)
                tmp = th.matmul(relations, tails).squeeze(-1)
                tmp = tmp.reshape(-1, chunk_size, hidden_dim)
                return th.bmm(tmp, heads)
            return fn
        else:
            def fn(heads, relations, tails, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                tails = tails.reshape(-1, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                heads = heads.unsqueeze(-1)
                relations = relations.view(-1, self.relation_dim, self.entity_dim)
                tmp = th.matmul(relations, heads).squeeze(-1)
                tmp = tmp.reshape(-1, chunk_size, hidden_dim)
                return th.bmm(tmp, tails)
            return fn

class RotatEScore(nn.Module):
    """RotatE score function
    Paper link: https://arxiv.org/abs/1902.10197
    """
    def __init__(self, gamma, emb_init):
        super(RotatEScore, self).__init__()
        self.gamma = gamma
        self.emb_init = emb_init

    def edge_func(self, edges):
        re_head, im_head = th.chunk(edges.src['emb'], 2, dim=-1)
        re_tail, im_tail = th.chunk(edges.dst['emb'], 2, dim=-1)

        phase_rel = edges.data['emb'] / (self.emb_init / np.pi)
        re_rel, im_rel = th.cos(phase_rel), th.sin(phase_rel)
        re_score = re_head * re_rel - im_head * im_rel
        im_score = re_head * im_rel + im_head * re_rel
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        score = th.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        return {'score': self.gamma - score.sum(-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        re_head, im_head = th.chunk(head_emb, 2, dim=-1)
        re_tail, im_tail = th.chunk(tail_emb, 2, dim=-1)

        phase_rel = rel_emb / (self.emb_init / np.pi)
        re_rel, im_rel = th.cos(phase_rel), th.sin(phase_rel)
        re_score = re_head.unsqueeze(1) * re_rel.unsqueeze(0) - im_head.unsqueeze(1) * im_rel.unsqueeze(0)
        im_score = re_head.unsqueeze(1) * im_rel.unsqueeze(0) + im_head.unsqueeze(1) * re_rel.unsqueeze(0)

        re_score = re_score.unsqueeze(2) - re_tail.unsqueeze(0).unsqueeze(0)
        im_score = im_score.unsqueeze(2) - im_tail.unsqueeze(0).unsqueeze(0)
        score = th.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        return self.gamma - score.sum(-1)

    def predict(self, head, rel, tail):
        re_head, im_head = th.chunk(head, 2, dim=-1)
        re_tail, im_tail = th.chunk(tail, 2, dim=-1)

        phase_rel = rel / (self.emb_init / np.pi)
        re_rel, im_rel = th.cos(phase_rel), th.sin(phase_rel)
        re_score = re_head * re_rel - im_head * im_rel
        im_score = re_head * im_rel + im_head * re_rel

        re_score = re_score - re_tail
        im_score = im_score - im_tail
        score = th.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        return self.gamma - score.sum(-1)


    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg(self, neg_head):
        gamma = self.gamma
        emb_init = self.emb_init
        if neg_head:
            def fn(heads, relations, tails, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = tails[..., :hidden_dim // 2]
                emb_imag = tails[..., hidden_dim // 2:]

                phase_rel = relations / (emb_init / np.pi)
                rel_real, rel_imag = th.cos(phase_rel), th.sin(phase_rel)
                real = emb_real * rel_real + emb_imag * rel_imag
                imag = -emb_real * rel_imag + emb_imag * rel_real
                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(-1, chunk_size, 1, hidden_dim)
                heads = heads.reshape(-1, 1, neg_sample_size, hidden_dim)
                score = tmp - heads
                score = th.stack([score[..., :hidden_dim // 2],
                                  score[..., hidden_dim // 2:]], dim=-1).norm(dim=-1)
                return gamma - score.sum(-1)

            return fn
        else:
            def fn(heads, relations, tails, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = heads[..., :hidden_dim // 2]
                emb_imag = heads[..., hidden_dim // 2:]

                phase_rel = relations / (emb_init / np.pi)
                rel_real, rel_imag = th.cos(phase_rel), th.sin(phase_rel)
                real = emb_real * rel_real - emb_imag * rel_imag
                imag = emb_real * rel_imag + emb_imag * rel_real

                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(-1, chunk_size, 1, hidden_dim)
                tails = tails.reshape(-1, 1, neg_sample_size, hidden_dim)
                score = tmp - tails
                score = th.stack([score[..., :hidden_dim // 2],
                                  score[..., hidden_dim // 2:]], dim=-1).norm(dim=-1)

                return gamma - score.sum(-1)

            return fn

class SimplEScore(nn.Module):
    """SimplE score function
    Paper link: http://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs.pdf
    """
    def __init__(self):
        super(SimplEScore, self).__init__()

    def edge_func(self, edges):
        head_i, head_j = th.chunk(edges.src['emb'], 2, dim=-1)
        tail_i, tail_j = th.chunk(edges.dst['emb'], 2, dim=-1)
        rel, rel_inv = th.chunk(edges.data['emb'], 2, dim=-1)
        forward_score = head_i * rel * tail_j
        backward_score = tail_i * rel_inv * head_j
        # clamp as official implementation does to avoid NaN output
        # might because of gradient explode
        score = th.clamp(1 / 2 * (forward_score + backward_score).sum(-1), -20, 20)
        return {'score': score}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_i, head_j = th.chunk(head_emb.unsqueeze(1), 2, dim=-1)
        tail_i, tail_j = th.chunk(tail_emb.unsqueeze(0).unsqueeze(0), 2, dim=-1)
        rel, rel_inv = th.chunk(rel_emb.unsqueeze(0), 2, dim=-1)
        forward_tmp = (head_i * rel).unsqueeze(2) * tail_j
        backward_tmp = (head_j * rel_inv).unsqueeze(2) * tail_i
        score = (forward_tmp + backward_tmp) * 1 / 2
        return th.sum(score, dim=-1)

    def predict(self, head, rel, tail):
        head_i, head_j = th.chunk(head, 2, dim=-1)
        tail_i, tail_j = th.chunk(tail, 2, dim=-1)
        rel, rel_inv = th.chunk(rel, 2, dim=-1)
        forward_tmp = (head_i * rel) * tail_j
        backward_tmp = (head_j * rel_inv) * tail_i
        score = (forward_tmp + backward_tmp) * 1 / 2
        return th.sum(score, dim=-1)

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn


    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, chunk_size, neg_sample_size):
                hidden_dim = tails.shape[1]
                tail_i = tails[..., :hidden_dim // 2]
                tail_j = tails[..., hidden_dim // 2:]
                rel = relations[..., : hidden_dim // 2]
                rel_inv = relations[..., hidden_dim // 2:]
                forward_tmp = (rel * tail_j).reshape(-1, chunk_size, hidden_dim//2)
                backward_tmp = (rel_inv * tail_i).reshape(-1, chunk_size, hidden_dim//2)
                heads = heads.reshape(-1, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                head_i = heads[..., :hidden_dim // 2, :]
                head_j = heads[..., hidden_dim // 2:, :]
                tmp = 1 / 2 * (th.bmm(forward_tmp, head_i) + th.bmm(backward_tmp, head_j))
                score = th.clamp(tmp, -20, 20)
                return score
            return fn
        else:
            def fn(heads, relations, tails, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                head_i = heads[..., :hidden_dim // 2]
                head_j = heads[..., hidden_dim // 2:]
                rel = relations[..., :hidden_dim // 2]
                rel_inv = relations[..., hidden_dim // 2:]
                forward_tmp = (head_i * rel).reshape(-1, chunk_size, hidden_dim//2)
                backward_tmp = (rel_inv * head_j).reshape(-1, chunk_size, hidden_dim//2)
                tails = tails.reshape(-1, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                tail_i = tails[..., :hidden_dim // 2, :]
                tail_j = tails[..., hidden_dim // 2:, :]
                tmp = 1 / 2 * (th.bmm(forward_tmp, tail_j) + th.bmm(backward_tmp, tail_i))
                score = th.clamp(tmp, -20, 20)
                return score
            return fn

class ConvEScore(nn.Module):
    """ConvE score function
    Paper link: https://arxiv.org/pdf/1707.01476.pdf
    """
    def __init__(self, hidden_dim, tensor_height, dropout_ratio: tuple = (0, 0, 0), batch_norm=False):
        super(ConvEScore, self).__init__()
        self._build_model(hidden_dim, tensor_height, dropout_ratio, batch_norm)


    def _build_model(self, hidden_dim, tensor_height, dropout_ratio, batch_norm):
        # get height of reshape tensor
        assert hidden_dim % tensor_height == 0, 'input dimension %d must be divisible to tensor height %d' % (hidden_dim, tensor_height)
        h = tensor_height
        w = hidden_dim // h
        conv = []
        if batch_norm:
            conv += [nn.BatchNorm2d(1)]
        if dropout_ratio[0] != 0:
            conv += [nn.Dropout(p=dropout_ratio[0])]
        conv += [nn.Conv2d(1, 32, 3, 1, 0, bias=True)]
        if batch_norm:
            conv += [nn.BatchNorm2d(32)]
        conv += [nn.ReLU()]
        if dropout_ratio[1] != 0:
            conv += [nn.Dropout2d(p=dropout_ratio[1])]
        self.conv = nn.Sequential(*conv)
        fc = []
        linear_dim = 32 * (h* 2 - 2) * (w - 2)
        fc += [nn.Linear(linear_dim, hidden_dim)]
        if dropout_ratio[2] != 0:
            fc += [nn.Dropout(p=dropout_ratio[2])]
        if batch_norm:
            fc += [nn.BatchNorm1d(hidden_dim)]
        fc += [nn.ReLU()]
        self.fc = nn.Sequential(*fc)

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = self.model(head, rel, tail)
        return {'score': score}


    def infer(self, head_emb, rel_emb, tail_emb):
        pass

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        if neg_head:
            def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
                return head, tail
            return fn
        else:
            def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
                return head, tail
            return fn

    def forward(self, embs, mode='all', bmm=False):
        """
        Parameters
        ----------
        concat_emb : Tensor
            embedding concatenated from head and tail and reshaped
        tail_emb : Tensor
            tail embedding
        mode : str
            choice = ['lhs', 'rhs', 'full']. Which part of score function to perform. This is used to accelerate test process.
        """

        if mode in ['all', 'lhs']:
            concat_emb = embs[0]
            # reshape tensor to fit in conv
            if concat_emb.dim() == 3:
                batch, height, width = concat_emb.shape
                concat_emb = concat_emb.reshape(batch, 1, height, width)
            x = self.conv(concat_emb)
            x = x.view(x.shape[0], -1)
            fc = self.fc(x)
            if mode == 'lhs':
                return fc
        else:
            fc = embs[0]

        tail_emb, bias = embs[1:]

        if not bmm:
            assert fc.dim() == tail_emb.dim() == bias.dim(), 'batch operation only allow embedding with same dimension'
            x = th.sum(fc * tail_emb, dim=-1, keepdim=True)
        else:
            if tail_emb.dim() == 3:
                tail_emb = tail_emb.transpose(1, 2)
                x = th.bmm(fc, tail_emb)
                bias = bias.transpose(1, 2).expand_as(x)
            else:
                tail_emb = tail_emb.transpose(1, 0)
                x = th.mm(fc, tail_emb)
                bias = bias.transpose(1, 0).expand_as(x)
        x = x + bias
        return x

    def reset_parameters(self):
        # use default init tech of pytorch
        pass

    def update(self, gpu_id=-1):
        pass

    def save(self, path, name):
        file_name = os.path.join(path, name)
        # MARK - is .cpu() available if it's already in CPU ?
        th.save(self.cpu().state_dict(), file_name)

    def load(self, path, name):
        file_name = os.path.join(path, name)
        # TODO: lingfei - determine whether to map location here
        self.load_state_dict(th.load(file_name))

    def create_neg(self, neg_head):
        pass

class ATTHScore(nn.Module):
    def __init__(self):
        super(ATTHScore, self).__init__()

    def predict(self, head, head_bias, rel, rel_diag, curvature, context, scale, tail, tail_bias):
        hidden_dim = head.shape[-1]
        curvature = th.nn.functional.softplus(curvature)
        rot_mat, ref_mat = th.chunk(rel_diag, 2, dim=1)
        rot_q = givens_rotations(rot_mat, head,).view((-1, 1, hidden_dim))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, hidden_dim))
        cands = th.cat([ref_q, rot_q], dim=1)
        context = context.view(-1, 1, hidden_dim)
        att_weights = th.sum(context * cands * scale, dim=-1, keepdim=True)
        att_weights = th.nn.functional.softmax(att_weights, dim=1)
        att_q = th.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, curvature)
        rel = expmap0(rel, curvature)
        res = project(mobius_add(lhs, rel, curvature), curvature)
        score = - hyp_distance_multi_c(res, tail, curvature) ** 2 + head_bias + tail_bias
        return score.squeeze(-1)

    def create_neg(self, neg_heads):
        if neg_heads:
            def fn(head, head_bias, rel, rel_diag, curvature, context, scale, tail, tail_bias, chunk_size, neg_sample_size):
                num_neg_chunk = head.shape[0] // neg_sample_size
                num_chunk = rel.shape[0] // chunk_size
                head = head.view(num_neg_chunk, neg_sample_size, -1)
                head_bias = head_bias.view(num_neg_chunk, neg_sample_size, -1).unsqueeze(1)
                rel = rel.view(num_chunk, chunk_size, -1)
                rel_diag = rel_diag.view(num_chunk, chunk_size, -1)
                curvature = curvature.view(num_chunk, chunk_size, -1)
                context = context.view(num_chunk, chunk_size, -1)
                tail = tail.view(num_chunk, chunk_size, -1).unsqueeze(2)
                tail_bias = tail_bias.view(num_chunk, chunk_size, -1).unsqueeze(2)

                c = th.nn.functional.softplus(curvature)
                rot_mat, ref_mat = th.chunk(rel_diag, 2, dim=-1)
                rot_q = givens_rotations_bmm(rot_mat, head).unsqueeze(-2)
                ref_q = givens_reflection_bmm(ref_mat, head).unsqueeze(-2)
                cands = th.cat([ref_q, rot_q], dim=-2)
                context_vec = context.unsqueeze(2).unsqueeze(3)
                attn_weights = th.sum(context_vec * cands * scale, dim=-1, keepdim=True)
                attn_weights = th.nn.functional.softmax(attn_weights, dim=-2)
                att_q = th.sum(attn_weights * cands, dim=-2)
                expand_c = c.unsqueeze(2)
                lhs = expmap0(att_q, expand_c)
                rel = expmap0(rel, c).unsqueeze(2)
                res = project(mobius_add(lhs, rel, expand_c), expand_c)
                score = - hyp_distance_multi_c(res, tail, expand_c) ** 2 + head_bias + tail_bias
                return score.squeeze(-1)

        else:
            def fn(head, head_bias, rel, rel_diag, curvature, context, scale, tail, tail_bias, chunk_size, neg_sample_size):
                num_chunk = head.shape[0] // chunk_size
                num_neg_chunk = tail.shape[0] // neg_sample_size
                hidden_dim = head.shape[-1]
                tail = tail.view(num_neg_chunk, neg_sample_size, -1)
                tail_bias = tail_bias.view(num_neg_chunk, neg_sample_size, -1)
                head_bias = head_bias.view(num_chunk, chunk_size, -1)

                c = th.nn.functional.softplus(curvature)
                rot_mat, ref_mat = th.chunk(rel_diag, 2, dim=-1)
                rot_q = givens_rotations(rot_mat, head).view(-1, 1, hidden_dim)
                ref_q = givens_reflection(ref_mat, head).view(-1, 1, hidden_dim)
                cands = th.cat([ref_q, rot_q], dim=1)
                context_vec = context.view(-1, 1, hidden_dim)
                att_weights = th.sum(context_vec * cands * scale, dim=-1, keepdim=True)
                att_weights = th.nn.functional.softmax(att_weights, dim=1)
                att_q = th.sum(att_weights * cands, dim=1)
                lhs = expmap0(att_q, c)
                rel = expmap0(rel, c)
                res = project(mobius_add(lhs, rel, c), c)
                c = c.view(c.shape[0] // chunk_size, chunk_size, -1)
                score = - hyp_distance_multi_c_bmm(res.view(res.shape[0] // chunk_size, chunk_size, -1), tail, c) ** 2
                score = score.unsqueeze(-1) + head_bias.unsqueeze(2) + tail_bias.unsqueeze(1)
                return score.squeeze(-1)
        return fn


def hyp_distance_multi_c(x, v, c):
    """Hyperbolic distance on Poincare balls with varying curvatures c.

    Parameters
    ----------
    x: torch.Tensor
        size B x d with hyperbolic queries
    v: torch.Tensor
        hyperbolic queries, shape (B x d)
    c: torch.Tensor
        shape (B x d) with absolute hyperbolic curvatures

    Returns
    -------
    torch.Tensor
        hyperbolic distances, B x n_entities matrix with all pairs distances
    """
    sqrt_c = c ** 0.5
    vnorm = th.norm(v, p=2, dim=-1, keepdim=True)
    xv = th.sum(x * v / vnorm, dim=-1, keepdim=True)
    gamma = tanh(sqrt_c * vnorm) / sqrt_c
    x2 = th.sum(x * x, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
    c2 = 1 - c * x2
    num = th.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv)
    denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c

def hyp_distance_multi_c_bmm(x, v, c):
    """batch matrix multiplication verison of Hyperbolic distance on Poincare balls with varying curvatures c.

    Parameters
    ----------
    x: torch.Tensor
        size num_chunk x chunk_size x d with hyperbolic queries
    v: torch.Tensor
        hyperbolic queries, shape (num_chunk x neg_sample_size x d)
    c: torch.Tensor
        shape (num_chunk x chunk_size x d) with absolute hyperbolic curvatures

    Returns
    -------
    torch.Tensor
        hyperbolic distances, (num_chunk x chunk_size x neg_sample_size x d) matrix with all pairs distances
    """
    sqrt_c = c ** 0.5
    vnorm = th.norm(v, p=2, dim=-1, keepdim=True).transpose(1, 2).expand(x.shape[0], -1, -1)
    xv = th.bmm(x, v.transpose(1, 2).expand(x.shape[0], -1, -1)) / vnorm
    gamma = tanh(th.bmm(sqrt_c, vnorm)) / sqrt_c
    x2 = th.sum(x * x, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
    c2 = 1 - c * x2
    num = th.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv)
    denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c


def givens_rotations(r, x):
    """ Givens rotations.

    Parameters
    ----------
    r: torch.Tensor
        shape (N x d), rotation parameters
    x: torch.Tensor
        shape (N x d), points to rotate
    Returns
    -------
    torch.Tensor
        shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / th.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(MIN_NORM)
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * th.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))

def givens_rotations_bmm(r, x):
    """ batch matrix multiplication version of Givens rotations.

    Parameters
    ----------
    r: torch.Tensor
        shape (num_chunk x chunk_size x d), rotation parameters
    x: torch.Tensor
        shape (num_chunk x num_neg_sample x d), points to rotate
    Returns
    -------
    torch.Tensor
        shape (num_chunk x chunk_size x num_neg_sample x d ) representing rotation of x by r
    """
    givens = r.view((r.shape[0], r.shape[1], -1, 2))
    givens = givens / th.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(MIN_NORM)
    x = x.view((x.shape[0], x.shape[1], -1, 2))
    x_rot_a = th.einsum('bcde,bnde->bcnde', givens[:, :, :, 0:1].expand(-1, -1, -1, 2), x.expand(givens.shape[0], -1, -1, -1))
    x_rot_b = th.einsum('bcde,bnde->bcnde', givens[:, :, :, 1:].expand(-1, -1, -1, 2),
                        th.cat((-x[:, :, :, 1:], x[:, :, :, 0:1]), dim=-1).expand(givens.shape[0], -1, -1, -1))
    x_rot = x_rot_a + x_rot_b
    return x_rot.view((r.shape[0], r.shape[1], x.shape[1], -1))

def givens_reflection(r, x):
    """ Givens reflection.

    Parameters
    ----------
    r: torch.Tensor
        shape (N x d), reflection parameters
    x: torch.Tensor
        shape (N x d), points to reflect
    Returns
    -------
    torch.Tensor
        shape (N x d) representing reflection of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / th.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(MIN_NORM)
    x = x.view((r.shape[0], -1, 2))
    x_ref = givens[:, :, 0:1] * th.cat((x[:, :, 0:1], -x[:, :, 1:]), dim=-1) + givens[:, :, 1:] * th.cat(
        (x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_ref.view((r.shape[0], -1))

def givens_reflection_bmm(r, x):
    """ batch matrix multiplication version of Givens reflections.

    Parameters
    ----------
    r: torch.Tensor
        shape (num_chunk x neg_sample_size x d), reflection parameters
    x: torch.Tensor
        shape (num_chunk x chunk_size x d), points to reflect
    Returns
    -------
    torch.Tensor
        shape (num_chunk x chunk_size x num_neg_sample x d ) representing reflection of x by r
    """
    givens = r.view((r.shape[0], r.shape[1], -1, 2))
    givens = givens / th.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(MIN_NORM)
    x = x.view((x.shape[0], x.shape[1], -1, 2))
    x_ref_a = th.einsum('bcde,bnde->bcnde', givens[:, :, :, 0:1].expand(-1, -1, -1, 2),
                        th.cat((x[:, :, :, 0:1], -x[:, :, :, 1:]), dim=-1).expand(givens.shape[0], -1, -1, -1))
    x_ref_b = th.einsum('bcde,bnde->bcnde', givens[:, :, :, 1:].expand(-1, -1, -1, 2), th.cat((x[:, :, :, 1:], x[:, :, :, 0:1]), dim=-1).expand(givens.shape[0], -1, -1, -1))
    x_ref = x_ref_a + x_ref_b
    return x_ref.view((r.shape[0], r.shape[1], x.shape[1], -1))



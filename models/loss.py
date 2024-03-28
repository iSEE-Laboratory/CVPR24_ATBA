import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from queue import Queue


class BoundaryDetector:
    def __init__(self, kernel_size, win_scale, args):
        self.kernel_size = kernel_size
        self.win_scale = win_scale

        # class-agnostic kernel
        self.kernel = torch.zeros((kernel_size, kernel_size))
        center = self.kernel_size // 2
        self.kernel[:center, :center] = 1
        self.kernel[center:, center:] = 1
        self.kernel[:center, center:] = -1
        self.kernel[center:, :center] = -1
        self.kernel[center, center] = 0

        self.args = args

        self.special_win_scale = {}

        # transition kernel
        self.cs_size = args.cs_kernel
        cs_center = self.cs_size // 2
        self.cls_score_kernel = torch.zeros((self.cs_size, 2))
        self.cls_score_kernel[:cs_center, 0] = 1
        self.cls_score_kernel[cs_center:, 0] = -1
        self.cls_score_kernel[:cs_center, 1] = -1
        self.cls_score_kernel[cs_center:, 1] = 1

    def pair_sim(self, prob):
        # x (b, t, c)
        b, t = prob.shape[0:2]
        px = torch.repeat_interleave(prob, repeats=t, dim=1)  # (b, t*t, c)
        qx = prob.repeat(1, t, 1)     # (b, t*t, c)
        qx = (px + qx) / 2
        kl = torch.sum(F.kl_div((qx + 1e-32).log(), px, reduction='none'), dim=-1).reshape(b, t, t)  # (b, t, t)
        js = (kl + kl.transpose(1, 2)) / 2
        js = - (js / math.log(2) * 2 - 1)  # (-1, 1)
        return js       # (b, t, t)

    def set_kernel(self, kernel_size):
        center = kernel_size // 2
        kernel = torch.zeros((kernel_size, kernel_size))
        kernel[:center, :center] = 1
        kernel[center:, center:] = 1
        kernel[:center, center:] = -1
        kernel[center:, :center] = -1
        kernel[center, center] = 0
        return kernel

    def get_bdy_score(self, prob):
        kernel_size = self.kernel_size
        kernel = self.kernel

        t, c = prob.shape
        with torch.no_grad():
            # temporal windows (t, ks, c)
            unfold = F.unfold(prob.detach().unsqueeze(0).unsqueeze(0), kernel_size=(kernel_size, c),
                              padding=(kernel_size // 2, 0))[0].permute(1, 0).reshape(t, kernel_size, c)

            # pair-wise similarity in each window
            slide_sim = self.pair_sim(unfold)  # (t, ks, ks)
            dig_score = torch.mean(kernel.repeat(t, 1, 1).to(prob.device) * slide_sim, dim=(-1, -2))  # (t,)
        bdy_score = dig_score
        return bdy_score

    def get_cls_score(self, prob, tr, bdy_pts):
        kernel_size = self.cs_size
        kernel = self.cls_score_kernel

        norm_prob = prob
        t = prob.shape[0]
        cs_lst = []
        for tr_idx in range(len(tr) - 1):
            first_seg, second_seg = tr[tr_idx], tr[tr_idx + 1]
            center = kernel_size // 2
            padding_prob = torch.zeros((t + 2 * center, 2), device=norm_prob.device)
            padding_prob[center: center + t] = norm_prob[:, [first_seg, second_seg]]
            unfold = F.unfold(padding_prob.unsqueeze(0).unsqueeze(0),
                              kernel_size=(kernel_size, 2))[0].permute(1, 0).reshape(t, kernel_size, 2)[bdy_pts]
            score = torch.mean(kernel.repeat(len(bdy_pts), 1, 1).to(norm_prob.device) * unfold, dim=(-1, -2))
            cs_lst.append(score)
        return torch.stack(cs_lst, dim=0)

    def find_bdy_pts_with_cls_transition(self, bdy_score, pt_num, window_len, prob, tr, name):
        candidate_pts = []
        t = len(bdy_score)
        bdy_score_backup = bdy_score.clone()
        for _ in range(self.args.candidate_mul * pt_num):
            val, idx = torch.max(bdy_score, dim=0)
            if val == float('-inf'):
                break
            candidate_pts.append(idx)
            left = max(0, idx - window_len)
            right = min(t, idx + window_len + 1)
            bdy_score[left: right] = float('-inf')  # nms

        # find the optimal alignment
        candidate_pts, _ = torch.sort(torch.stack(candidate_pts, dim=0))
        cls_score = self.get_cls_score(prob, tr, candidate_pts).cpu()  # (tr_len-1, n)
        cls_score += bdy_score_backup[candidate_pts].unsqueeze(0).repeat(len(tr) - 1, 1)
        optim_idx = self.dp(cls_score)
        bdy_pts = candidate_pts[optim_idx]

        assert len(bdy_pts) == pt_num
        return bdy_pts

    def forward(self, epoch, b_logit, mask, transcript, name):
        # b_logit (b, t, c)
        bdy_lst = []
        for i in range(b_logit.shape[0]):
            logit = b_logit[i][mask[i]].detach()
            prob = torch.softmax(logit, dim=-1)
            t, c = logit.shape
            t_len = len(transcript[i])
            window_len = int(t / t_len * self.win_scale)

            bdy_score = self.get_bdy_score(prob).cpu()
            bdy_score[0] = float('-inf')    # the 1st segment needs at least 1 frame

            bdy_pts = self.find_bdy_pts_with_cls_transition(bdy_score, t_len-1, window_len, prob, transcript[i], name[i])

            bdy_lst.append(bdy_pts)

        return bdy_lst

    def dp(self, cls_score):
        # cls_score (tr_len-1, n)
        cost = - cls_score
        empty_cost = 0
        bdy_idx = []

        n_tr, n_x = cls_score.shape[0], cls_score.shape[1]
        dist_mat = torch.ones((n_x, 2 * n_tr + 1), device=cls_score.device) * float('inf')
        # 0: from (i-1, j)，1: from (i-1, j-1)，2: form (i-1, j-2)
        dir_mat = torch.zeros((n_x, 2 * n_tr + 1), device=cls_score.device, dtype=torch.int)
        dist_mat[:n_x - n_tr, 0] = empty_cost
        dist_mat[:n_x - n_tr + 1, 1] = cost[0, :n_x - n_tr + 1]
        dir_mat[:n_x - n_tr + 1, 1] = 1

        for ii in range(1, n_x):
            for jj in range(2, 2 * n_tr + 1):
                if jj % 2 == 0:
                    # empty symbol
                    if dist_mat[ii - 1, jj] < dist_mat[ii - 1, jj - 1]:
                        dist_mat[ii, jj] = empty_cost + dist_mat[ii - 1, jj]
                        dir_mat[ii, jj] = 0
                    else:
                        dist_mat[ii, jj] = empty_cost + dist_mat[ii - 1, jj - 1]
                        dir_mat[ii, jj] = 1
                else:
                    # transition symbol
                    if dist_mat[ii - 1, jj - 1] < dist_mat[ii - 1, jj - 2]:
                        dist_mat[ii, jj] = cost[jj // 2, ii] + dist_mat[ii - 1, jj - 1]
                        dir_mat[ii, jj] = 1
                    else:
                        dist_mat[ii, jj] = cost[jj // 2, ii] + dist_mat[ii - 1, jj - 2]
                        dir_mat[ii, jj] = 2

        # backtracking
        cur_i = n_x - 1
        cur_j = 2 * n_tr if dist_mat[-1, -1] < dist_mat[-1, -2] else 2 * n_tr - 1
        while cur_i >= 0:
            if cur_j % 2 == 1:
                bdy_idx.append(cur_i)
            cur_j -= dir_mat[cur_i, cur_j]
            cur_i -= 1
        return list(reversed(bdy_idx))


class LossFn(nn.Module):
    def __init__(self, num_cls, bg_cls, args, device):
        super(LossFn, self).__init__()
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma

        self.num_cls = num_cls
        self.bg_cls = bg_cls
        
        self.bdydet = BoundaryDetector(args.bdy_kernel, args.bdy_scale, args)

        self.warm_epc = args.warm_epc

        self.T = args.cts_temp

        self.args = args

        self.device = device

    def forward(self, epoch, tok_logit, fr_logit, mask, transcript, vid_multi_hot, feat, name):
        # tok_logit (b, cls), fr_logit (b, t, cls), mask (b, t), vid_multi_hot (b, cls), feat (b, t, c)
        # transcript: list(tensor)
        tok_loss = self.token_cls_loss(tok_logit, vid_multi_hot)    # vid loss in the paper
        if epoch < self.warm_epc:
            warm_loss = tok_loss
            return {
                'total': warm_loss,
                'tok_loss': tok_loss,
                'fr_loss': 0.0,
                'glc_loss': 0.0,
            }
        bdy_lst = self.bdydet.forward(epoch, fr_logit, mask, transcript, name)
        pse_la = self.generate_pseudo_label(bdy_lst, fr_logit, mask, transcript, name)

        fr_loss = self.fr_cls_loss(fr_logit, mask, pse_la, name)
        glc_loss = self.contrast_loss(feat[:, :self.num_cls, :], feat[:, self.num_cls:, :], mask, pse_la, fr_logit)
        total_loss = self.alpha * tok_loss + self.beta * fr_loss + self.gamma * glc_loss
        return {
            'total': total_loss,
            'tok_loss': tok_loss,
            'fr_loss': fr_loss,
            'glc_loss': glc_loss,
        }

    def token_cls_loss(self, tok_logit, vid_multi_hot):
        b = tok_logit.shape[0]
        tok_prob = torch.sigmoid(tok_logit)
        return F.binary_cross_entropy(tok_prob, vid_multi_hot.float())

    def fr_cls_loss(self, fr_logit, mask, pse_la, name):
        if self.args.bgw < 1.0:
            assert self.args.dataset == 'hollywood' or self.args.dataset == 'crosstask'
            weight = torch.ones(self.num_cls).to(fr_logit.device)
            weight[self.bg_cls] = self.args.bgw     # set small weight to pseudo background frames
            fr_logit = fr_logit[mask]     # (n, cls)
            pse_la = pse_la[mask]
            return F.cross_entropy(fr_logit, pse_la, weight=weight)

        ce = F.cross_entropy(fr_logit.transpose(-1, -2), pse_la, ignore_index=-100, reduction='none')
        return torch.mean(ce[mask])

    def generate_pseudo_label(self, bdy_lst, fr_logit, mask, transcript, name):
        b, max_t = mask.shape
        pla_lst = []
        for i in range(b):
            bdy = bdy_lst[i]
            tr = transcript[i]
            v_pla = torch.ones(max_t, device=mask.device) * -100
            for j in range(len(tr)):
                if j == 0:
                    start = 0
                    end = bdy[j]    # open interval
                elif j == len(tr) - 1:
                    start = bdy[j - 1]
                    end = torch.sum(mask[i])
                else:
                    start = bdy[j - 1]
                    end = bdy[j]
                v_pla[start: end] = tr[j]

            pla_lst.append(v_pla)
        return torch.stack(pla_lst, dim=0).long()      # (b, t)

    def contrast_loss(self, tok_feat, fr_feat, mask, pse_la, fr_logit, aggregation='mean'):
        # tok_feat (b, n_cls, c), fr_feat (b, t, c)
        tok_feat = tok_feat.detach()
        b = tok_feat.shape[0]
        loss_lst = []
        for i in range(b):
            labels = torch.unique(pse_la[i][mask[i]])
            tok, fr = tok_feat[i], fr_feat[i][mask[i]]       # (t, c)
            logit = fr_logit[i][mask[i]]
            la_feat_lst = []
            for la in labels:
                idx = pse_la[i][mask[i]] == la      # (t,)
                la_feat = torch.mean(fr[idx], dim=0)
                la_feat_lst.append(la_feat)
            la_feats = torch.stack(la_feat_lst, dim=0)      # (n', c)
            tok = F.normalize(tok, dim=-1)      # (n, c)
            la_feats = F.normalize(la_feats, dim=-1)    # (n', c)
            sim = torch.mm(la_feats, tok.T) / self.T   # (n', n)

            c_loss = F.cross_entropy(sim, labels, ignore_index=-1)
            loss_lst.append(c_loss)
        if aggregation is None:
            return torch.stack(loss_lst, dim=0)
        else:
            return torch.mean(torch.stack(loss_lst, dim=0))

import math

import torch
from torch import nn


class KNNLoss(nn.Module):
    def __init__(self, n_bgroups, k):
        super(KNNLoss, self).__init__()
        self.n_bgroups = n_bgroups
        self.k = k

    def forward(self, expected, actual, output_indices=False):
        # [batch, frames, points, pos + bgroups]
        batches = expected.shape[0]
        frames = expected.shape[1]

        # p_e = torch.reshape(expected[..., :3].contiguous().transpose(1, 2), (batches, expected.shape[2], -1))
        # p_a = torch.reshape(actual[..., :3].contiguous().transpose(1, 2), (batches, actual.shape[2], -1))
        p_e = expected[..., :3].clone()
        p_a = actual[..., :3].clone()
        v_e = expected[..., :3].clone()
        v_e[:, 1:] = v_e[:, 1:] - v_e[:, :-1]
        v_e[:, 0] = 0
        v_e = torch.reshape(v_e.transpose(1, 2), (batches, expected.shape[2], -1))
        v_a = actual[..., :3].clone()
        v_a[:, 1:] = v_a[:, 1:] - v_a[:, :-1]
        v_a[:, 0] = 0
        v_a = torch.reshape(v_a.transpose(1, 2), (batches, actual.shape[2], -1))

        # [batch, points]
        bgroups_e = torch.argmax(expected[0, 0, :, 3:].contiguous(), dim=-1)
        bgroups_a = torch.argmax(actual[0, 0, :, 3:].contiguous(), dim=-1)
        # print(torch.argmax(expected[0, 0, :, 3:].contiguous(), dim=-1))
        # print(torch.argmax(expected[1, 0, :, 3:].contiguous(), dim=-1))
        # print(torch.argmax(expected[-1, 0, :, 3:].contiguous(), dim=-1))
        # print(torch.argmax(expected[-2, 0, :, 3:].contiguous(), dim=-1))

        dists_glob_p = []
        dists_norm_p = []
        dists_v = []

        g_indices = []
        base_indices = torch.arange(p_a.shape[2], dtype=torch.int64, device=p_a.device)

        for g in range(self.n_bgroups):
            p_e_ = p_e.transpose(0, 2)[bgroups_e == g].transpose(0, 2).clone()
            p_a_ = p_a.transpose(0, 2)[bgroups_a == g].transpose(0, 2).clone()
            v_e_ = v_e.transpose(0, 1)[bgroups_e == g].transpose(0, 1).clone()
            v_a_ = v_a.transpose(0, 1)[bgroups_a == g].transpose(0, 1).clone()

            group_indices = base_indices[bgroups_a == g].clone()

            norm_p_e_ = p_e_ - torch.mean(p_e_, dim=(1, 2), keepdim=True)
            norm_p_e_ = norm_p_e_ / torch.std(norm_p_e_, dim=(2, 3), keepdim=True)
            norm_p_a_ = p_a_ - torch.mean(p_a_, dim=(1, 2), keepdim=True)
            norm_p_a_ = norm_p_a_ / torch.std(norm_p_a_, dim=(2, 3), keepdim=True)

            p_e_ = torch.reshape(p_e_.transpose(1, 2), (batches, p_e_.shape[2], -1))
            p_a_ = torch.reshape(p_a_.transpose(1, 2), (batches, p_a_.shape[2], -1))
            norm_p_e_ = torch.reshape(norm_p_e_.transpose(1, 2), (batches, norm_p_e_.shape[2], -1))
            norm_p_a_ = torch.reshape(norm_p_a_.transpose(1, 2), (batches, norm_p_a_.shape[2], -1))

            dists_glob_p_ = torch.cdist(p_e_, p_a_, p=2)
            dists_norm_p_ = torch.cdist(norm_p_e_, norm_p_a_, p=2)
            dists_v_ = torch.cdist(v_e_, v_a_, p=2)

            (dists_glob_p_, indices) = torch.sort(dists_glob_p_, dim=-1)
            (dists_norm_p_, _) = torch.sort(dists_norm_p_, dim=-1)
            dists_v_ = torch.gather(dists_v_, -1, indices)
            g_indices.append(group_indices[indices])

            dists_glob_p.append(dists_glob_p_[..., :self.k])
            dists_norm_p.append(dists_norm_p_[..., :self.k])
            dists_v.append(dists_v_[..., :self.k])

        loss_glob_p = torch.mean(torch.cat(dists_glob_p, dim=1)) / math.sqrt(frames)
        loss_norm_p = torch.mean(torch.cat(dists_norm_p, dim=1)) / math.sqrt(frames)
        loss_v = torch.mean(torch.cat(dists_v, dim=1)) / math.sqrt(frames)

        if output_indices:
            return loss_glob_p, loss_norm_p, loss_v, g_indices
        else:
            return loss_glob_p, loss_norm_p, loss_v
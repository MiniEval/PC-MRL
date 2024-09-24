import json
import math
import os
import random
import sys

import numpy as np
import pyqtgraph
import torch

from data.cmu import CMUDataLoader, PSLaFANDataLoader
from data.h36m import H36MDataLoader
from data.lafan import LaFANDataLoader
from models.inpainter_citl import CITL, CITLEncoder, CITLInterm, CITLDecoder
from models.utils.interp import Lerp
from models.utils.npss import compute_npss

from viewport.visualiser import Visualiser


class InpainterEval:
    def _augment_roll(self, glob_q):
        # glob_q: [batch, frames, joints, 4]
        joint_tails = self.tgt_skeleton.joint_tails.unsqueeze(0)
        kp_rot = self.tgt_skeleton.q_to_keypoint(glob_q[:, :1].transpose(1, 2), joint_tails)
        kp_rot[..., 3] = 1
        kp_rot[..., 4] = 0
        q_aug = self.tgt_skeleton.batch_single_ik(joint_tails, kp_rot[..., :3], kp_rot[..., 3:])
        q_inv = torch.cat([glob_q[:, :1, :, :1], -glob_q[:, :1, :, 1:]], dim=-1)
        q_mod = self.tgt_skeleton._qmul(q_inv, q_aug)
        q_mod = q_mod * self.tgt_skeleton.roll_mask.view(-1, 1)
        q_mod[..., 0] = q_mod[..., 0] + (1 - self.tgt_skeleton.roll_mask.int())

        # print(glob_q.shape, q_mod.shape, q_aug.shape)
        q = self.tgt_skeleton._qmul(glob_q, q_mod.repeat(1, glob_q.shape[1], 1, 1))
        q_mod_inv = torch.cat([q_mod[..., :1], -q_mod[..., 1:]], dim=-1)
        return q, q_mod_inv

    def _augment_q_offset(self, glob_q):
        # glob_q: [batch, frames, joints, 4]
        q_base = glob_q[:, :1]
        q_inv = torch.cat([q_base[..., :1], -q_base[..., 1:]], dim=-1)
        q_offset = self.tgt_skeleton._qmul(glob_q, q_inv.repeat(1, glob_q.shape[1], 1, 1))

        return q_offset, q_base

    def __init__(self, dataloader, ckpt, cfg, device='cpu'):
        self.cfg = cfg
        self.device = device

        self.dataloader = dataloader
        self.tgt_skeleton = dataloader.skeleton

        d_pose_input = 0
        d_pose_output = 3
        # Filter out zero-length joints. Retain valid joints idx array for remapping
        self.valid_joints = []
        for i in range(len(self.tgt_skeleton.joint_lengths)):
            if self.tgt_skeleton.joint_lengths[i] > 1e-6:
                d_pose_input += 10
                d_pose_output += 4
                self.valid_joints.append(i)
        self.valid_joints = torch.tensor(self.valid_joints, dtype=torch.int64, device=device)

        encoder = CITLEncoder(self.cfg["CITL"]["d_model"], d_pose_input, self.cfg["CITL"]["n_layers"],
                              self.cfg["CITL"]["n_heads"])
        interm = CITLInterm(self.cfg["CITL"]["d_model"], self.cfg["CITL"]["n_layers"], self.cfg["CITL"]["n_heads"])
        decoder = CITLDecoder(self.cfg["CITL"]["d_model"], d_pose_output, self.cfg["CITL"]["n_layers"],
                              self.cfg["CITL"]["n_heads"])
        self.inpainter_model = CITL(encoder, interm, decoder, device=device)
        self.inpainter_model.load_state_dict(torch.load(ckpt))
        self.inpainter_model.eval()

        self.vis = Visualiser(framerate=24)

    def eval(self, n_tests, keyframe_indices, batch_size=-1):
        if batch_size == -1:
            batch_size = n_tests
        max_length = self.cfg["GENERAL"]["max_length"]

        n_vis = 1
        l2p = []
        l2q = []
        npss = []

        for ep in range(math.ceil(n_tests / batch_size)):
            # Generate motion sample

            with torch.no_grad():
                root_p, local_q = self.dataloader.load_samples(batch_size, max_length)
                real_global_p, real_global_q = self.tgt_skeleton.fk(root_p.to(self.device), local_q.to(self.device), local_q=True)

                # Filter out zero-length joints
                joint_indices = torch.reshape(self.valid_joints, (1, 1, -1, 1))
                real_global_q_ = real_global_q.transpose(0, 2).clone()
                real_global_q = torch.zeros(real_global_q_.shape, dtype=torch.float32, device=self.device)
                real_global_q[..., 0] = 1
                real_global_q[self.valid_joints] = real_global_q_[self.valid_joints]
                real_global_q = real_global_q.transpose(0, 2)
                real_global_q /= torch.linalg.vector_norm(real_global_q, dim=-1, keepdim=True)
                real_global_p_filter = torch.gather(real_global_p, 2, joint_indices.repeat(real_global_p.shape[0], real_global_p.shape[1], 1, 3))
                real_global_q_filter = torch.gather(real_global_q, 2, joint_indices.repeat(real_global_q.shape[0], real_global_q.shape[1], 1, 4))

                # quaternion offset & 3D roll-invariant base vector
                real_q_offset, q_base = self._augment_q_offset(real_global_q_filter)
                p_base = torch.reshape(self.tgt_skeleton.joint_tails[self.valid_joints], (1, 1, -1, 3))
                p_base = p_base / torch.linalg.vector_norm(p_base, dim=-1, keepdim=True)
                p_base = self.tgt_skeleton._qrot(p_base, q_base)
                p_base = p_base.repeat(1, max_length, 1, 1)

                # Sample keyframes
                keyframes = torch.zeros((batch_size, max_length), dtype=torch.bool, device=self.device)
                keyframes[..., 0] = True
                keyframes[..., -1] = True

                for b in range(batch_size):
                    keyframes[b][keyframe_indices[b]] = True

                # sanity check, filter out non-keyframe data from input
                keyframes_p = real_global_p_filter.clone()
                keyframes_p[keyframes == 0] = 0
                keyframes_q_offset = real_q_offset.clone()
                keyframes_q_offset[keyframes == 0] = 0
                keyframes_p_base = p_base.clone()
                keyframes_p_base[keyframes == 0] = 0

                p_mean = torch.sum(keyframes_p, dim=(1, 2)) / torch.sum(keyframes, dim=-1, keepdim=True)
                p_mean = torch.reshape(p_mean / keyframes_p.shape[2], (-1, 1, 1, 3))
                keyframes_p -= p_mean
                keyframes_p[keyframes == 0] = 0
                real_global_p -= p_mean
                real_global_p_filter -= p_mean

            model_input = torch.cat([keyframes_p, keyframes_p_base, keyframes_q_offset], dim=-1)
            pred = self.inpainter_model(torch.reshape(model_input, (batch_size, keyframes.shape[-1], -1)),
                                        keyframes)

            pred_p = pred[..., :3]
            pred_q_offset = pred[..., 3:]
            pred_q_offset = torch.reshape(pred_q_offset, (batch_size, max_length, -1, 4))
            pred_q_offset_norm = pred_q_offset / torch.clamp_min(
                torch.linalg.vector_norm(pred_q_offset, dim=-1, keepdim=True), 0.001)
            pred_global_q = self.tgt_skeleton._qmul(pred_q_offset_norm, q_base.repeat(1, pred.shape[1], 1, 1))

            # laplacian smooth
            pred_p[keyframes == 1] = real_global_p_filter[:, :, 0][keyframes == 1]
            pred_global_q[keyframes == 1] = real_global_q_filter[keyframes == 1]
            for _ in range(4):
                pred_p_ = pred_p.clone()
                pred_global_q_ = pred_global_q.clone()
                pred_p_[:, 1:-1] = (pred_p_[:, 2:] + pred_p_[:, :-2]) / 2
                pred_global_q_[:, 1:-1] = (pred_global_q_[:, 2:] + pred_global_q_[:, :-2]) / 2
                pred_p[keyframes == 0] = pred_p_[keyframes == 0]
                pred_global_q[keyframes == 0] = pred_global_q_[keyframes == 0]
                pred_global_q = pred_global_q / torch.linalg.vector_norm(pred_global_q, dim=-1, keepdim=True)

            pred_q_full = torch.zeros((pred.shape[0], pred.shape[1], self.tgt_skeleton.n_joints, 4), dtype=torch.float32, device=self.device)
            pred_q_full[..., 0] = 1
            indices = torch.reshape(self.valid_joints, (1, 1, -1, 1)).repeat(pred.shape[0], pred.shape[1], 1, 4)
            pred_q_full = torch.scatter(pred_q_full, 2, indices, pred_global_q)
            pred_global_q_norm = pred_q_full
            pred_global_p, pred_global_q = self.tgt_skeleton.fk(pred_p, pred_q_full, local_q=False, normalise=True)

            keyframe_mask = torch.reshape(keyframes, (*keyframes.shape, 1, 1)).repeat(1, 1, pred_global_p.shape[-2], 1)
            pred_global_p = torch.where(keyframe_mask.repeat(1, 1, 1, 3), real_global_p, pred_global_p)
            pred_q_full = torch.where(keyframe_mask.repeat(1, 1, 1, 4), real_global_q, pred_q_full)

            with torch.no_grad():
                l2p_ = torch.mean(torch.sqrt(torch.sum((pred_global_p - real_global_p) ** 2, dim=(-2, -1))))
                l2q_ = torch.mean(torch.sqrt(torch.sum((pred_q_full - real_global_q) ** 2, dim=(-2, -1))))
                npss_ = compute_npss(torch.reshape(torch.cat([real_global_p, real_global_q], dim=-1),
                                                   (batch_size, max_length, -1)).cpu().detach().numpy(),
                                     torch.reshape(torch.cat([pred_global_p, pred_q_full], dim=-1),
                                                   (batch_size, max_length, -1)).cpu().detach().numpy())
                l2p.append(torch.mean(l2p_).item())
                l2q.append(torch.mean(l2q_).item())
                npss.append(np.mean(npss_).item())

                real_heads = real_global_p[:n_vis].clone()
                real_tails = self.tgt_skeleton.get_tails(real_heads, real_global_q[:n_vis])
                pred_heads = pred_global_p[:n_vis].clone()
                pred_tails = self.tgt_skeleton.get_tails(pred_heads, pred_global_q_norm[:n_vis])

                vis_offset = torch.reshape(torch.arange(n_vis, device=self.device) * 2, (-1, 1, 1))
                real_heads[..., 0] += vis_offset
                real_tails[..., 0] += vis_offset
                pred_heads[..., 0] += vis_offset
                pred_tails[..., 0] += vis_offset

                heads = torch.cat([pred_heads], dim=-2)
                heads = torch.reshape(heads.transpose(0, 1), (max_length, -1, 3))
                tails = torch.cat([pred_tails], dim=-2)
                tails = torch.reshape(tails.transpose(0, 1), (max_length, -1, 3))

                z_min = torch.min(real_heads[..., 2])
                heads[..., 2] -= z_min
                tails[..., 2] -= z_min

                joint_color_palette = torch.tensor(
                    [pyqtgraph.hsvColor((x / 13.3) % 1.0, 1.0, 1.0, 1.0).getRgb() for x in range(40)],
                    device=device) / 255
                colors = joint_color_palette[:pred_heads.shape[-2]].clone().unsqueeze(0)
                colors = torch.repeat_interleave(colors, heads.shape[0], 0)
                self.vis.width = 10
                self.vis.update_data(heads, tails, colors)

        print("Interval %d - L2P: %.4f; L2Q: %.4f; NPSS: %.4f" % (N_INTERVAL, sum(l2p) / len(l2p), sum(l2q) / len(l2q), sum(npss) / len(npss)))


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    os.environ["CUDA_LAUNCH_BLOCKING"] = '0'

    if len(sys.argv) > 1:
        cfg_file = sys.argv[2]
    else:
        cfg_file = './config.json'
    with open(cfg_file, 'r') as f:
        cfg = json.load(f)

    device = torch.device("cuda")
    dataloaders = CMUDataLoader(cfg["GENERAL"]["cmu_path"], cfg["GENERAL"]["cmu_skeleton"], min_length=128, data_device=device, skel_device=device)

    test = InpainterEval(dataloaders, "./ckpt_citl/CITL40000.pt", cfg, device=device)

    for N_INTERVAL in range(5, 30):
        N_TESTS = 64
        keyframes = []
        for i in range(0, 128 - 1, N_INTERVAL):
            keyframes.append(i)
        keyframes.append(128 - 1)
        keyframes = torch.tensor(keyframes, dtype=torch.int64)
        keyframes = keyframes.unsqueeze(0).repeat(N_TESTS, 1)

        random.seed(1000)
        torch.random.manual_seed(1000)
        test.eval(N_TESTS, keyframes, batch_size=64)

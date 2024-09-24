import json
import os
import random
import sys

import torch
from matplotlib.font_manager import json_load
from torch import nn

from data.cmu import CMUDataLoader
from data.h36m import H36MDataLoader
from data.lafan import LaFANDataLoader
from data.skeleton import Skeleton
from models.inpainter_citl import CITL, CITLEncoder, CITLInterm, CITLDecoder
from models.pc_mrl import PointCloudDecoder
from viewport.visualiser import Visualiser


class InpainterTrainer:
    def __init__(self, dataloaders, tgt_skeleton: Skeleton, pointcloud_ckpt, cfg, device='cpu'):
        self.cfg = cfg
        self.device = device

        self.dataloaders = dataloaders
        self.tgt_skeleton = tgt_skeleton

        # input: 3D current position + 3D initial position + 4D quaternion per joint
        # output: 4D rotations + 3D root position
        d_pose_input = 0
        d_pose_output = 3
        # Filter out zero-length joints. Retain valid joints idx array for remapping
        self.valid_joints = []
        for i in range(len(tgt_skeleton.joint_lengths)):
            if tgt_skeleton.joint_lengths[i] > 1e-6:
                d_pose_input += 10
                d_pose_output += 4
                self.valid_joints.append(i)
        self.valid_joints = torch.tensor(self.valid_joints, dtype=torch.int64, device=device)

        self.pointcloud_model = PointCloudDecoder(self.cfg["PC-MRL"]["n_bgroups"], self.cfg["PC-MRL"]["d_model"], d_pose_output,
                                                  self.cfg["PC-MRL"]["temporal_n_heads"], self.cfg["PC-MRL"]["temporal_n_layers"],
                                                  self.cfg["PC-MRL"]["k_farthest_first"], self.cfg["GENERAL"]["max_length"]).to(self.device)
        self.pointcloud_model.load_state_dict(torch.load(pointcloud_ckpt))
        self.pointcloud_model.eval()

        encoder = CITLEncoder(self.cfg["CITL"]["d_model"], d_pose_input, self.cfg["CITL"]["n_layers"],
                              self.cfg["CITL"]["n_heads"])
        interm = CITLInterm(self.cfg["CITL"]["d_model"], self.cfg["CITL"]["n_layers"], self.cfg["CITL"]["n_heads"])
        decoder = CITLDecoder(self.cfg["CITL"]["d_model"], d_pose_output, self.cfg["CITL"]["n_layers"],
                              self.cfg["CITL"]["n_heads"])
        self.inpainter_model = CITL(encoder, interm, decoder, device=device)
        self.optimiser = torch.optim.Adam(self.inpainter_model.parameters(), lr=self.cfg["CITL"]["learning_rate"])

        self.vis = Visualiser(framerate=24)

    def save(self, ep):
        torch.save(self.inpainter_model.state_dict(), "./ckpt_citl/CITL%d.pt" % ep)

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

    def _augment_symmetrical(self, heads, tails, glob_q):
        root_p = heads[:, :, 0].clone()

        # PAIR AUGMENTATION
        # aug_joint_tails: [batch, joints, 3]
        aug_joint_tails = self.tgt_skeleton.joint_tails.clone().unsqueeze(0)
        aug_joint_tails = torch.repeat_interleave(aug_joint_tails, root_p.shape[0], dim=0)
        aug = torch.zeros(aug_joint_tails.shape, dtype=torch.float32, device=self.device)

        # aug_pairs: [2, batch, pairs, 3]
        aug_pairs = self.tgt_skeleton.pairs.transpose(0, 1)
        aug_pairs = torch.reshape(aug_pairs, (2, 1, -1, 1)).repeat(1, root_p.shape[0], 1, 3).to(self.device)
        aug_range = self.cfg["CITL"]["aug_range"]
        # aug_offset: [2, batch, pairs, 3]
        aug_offset = (torch.rand((1, root_p.shape[0], aug_pairs.shape[-2], 3), dtype=torch.float32, device=self.device) - 0.5) * aug_range
        aug_offset = torch.repeat_interleave(aug_offset, 2, dim=0).to(self.device)
        aug_offset[1, ..., 1] = -aug_offset[0, ..., 1]

        aug = torch.scatter(aug, 1, aug_pairs[0], aug_offset[0])
        aug = torch.scatter(aug, 1, aug_pairs[1], aug_offset[1])

        aug_joint_tails = aug_joint_tails + aug

        heads_aug = []
        q_aug = []
        no_roll = torch.zeros((glob_q.shape[0], glob_q.shape[1], 2), dtype=torch.float32, device=self.device)
        no_roll[..., 1] = 1

        for i, p_idx in enumerate(self.tgt_skeleton.joint_hierarchy):
            if p_idx == -1:
                head = root_p
            else:
                offset = self.tgt_skeleton.joint_offsets[i:i+1].unsqueeze(0)
                head = self.tgt_skeleton._qrot(offset, q_aug[p_idx])
                head = head + heads_aug[p_idx]

            kp_orig = tails[:, :, i] - heads[:, :, i]
            kp_mod = tails[:, :, i] - head
            q_mod = self.tgt_skeleton.batch_single_ik(kp_orig, kp_mod, no_roll)
            q_base = self.tgt_skeleton._qmul(q_mod, glob_q[:, :, i].contiguous())

            kp_aug = self.tgt_skeleton.q_to_keypoint(q_base, aug_joint_tails[:, i])
            q_mod = self.tgt_skeleton.batch_single_ik(kp_mod, kp_aug[..., :3], no_roll)
            q = self.tgt_skeleton._qmul(q_mod, q_base)

            heads_aug.append(head)
            q_aug.append(q)

        return torch.stack(heads_aug, dim=2), torch.stack(q_aug, dim=2)

    def train(self, epochs, start=1):
        max_length = self.cfg["GENERAL"]["max_length"]
        interp_keys_min = self.cfg["CITL"]["interp_keys_min"]
        interp_keys_max = self.cfg["CITL"]["interp_keys_max"]
        warmup_steps = self.cfg["CITL"]["warmup_steps"]
        n_points = self.cfg["PC-MRL"]["src_n_points"]
        std_cloud = self.cfg["PC-MRL"]["point_sampling_std"]
        # batch_size = (max_length - inb_seed_min - 1) // 2
        # input_batch_size = batch_size * 2
        batch_size = interp_keys_max - interp_keys_min + 1
        input_batch_size = batch_size * len(self.dataloaders)

        n_vis = 8

        for ep in range(start, epochs + start):
            # Generate motion sample
            self.optimiser.zero_grad()

            with torch.no_grad():
                if ep <= warmup_steps:
                    lr = self.cfg["CITL"]["learning_rate"] * ep / warmup_steps
                else:
                    lr = self.cfg["CITL"]["learning_rate"] * (1 - (ep - warmup_steps) / (epochs - warmup_steps))
                for g in self.optimiser.param_groups:
                    g["lr"] = lr

                samples = []
                end_joints = []
                for dataloader in self.dataloaders:
                    root_p, q = dataloader.load_samples(batch_size, 128)
                    global_p, global_q = dataloader.skeleton.fk(root_p.to(self.device), q.to(self.device), local_q=True)
                    samples.append(dataloader.skeleton.generate_pointcloud(global_p, global_q, n_points=n_points, std=std_cloud))
                    end_joints.append(global_p.transpose(0, 2).clone()[dataloader.skeleton.end_joints].transpose(0, 2))
                samples = torch.cat(samples, dim=0)
                end_joints = torch.cat(end_joints, dim=0)

                means = torch.mean(samples[..., :3], dim=(1, 2), keepdim=True)
                samples[..., :3] -= means
                end_joints -= means

                pointcloud_eval = self.pointcloud_model(samples)
                root_p = pointcloud_eval[..., :3]
                q = torch.reshape(pointcloud_eval[..., 3:], (pointcloud_eval.shape[0], pointcloud_eval.shape[1], -1, 4))
                q /= torch.linalg.vector_norm(q, dim=-1, keepdim=True)
                sign = torch.gt(torch.linalg.vector_norm(q[:, 1:] - q[:, :-1], dim=-1, keepdim=True),
                                torch.linalg.vector_norm(q[:, 1:] + q[:, :-1], dim=-1, keepdim=True)).int()
                sign = (torch.cumsum(sign, dim=1) % 2) * -2 + 1
                q = torch.cat([q[:, :1], q[:, 1:] * sign], dim=1)

                q_full = torch.zeros((q.shape[0], q.shape[1], self.tgt_skeleton.n_joints, 4), dtype=torch.float32, device=self.device)
                q_full[..., 3] = 1
                joint_indices = torch.reshape(self.valid_joints, (1, 1, -1, 1))
                q_full = torch.scatter(q_full, 2, joint_indices.repeat(q.shape[0], q.shape[1], 1, 4), q)
                real_global_p, real_global_q = self.tgt_skeleton.fk(root_p, q_full, local_q=False)
                real_global_p = real_global_p[:, :max_length]
                real_global_q = real_global_q[:, :max_length]

                # Augment variation
                tails = self.tgt_skeleton.get_tails(real_global_p, real_global_q)
                real_global_p, real_global_q = self._augment_symmetrical(real_global_p, tails, real_global_q)

                # Filter out zero-length joints
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
                keyframes = torch.zeros((input_batch_size, max_length), dtype=torch.bool, device=self.device)
                keyframes[..., 0] = True
                keyframes[..., -1] = True

                for d in range(len(self.dataloaders)):
                    for b in range(batch_size):
                        perm = torch.randperm(max_length - 2)[:interp_keys_min + b - 2] + 1
                        keyframes[d * batch_size + b][perm] = True

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
                model_input = torch.cat([keyframes_p, keyframes_p_base, keyframes_q_offset], dim=-1)

            pred = self.inpainter_model(torch.reshape(model_input, (input_batch_size, max_length, -1)), keyframes)
            pred_p = pred[..., :3]

            pred_q_offset = pred[..., 3:]
            pred_q_offset = torch.reshape(pred_q_offset, (input_batch_size, max_length, -1, 4))
            pred_q_offset_norm = pred_q_offset / torch.clamp_min(torch.linalg.vector_norm(pred_q_offset, dim=-1, keepdim=True), 0.001)
            pred_global_q = self.tgt_skeleton._qmul(pred_q_offset_norm, q_base.repeat(1, pred.shape[1], 1, 1))

            pred_q_full = torch.zeros((pred.shape[0], pred.shape[1], self.tgt_skeleton.n_joints, 4), dtype=torch.float32, device=self.device)
            pred_q_full[..., 0] = 1
            indices = torch.reshape(self.valid_joints, (1, 1, -1, 1)).repeat(pred.shape[0], pred.shape[1], 1, 4)
            pred_q_full = torch.scatter(pred_q_full, 2, indices, pred_global_q)
            pred_global_q_norm = pred_q_full
            pred_global_p, _ = self.tgt_skeleton.fk(pred_p.clone().detach(), pred_global_q_norm, local_q=False, normalise=False)

            keyframe_mask = torch.reshape(keyframes, (*keyframes.shape, 1, 1)).repeat(1, 1, pred_global_p.shape[-2], 1)
            pred_global_p = torch.where(keyframe_mask.repeat(1, 1, 1, 3), real_global_p, pred_global_p)
            pred_global_q_norm = torch.where(keyframe_mask.repeat(1, 1, 1, 4), real_global_q, pred_global_q_norm)

            loss_root_p = torch.mean(torch.sum(torch.abs(pred_p - real_global_p[:, :, 0]), dim=-1))
            loss_global_p = torch.mean(torch.sum(torch.abs(pred_global_p - real_global_p), dim=-1))
            loss_global_q = torch.mean(torch.sum(torch.abs(pred_q_offset - real_q_offset), dim=-1))
            loss_global_q_norm = torch.mean(torch.sum(torch.abs(pred_global_q_norm - real_global_q), dim=-1))

            loss = (loss_root_p + loss_global_q) + (loss_global_p + loss_global_q_norm) * 0.1
            loss.backward()
            self.optimiser.step()

            with torch.no_grad():
                l2p = torch.mean(torch.sqrt(torch.sum((pred_global_p - real_global_p) ** 2, dim=(-2, -1))))
                l2q = torch.mean(torch.sqrt(torch.sum((pred_global_q_norm - real_global_q) ** 2, dim=(-2, -1))))

                if ep % 50 == 1:
                    real_heads = real_global_p[:n_vis].clone()
                    real_tails = self.tgt_skeleton.get_tails(real_heads, real_global_q[:n_vis])
                    pred_heads = pred_global_p[:n_vis].clone()
                    pred_tails = self.tgt_skeleton.get_tails(pred_heads, pred_global_q_norm[:n_vis])

                    vis_offset = torch.reshape(torch.arange(n_vis, device=self.device) * 2, (-1, 1, 1))
                    real_heads[..., 0] += vis_offset
                    real_tails[..., 0] += vis_offset
                    pred_heads[..., 0] += vis_offset
                    pred_tails[..., 0] += vis_offset

                    real_heads[..., 2] += 1
                    real_tails[..., 2] += 1
                    pred_heads[..., 2] -= 1
                    pred_tails[..., 2] -= 1

                    heads = torch.cat([real_heads, pred_heads], dim=-2)
                    heads = torch.reshape(heads.transpose(0, 1), (max_length, -1, 3))
                    tails = torch.cat([real_tails, pred_tails], dim=-2)
                    tails = torch.reshape(tails.transpose(0, 1), (max_length, -1, 3))

                    self.vis.update_data(heads, tails)

                print("Epoch %d - loss_root_p: %.4f; loss_global_p: %.4f; loss_global_q: %.4f; loss_global_q_norm: %.4f; L2P: %.4f; L2Q: %.4f" %
                      (ep, loss_root_p.item(), loss_global_p.item(), loss_global_q.item(), loss_global_q_norm.item(), l2p.item(), l2q.item()))

                with open("./ckpt_citl/log.txt", 'a') as log:
                    log.write("Epoch %d - loss_root_p: %.4f; loss_global_p: %.4f; loss_global_q: %.4f; loss_global_q_norm: %.4f; L2P: %.4f; L2Q: %.4f" %
                              (ep, loss_root_p.item(), loss_global_p.item(), loss_global_q.item(), loss_global_q_norm.item(), l2p.item(), l2q.item()))

            if ep % 5000 == 0:
                self.save(ep)


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
    dataloaders = [H36MDataLoader(cfg["GENERAL"]["human3.6m_path"], cfg["GENERAL"]["human3.6m_skeleton"], min_length=128, data_device=device, skel_device=device),
                   LaFANDataLoader(cfg["GENERAL"]["lafan1_path"], cfg["GENERAL"]["lafan1_skeleton"], min_length=128, data_device=device, skel_device=device)]

    skeleton = CMUDataLoader.setup_skeleton(cfg["GENERAL"]["cmu_skeleton"], device)
    trainer = InpainterTrainer(dataloaders, skeleton, cfg["CITL"]["ckpt_pc-mrl"], cfg, device=device)

    trainer.train(50000)

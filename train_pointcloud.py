import json
import sys

import torch
from torch import nn
import os

from data.h36m import H36MDataLoader
from data.lafan import LaFANDataLoader
from data.cmu import CMUDataLoader
from models.pc_mrl import PointCloudDecoder
from models.utils.knn_loss import KNNLoss
from viewport.visualiser import Visualiser


class PointCloudTrainer:
    def __init__(self, dataloaders, tgt_skeleton, cfg, device='cpu'):
        self.cfg = cfg
        self.device = device

        self.dataloaders = dataloaders
        self.tgt_skeleton = tgt_skeleton

        # input: 3D point cloud
        # output: 4D rotations + 3D root position
        d_pose_output = 3
        # Filter out zero-length joints. Retain valid joints idx array for remapping
        self.valid_joints = []
        for i in range(len(tgt_skeleton.joint_lengths)):
            if tgt_skeleton.joint_lengths[i] > 1e-6:
                # IK + complex roll representation
                d_pose_output += 4
                self.valid_joints.append(i)
        self.valid_joints = torch.tensor(self.valid_joints, dtype=torch.int64, device=device)

        self.model = PointCloudDecoder(self.cfg["PC-MRL"]["n_bgroups"], self.cfg["PC-MRL"]["d_model"], d_pose_output,
                                       self.cfg["PC-MRL"]["temporal_n_heads"], self.cfg["PC-MRL"]["temporal_n_layers"],
                                       self.cfg["PC-MRL"]["k_farthest_first"], self.cfg["GENERAL"]["max_length"]).to(self.device)
        self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["PC-MRL"]["learning_rate"], amsgrad=True)
        # self.model = torch.compile(self.model, mode="reduce-overhead")
        self.knn_loss = KNNLoss(self.cfg["PC-MRL"]["n_bgroups"], self.cfg["PC-MRL"]["knn_loss_k"])

        self.vis = Visualiser(framerate=24, width=5)

    def save(self, ep):
        torch.save(self.model.state_dict(), "./ckpt_pc-mrl/PCMRL%d.pt" % ep)

    def train(self, epochs):
        batch_size = self.cfg["PC-MRL"]["batch_size"]
        max_length = self.cfg["GENERAL"]["max_length"]
        warmup_steps = self.cfg["PC-MRL"]["warmup_steps"]
        src_n_points = self.cfg["PC-MRL"]["src_n_points"]
        tgt_n_points = self.cfg["PC-MRL"]["tgt_n_points"]
        std_cloud = self.cfg["PC-MRL"]["point_sampling_std"]

        n_vis = 4

        for ep in range(1, epochs + 1):
            lr = self.cfg["PC-MRL"]["learning_rate"] * min(ep / warmup_steps, (epochs - ep) / (epochs - warmup_steps))
            alpha_l = min(ep / warmup_steps, (epochs - ep) / (epochs - warmup_steps))
            for g in self.optimiser.param_groups:
                g["lr"] = lr
            with torch.no_grad():
                samples = []
                end_joints = []
                real_global_p = []
                real_global_q = []
                for dataloader in self.dataloaders:
                    root_p, q = dataloader.load_samples(batch_size, max_length)
                    global_p, global_q = dataloader.skeleton.fk(root_p.to(self.device), q.to(self.device), local_q=True)
                    samples.append(dataloader.skeleton.generate_pointcloud(global_p, global_q, n_points=src_n_points, std=std_cloud))
                    end_joints.append(global_p.transpose(0, 2).clone()[dataloader.skeleton.end_joints].transpose(0, 2))
                    real_global_p.append(global_p)
                    real_global_q.append(global_q)
                samples = torch.cat(samples, dim=0)
                end_joints = torch.cat(end_joints, dim=0)

                means = torch.mean(samples[..., :3], dim=(1, 2), keepdim=True)
                samples[..., :3] -= means
                end_joints -= means
                for d in range(len(self.dataloaders)):
                    real_global_p[d] -= means[d * batch_size:(d+1) * batch_size]
            self.optimiser.zero_grad()

            pred = self.model(samples)
            pred_p = pred[..., :3]
            pred_q = torch.reshape(pred[..., 3:], (pred.shape[0], pred.shape[1], -1, 4))
            pred_q_full = torch.zeros((pred.shape[0], pred.shape[1], self.tgt_skeleton.n_joints, 4), dtype=torch.float32, device=self.device)
            pred_q_full[..., 0] = 1
            indices = torch.reshape(self.valid_joints, (1, 1, -1, 1)).repeat(pred_q.shape[0], pred_q.shape[1], 1, 4)
            pred_q_full = torch.scatter(pred_q_full, 2, indices, pred_q)
            pred_global_p, pred_global_q = self.tgt_skeleton.fk(pred_p, pred_q_full, local_q=False)

            pred_pointcloud = self.tgt_skeleton.generate_pointcloud(pred_global_p, pred_global_q, n_points=tgt_n_points, std=std_cloud)
            knn_loss_g = 0
            knn_loss_l = 0
            knn_loss_v = 0
            for i in range(len(self.dataloaders)):
                knn_loss_g_, knn_loss_l_, knn_loss_v_ = self.knn_loss(samples[i * batch_size:(i+1) * batch_size],
                                                                      pred_pointcloud[i * batch_size:(i+1) * batch_size])
                knn_loss_g = knn_loss_g + knn_loss_g_
                knn_loss_l = knn_loss_l + knn_loss_l_
                knn_loss_v = knn_loss_v + knn_loss_v_

            pred_end_joints = pred_global_p.transpose(0, 2)[self.tgt_skeleton.end_joints].transpose(0, 2)
            end_loss = torch.mean(torch.abs(end_joints - pred_end_joints))
            unit_loss = torch.mean(torch.abs(1 - torch.linalg.vector_norm(pred_q, dim=-1)))

            loss = 1.0 * knn_loss_g + alpha_l * knn_loss_l + knn_loss_v + 0.0 * end_loss + 0.01 * unit_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimiser.step()
            print("Epoch %d - knn_loss_g: %.4f; knn_loss_l: %.4f; knn_loss_v: %.4f; end_loss: %.4f; unit_loss: %.4f" %
                  (ep, knn_loss_g.item(), knn_loss_l.item(), knn_loss_v.item(), end_loss.item(), unit_loss.item()))

            with open("./ckpt_pc-mrl/log.txt", 'a') as log:
                log.write("Epoch %d - knn_loss_g: %.4f; knn_loss_l: %.4f; knn_loss_v: %.4f; end_loss: %.4f; unit_loss: %.4f\n" %
                          (ep, knn_loss_g.item(), knn_loss_l.item(), knn_loss_v.item(), end_loss.item(), unit_loss.item()))

            with torch.no_grad():
                if ep % 50 == 1:
                    heads = []
                    tails = []
                    for d, dataloader in enumerate(self.dataloaders):
                        for i in range(n_vis):
                            real_heads = real_global_p[d][i]
                            real_tails = dataloader.skeleton.get_tails(real_heads, real_global_q[d][i])
                            pred_heads = pred_global_p[d * batch_size + i]
                            pred_tails = self.tgt_skeleton.get_tails(pred_heads, pred_global_q[d * batch_size + i])
                            real_heads[..., 2] += 1
                            real_tails[..., 2] += 1
                            pred_heads[..., 2] -= 1
                            pred_tails[..., 2] -= 1
                            sample_heads = torch.cat([real_heads, pred_heads], dim=-2)
                            sample_tails = torch.cat([real_tails, pred_tails], dim=-2)
                            sample_heads[..., 0] += 2 * (d * n_vis + i)
                            sample_tails[..., 0] += 2 * (d * n_vis + i)

                            heads.append(sample_heads)
                            tails.append(sample_tails)

                    heads = torch.cat(heads, dim=-2)
                    tails = torch.cat(tails, dim=-2)

                    self.vis.update_data(heads, tails)

            if ep % 5000 == 0:
                self.save(ep)


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)

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
    trainer = PointCloudTrainer(dataloaders, skeleton, cfg=cfg, device=device)
    trainer.train(50000)

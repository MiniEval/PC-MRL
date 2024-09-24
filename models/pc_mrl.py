import torch
from torch import nn

from models.modules import TransformerBlock, PositionalEmbedding, PointTransformer


class PointCloudDecoder(nn.Module):
    def __init__(self, n_bgroups, d_model, d_pose, n_heads, n_layers, k=8, max_length=64):
        super(PointCloudDecoder, self).__init__()
        self.cloud_encoder = PointTransformer(n_bgroups, d_model, k=k)

        self.pe = PositionalEmbedding(16, max_length, sinusoidal=True)
        self.motion_enc = nn.ModuleList([TransformerBlock(d_model * 16 + 16, n_heads, norm_first=False, activation="gelu",
                                                          norm_method="rmsnorm", dropout=0.0) for _ in range(n_layers)])
        self.motion_out = nn.Linear(d_model * 16 + 16, d_pose)

    def forward(self, p):
        # p: [batch, frames, points, pos]
        batches = p.shape[0]
        frames = p.shape[1]
        points = p.shape[2]

        x = torch.reshape(p, (batches * frames, points, -1))
        x = self.cloud_encoder(x)
        x = torch.reshape(x, (batches, frames, -1))
        x = x.transpose(0, 1)

        pe = self.pe(torch.zeros(frames, batches, 16, dtype=torch.float32, device=x.device))
        x = torch.cat([x, pe], dim=-1)
        for block in self.motion_enc:
            x = block(x)
        x = x.transpose(0, 1)
        x = self.motion_out(x)

        return x

from torch import nn
import torch

from models.modules import PositionalEmbedding, TransformerBlock, RMSNorm


class CITLEncoder(nn.Module):
    def __init__(self, d_model, d_pose, n_layers, n_heads):
        super(CITLEncoder, self).__init__()

        self.embed_in = nn.Linear(d_pose, d_model)
        self.encoder = nn.ModuleList([TransformerBlock(d_model, n_heads, activation="gelu", norm_method="rmsnorm",
                                                       relative_pe=True, dropout=0)
                                      for _ in range(n_layers)])

    def forward(self, x, keyframes):
        # x: [batches, frames, data]
        # keyframes: [batches, frames]
        h = self.embed_in(x)
        h = h.transpose(0, 1)

        for block in self.encoder:
            h = block(h, emb_mask=(keyframes == 0), norm_mask=keyframes)

        h = h.transpose(0, 1)

        # out: [batches, keyframes, enc]
        return h


class CITLInterm(nn.Module):
    def __init__(self, d_model, n_layers, n_heads):
        super(CITLInterm, self).__init__()

        self.encoder = nn.ModuleList([TransformerBlock(d_model, n_heads, activation="gelu", norm_method="rmsnorm",
                                                       relative_pe=True, dropout=0)
                                      for _ in range(n_layers)])

    def forward(self, keyframes, emb):
        # keyframes: [batches, frames]
        # emb: [batches, keyframes, data]
        h = torch.zeros(keyframes.shape[1], keyframes.shape[0], emb.shape[-1], device=emb.device)

        emb = emb.transpose(0, 1)

        for block in self.encoder:
            h = block(h, emb=emb, emb_mask=(keyframes == 0), norm_mask=(keyframes == 0))

        h = h.transpose(0, 1)

        return h


class CITLDecoder(nn.Module):
    def __init__(self, d_model, d_pose, n_layers, n_heads):
        super(CITLDecoder, self).__init__()

        self.key_ff_in = nn.Linear(d_model, d_model * 4)
        self.key_ff_relu = nn.GELU()
        self.key_ff_out = nn.Linear(d_model * 4, d_model)
        self.key_ff_norm = RMSNorm(d_model)

        self.interm_ff_in = nn.Linear(d_model, d_model * 4)
        self.interm_ff_relu = nn.GELU()
        self.interm_ff_out = nn.Linear(d_model * 4, d_model)
        self.interm_ff_norm = RMSNorm(d_model)

        self.decoder = nn.ModuleList([TransformerBlock(d_model, n_heads, activation="gelu", norm_method="rmsnorm",
                                                       relative_pe=True, dropout=0)
                                      for _ in range(n_layers)])

        self.output = nn.Linear(d_model, d_pose)

    def forward(self, key_x, interm_x, keyframes):
        # x: [batches, frames, data]
        # keyframes: [batches, frames]

        key_x = self.key_ff_in(key_x)
        key_x = self.key_ff_relu(key_x)
        key_x = self.key_ff_out(key_x)
        key_x = self.key_ff_norm(key_x)

        interm_x = self.interm_ff_in(interm_x)
        interm_x = self.interm_ff_relu(interm_x)
        interm_x = self.interm_ff_out(interm_x)
        interm_x = self.interm_ff_norm(interm_x)
        manifold = torch.where(keyframes.unsqueeze(-1).repeat(1, 1, key_x.shape[-1]), key_x, interm_x)

        manifold = manifold.transpose(0, 1)

        for block in self.decoder:
            manifold = block(manifold)

        manifold = manifold.transpose(0, 1)

        out = self.output(manifold)

        return out


class CITL(nn.Module):
    def __init__(self, encoder, interm, decoder, device="cpu"):
        super(CITL, self).__init__()
        self.device = device

        self.encoder = encoder.to(device)
        self.interm = interm.to(device)
        self.decoder = decoder.to(device)

    def forward(self, poses, keyframes):
        # poses: [batches, frames, data], frames dimension is max length, can be incomplete with keyframe mask
        # keyframes: [batches, frames], bool mask
        enc = self.encoder(poses, keyframes)
        interm = self.interm(keyframes, enc)
        dec = self.decoder(enc, interm, keyframes)

        return dec

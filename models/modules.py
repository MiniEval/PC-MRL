import math

import numpy as np
import torch
from torch import nn
import pytorch3d.ops


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, norm_first=False, activation="relu", norm_method="layernorm", relative_pe=False, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads

        if norm_method == "rmsnorm":
            self.attn_norm = RMSNorm(d_model)
        else:
            self.attn_norm = nn.LayerNorm(d_model)

        if relative_pe:
            self.attn = RelativeAttention(d_model, n_heads, dropout=dropout)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        if norm_method == "rmsnorm":
            self.ff_norm = RMSNorm(d_model)
        else:
            self.ff_norm = nn.LayerNorm(d_model)

        self.ff_in = nn.Linear(d_model, d_model * 4)
        if activation == "leakyrelu":
            self.ff_relu = nn.LeakyReLU()
        elif activation == "gelu":
            self.ff_relu = nn.GELU()
        else:
            self.ff_relu = nn.ReLU()
        self.ff_out = nn.Linear(d_model * 4, d_model)

        self.norm_first = norm_first

    def forward(self, x, emb=None, emb_mask=None, norm_mask=None, temp_pe=0):
        # [length, batch, data]
        # mask: [batch, keyframes]

        if self.norm_first:
            h = self.apply_norm(x, self.attn_norm, norm_mask)
        else:
            h = x

        if emb_mask is not None:
            attn_mask = torch.reshape(emb_mask, (emb_mask.shape[0], 1, 1, emb_mask.shape[-1]))
            attn_mask = attn_mask.repeat(1, self.n_heads, emb_mask.shape[-1], 1)
            attn_mask *= ~torch.eye(emb_mask.shape[-1], dtype=torch.bool, device=x.device)
            attn_mask = torch.reshape(attn_mask, (-1, emb_mask.shape[-1], emb_mask.shape[-1]))
        else:
            attn_mask = None

        h_pe = h + temp_pe
        if emb is None:
            # attn, _ = self.attn(h, h, h, need_weights=False, key_padding_mask=emb_mask)
            attn, _ = self.attn(h_pe, h_pe, h_pe, need_weights=False, attn_mask=attn_mask)
        else:
            emb_pe = emb + temp_pe
            # attn, _ = self.attn(h, emb, emb, need_weights=False, key_padding_mask=emb_mask)
            attn, _ = self.attn(h_pe, emb_pe, emb_pe, need_weights=False, attn_mask=attn_mask)

        attn = self.dropout(attn)
        h = h + attn

        if not self.norm_first:
            h = self.apply_norm(h, self.attn_norm, norm_mask)

        if self.norm_first:
            ff = self.apply_norm(h, self.ff_norm, norm_mask)
        else:
            ff = h
        ff = self.ff_in(ff)
        ff = self.ff_relu(ff)
        ff = self.ff_out(ff)
        ff = self.dropout(ff)

        h = h + ff
        if not self.norm_first:
            h = self.apply_norm(h, self.ff_norm, norm_mask)

        return h

    def apply_norm(self, x, norm_layer, mask=None):
        if mask is not None:
            out = x.clone()
            h = x[mask.transpose(0, 1)]
            h = norm_layer(h)
            out[mask.transpose(0, 1)] = h
            return out
        else:
            return norm_layer(x)


class PointAttention(nn.Module):
    def __init__(self, d_model, k=8):
        super(PointAttention, self).__init__()

        self.k = k
        self.d_model = d_model

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

    def forward(self, query, p_pos):
        # query: [points, batch, x]
        # p_pos: [batch, points, 3]
        # p_diff = p_pos.unsqueeze(2) - p_pos.unsqueeze(1)

        q = self.fc_q(query).transpose(0, 1)
        k = self.fc_k(query).transpose(0, 1)
        v = self.fc_v(query).transpose(0, 1)
        # le: [batch, points, points, x]
        # le_k = self.le_k(p_diff)
        # le_v = self.le_v(p_diff)

        # [batch, points, x] @ [batch, x, points] = [batch, points, points]
        attn1 = torch.matmul(q, k.transpose(-2, -1))
        # [batch, points, 1, x] @ [batch, points, x, points] = [batch, points, points]
        # attn2 = torch.matmul(q.unsqueeze(-2), le_k.transpose(-2, -1))[..., 0, :]

        # attn = (attn1 + attn2) / math.sqrt(self.d_model)
        attn = attn1 / math.sqrt(self.d_model)
        attn = torch.softmax(attn, dim=-1)

        # [batch, points, points] @ [batch, points, x] = [batch, points, x]
        val1 = torch.matmul(attn, v)
        # [batch, points, 1, points] @ [batch, points, points, x] = [batch, points, x]
        # val2 = torch.matmul(attn.unsqueeze(-2), le_v)[..., 0, :]

        # val = val1 + val2
        val = val1

        o = self.fc_o(val)
        o = o.transpose(0, 1)

        return o


class PointPool(nn.Module):
    def __init__(self, d_in, d_out, stride=2, k=8):
        super(PointPool, self).__init__()
        self.k = k
        self.stride = stride

        self.stride = stride
        self.mlp_in = nn.Linear(3 + d_in, d_out, bias=False)
        # self.norm = nn.InstanceNorm1d(d_out)
        self.relu = nn.LeakyReLU()
        self.mlp_out = nn.Linear(d_out, d_out, bias=False)

    def forward(self, x, p_pos):
        # x: [batch, points dim, d_in]
        # p_pos: [batch, points dim, 3]

        n_out = p_pos.shape[1] // self.stride
        # fp: [batch, points dim / stride, x], fp_idx: [batch, points dim / stride]
        fp_p_pos, _ = pytorch3d.ops.sample_farthest_points(p_pos, K=n_out)
        _, knn_idx, _ = pytorch3d.ops.knn_points(fp_p_pos, p_pos, K=self.k)
        # knn_x: [batch, points dim / stride, k, x]
        knn_x = pytorch3d.ops.knn_gather(x, knn_idx)

        h = torch.cat([fp_p_pos.unsqueeze(-2).expand(-1, -1, self.k, -1), knn_x], dim=-1)
        h = torch.reshape(h, (-1, h.shape[-2], h.shape[-1]))
        h = self.mlp_in(h)
        # h = self.norm(h).transpose(-2, -1)
        h = self.relu(h)
        h = torch.reshape(h, (knn_x.shape[0], knn_x.shape[1], self.k, -1))
        # max pool
        h = torch.amax(h, dim=-2)
        h = self.mlp_out(h)

        # out: [batch, points dim / stride, d_out]
        return h, fp_p_pos


class PointTransformerBlock(nn.Module):
    def __init__(self, d_model, stride=2, k=8):
        super(PointTransformerBlock, self).__init__()
        self.stride = stride

        self.attn_norm = RMSNorm(d_model)
        self.attn = PointAttention(d_model, k)

        self.ff_norm = RMSNorm(d_model)
        self.ff_in = nn.Linear(d_model, d_model * 4)
        self.ff_relu = nn.LeakyReLU()
        self.ff_out = nn.Linear(d_model * 4, d_model)

        self.pool = PointPool(d_model, d_model * 2, stride=stride, k=k)

    def forward(self, x, p_pos):
        # x: [points, batch, x]
        # p_pos: [batch, points, 3]
        h = self.attn_norm(x)
        attn = self.attn(h, p_pos)

        ff = self.ff_norm(attn)
        ff = self.ff_in(ff)
        ff = self.ff_relu(ff)
        ff = self.ff_out(ff)
        h = h + ff

        h, fp_p_pos = self.pool(h.transpose(0, 1), p_pos)
        h = h.transpose(0, 1)

        return h, fp_p_pos


class PointTransformer(nn.Module):
    def __init__(self, n_bgroups, d_model, k=8):
        super(PointTransformer, self).__init__()
        self.p_in = nn.Linear(3 + n_bgroups, d_model)

        self.attn_256 = PointTransformerBlock(d_model, stride=4, k=min(k, 256))
        self.attn_64 = PointTransformerBlock(d_model * 2, stride=4, k=min(k, 64))
        self.attn_16 = PointTransformerBlock(d_model * 4, stride=4, k=min(k, 16))
        self.attn_4 = PointTransformerBlock(d_model * 8, stride=4, k=min(k, 4))

        self.mlp_in = nn.Linear(d_model * 16, d_model * 16)
        self.mlp_relu = nn.LeakyReLU()
        self.mlp_out = nn.Linear(d_model * 16, d_model * 16)

    def forward(self, p):
        # p: [batch (* frames), points, 3 + n_bgroups]'
        p_pos = p[..., :3]
        x = self.p_in(p)
        x = x.transpose(0, 1)

        x, p_pos = self.attn_256(x, p_pos)
        x, p_pos = self.attn_64(x, p_pos)
        x, p_pos = self.attn_16(x, p_pos)
        x, _ = self.attn_4(x, p_pos)

        x = self.mlp_in(x[0])
        x_ = self.mlp_relu(x)
        x_ = self.mlp_out(x_)
        x = x + x_

        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, dimension, length, scale=1.0, sinusoidal=True):
        super(PositionalEmbedding, self).__init__()
        self.dim = dimension

        if sinusoidal:
            # Compute the positional encodings once in log space.
            pe = torch.zeros(length, dimension, dtype=torch.float32)
            pe.require_grad = False

            position = torch.arange(0, length).float().unsqueeze(1)

            div_term = (torch.arange(0, dimension, 2).float() * -(math.log(10000.0) / dimension)).exp()
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            pe *= scale

            self.register_buffer('pe', pe)

        else:
            pe = torch.zeros(length, dimension, dtype=torch.float32)
            nn.init.xavier_normal_(pe)
            self.register_parameter('pe', nn.Parameter(pe))

    def forward(self, x, indices=None):
        # x: [length, batch, data]
        # indices: [batch, length]

        if indices is None:
            x = x + self.pe[:x.shape[0]].unsqueeze(1)
        else:
            x = x + self.pe[indices].transpose(0, 1)

        return x


# RELATIVE ATTENTION from https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/


class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.zeros_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings


class RelativeAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, max_length=50):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = max_length

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, need_weights=False, attn_mask=None):
        # qkv: [length, batch, data]
        # attn_mask: [batch * heads, length, length] - bool
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size * self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if attn_mask is not None:
            attn_mask = torch.reshape(attn_mask, attn.shape)
            attn = attn.masked_fill(attn_mask, -1e10)

        attn = self.dropout(torch.softmax(attn, dim=-1))

        # attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size * self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)
        x = x.transpose(0, 1)

        # x = [query len, batch size, hid dim]

        return x, None

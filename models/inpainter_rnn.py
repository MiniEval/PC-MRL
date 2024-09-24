import torch
from torch import nn


class PLU(nn.Module):
    def __init__(self, alpha = 0.1, c = 1.0):
        super(PLU, self).__init__()
        self.alpha = alpha
        self.c = c
        self.relu = nn.ReLU()

    def forward(self, x):
        o1 = self.alpha * (x + self.c) - self.c
        o2 = self.alpha * (x - self.c) + self.c
        o3 = x - self.relu(x - o2)
        o4 = self.relu(o1 - o3) + o3
        return o4


class RNNEncoder(nn.Module):
    def __init__(self, state_dim=128, offset_dim=128, target_dim=128, hidden_dim=512, out_dim=256, max_tta=64):
        super(RNNEncoder, self).__init__()
        self.plu = PLU()

        self.state_lin0 = nn.Linear(state_dim, hidden_dim)
        self.state_lin1 = nn.Linear(hidden_dim, out_dim)

        self.offset_lin0 = nn.Linear(offset_dim, hidden_dim)
        self.offset_lin1 = nn.Linear(hidden_dim, out_dim)

        self.target_lin0 = nn.Linear(target_dim, hidden_dim)
        self.target_lin1 = nn.Linear(hidden_dim, out_dim)

        ztta = torch.zeros(max_tta, out_dim, dtype=torch.float32, requires_grad=False)
        basis = torch.pow(torch.full((out_dim,), 10000, dtype=torch.float32),
                          torch.arange(out_dim, dtype=torch.float32) / -out_dim).unsqueeze(0)
        ztta[:, 0::2] = torch.sin(torch.arange(max_tta, dtype=torch.float32).unsqueeze(-1) * basis[:, 0::2])
        ztta[:, 1::2] = torch.cos(torch.arange(max_tta, dtype=torch.float32).unsqueeze(-1) * basis[:, 1::2])
        self.register_buffer("ztta", ztta)

    def forward(self, state, offset, target, tta, noise=None):
        # state & offset: [batches, frames, data]
        # target: [batches, data]
        # tta: [frames]
        state_enc = self.state_lin0(state)
        state_enc = self.plu(state_enc)
        state_enc = self.state_lin1(state_enc)
        state_enc = state_enc + self.ztta[tta].unsqueeze(0)

        offset_enc = self.offset_lin0(offset)
        offset_enc = self.plu(offset_enc)
        offset_enc = self.offset_lin1(offset_enc)
        offset_enc = offset_enc + self.ztta[tta].unsqueeze(0)

        target_enc = self.target_lin0(target)
        target_enc = self.plu(target_enc)
        target_enc = self.target_lin1(target_enc)

        target_enc = target_enc.unsqueeze(1) + self.ztta[tta].unsqueeze(0)

        ot_enc = torch.cat([offset_enc, target_enc], dim=-1)

        if noise is not None:
            if tta >= 30:
                ztarget = 1
            elif tta < 5:
                ztarget = 0
            else:
                ztarget = (tta - 5) / 25

            ot_enc = ot_enc + noise * ztarget

        return torch.cat([state_enc, ot_enc], dim=-1)


class RNNDecoder(nn.Module):
    def __init__(self, latent_dim=768, hidden_dim=512, out_dim=256):
        super(RNNDecoder, self).__init__()
        self.plu = PLU()
        self.decoder_lin0 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.decoder_lin2 = nn.Linear(hidden_dim // 2, out_dim)

    def forward(self, x):
        x = self.decoder_lin0(x)
        x = self.plu(x)
        x = self.decoder_lin1(x)
        x = self.plu(x)
        x = self.decoder_lin2(x)
        return x


class RNNInpainter(nn.Module):
    def __init__(self, state_dim, offset_dim, target_dim, pose_dim, hidden_dim=512, latent_dim=256, max_tta=64):
        super(RNNInpainter, self).__init__()

        self.latent_dim = latent_dim

        self.enc = RNNEncoder(state_dim, offset_dim, target_dim, hidden_dim, latent_dim, max_tta)
        self.lstm = nn.LSTM(input_size=latent_dim * 3, hidden_size=latent_dim * 3, num_layers=1, batch_first=True)
        self.dec = RNNDecoder(latent_dim * 3, hidden_dim, pose_dim)

    def forward(self, seed_p, seed_q, target_p, target_q, frames, noise=False):
        # seed_p: [b, l, 3]
        # seed_q: [b, l, j, 4]
        # target_p: [b, 3]
        # target_q: [b, j, 4]

        # DATA PREP
        batches = seed_p.shape[0]
        seed_frames = seed_p.shape[1]

        target_p_ = torch.reshape(target_p, (batches, 1, 3))
        target_q_ = torch.reshape(target_q, (batches, 1, -1, 4))

        seed_root_v = torch.zeros(seed_p.shape, dtype=torch.float32, device=seed_p.device)
        seed_root_v[:, 1:] = seed_p[:, 1:] - seed_p[:, :-1]
        seed_offset_p = target_p_ - seed_p
        seed_offset_q = target_q_ - seed_q

        tta = frames - torch.arange(1, frames, device=seed_p.device)

        seed_state = torch.cat([seed_root_v, torch.reshape(seed_q, (batches, seed_frames, -1))], dim=-1)
        seed_offset = torch.cat([seed_offset_p, torch.reshape(seed_offset_q, (batches, seed_frames, -1))], dim=-1)
        target = torch.reshape(target_q, (batches, -1))
        seed_tta = tta[:seed_frames]

        if noise:
            target_noise = torch.normal(torch.zeros((batches, 1, self.latent_dim * 2), device=seed_p.device), 0.5)
        else:
            target_noise = torch.zeros((batches, 1, self.latent_dim * 2), device=seed_p.device)

        # SEED DATA PROCESS
        lstm_hc = None
        for f in range(seed_frames):
            x = self.enc(seed_state[:, f:f+1], seed_offset[:, f:f+1], target, seed_tta[f:f+1], target_noise)
            x, lstm_hc = self.lstm(x, lstm_hc)
            x = self.dec(x)

        # AUTO-REGRESSIVE EVAL
        out_pose = []

        last_p = seed_p[:, -1:]
        last_q = torch.reshape(seed_q[:, -1:], (batches, 1, -1))

        for f in range(seed_frames, frames - 1):
            root_v = x[:, -1:, :3]
            root_p = last_p + root_v
            q = x[:, -1:, 3:] + last_q
            q = torch.reshape(q, (batches, 1, -1, 4))
            q = q / torch.clamp_min(torch.linalg.vector_norm(q, dim=-1, keepdim=True), 1e-6)

            last_p = root_p
            last_q = torch.reshape(q, (batches, 1, -1))
            out_pose.append(torch.cat([last_p, last_q], dim=-1))

            if f == frames - 2:
                break

            offset_p = target_p_ - root_p
            offset_q = target_q_ - q

            state = torch.cat([root_v, torch.reshape(q, (batches, 1, -1))], dim=-1)
            offset = torch.cat([offset_p, torch.reshape(offset_q, (batches, 1, -1))], dim=-1)

            x = self.enc(state, offset, target, tta[f:f+1], target_noise)
            x, lstm_hc = self.lstm(x, lstm_hc)
            x = self.dec(x)

        out_pose = torch.cat(out_pose, dim=1)

        return out_pose, lstm_hc


class RNNDiscriminator(nn.Module):
    def __init__(self, pose_dim, hidden_dim=512, short_size=2, long_size=10):
        super(RNNDiscriminator, self).__init__()
        self.relu = nn.ReLU()

        self.conv_short1 = nn.Conv1d(pose_dim, hidden_dim, kernel_size=short_size)
        self.conv_short2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=short_size)
        self.out_short = nn.Linear(hidden_dim // 2, 1)
        self.conv_long1 = nn.Conv1d(pose_dim, hidden_dim, kernel_size=long_size)
        self.conv_long2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=long_size)
        self.out_long = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        # x: [batch, length, data]
        short = self.conv_short1(x.transpose(1, 2))
        short = self.relu(short)
        short = self.conv_short2(short)
        short = self.relu(short)
        short = self.out_short(short.transpose(1, 2))
        short = torch.mean(short, dim=(1, 2))

        long = self.conv_long1(x.transpose(1, 2))
        long = self.relu(long)
        long = self.conv_long2(long)
        long = self.relu(long)
        long = self.out_long(long.transpose(1, 2))
        long = torch.mean(long, dim=(1, 2))

        return short, long

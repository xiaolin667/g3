 from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.shape[1], :])


class ShapeTokenUnit(nn.Module):
    def __init__(self, window: int, hidden_dim: int):
        super().__init__()
        conv_dim = hidden_dim // 2
        stat_dim = hidden_dim // 4
        self.window = window
        self.raw_conv = nn.Sequential(
            nn.Conv1d(1, conv_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.diff_conv = nn.Sequential(
            nn.Conv1d(1, conv_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.stat = nn.Sequential(nn.Linear(2, stat_dim), nn.GELU())
        self.proj = nn.Sequential(nn.Linear(conv_dim * 2 + stat_dim, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, length = x.shape
        patches = x.unfold(dimension=2, size=self.window, step=self.window).squeeze(1)
        n = patches.shape[1]
        patch_flat = patches.reshape(b * n, 1, self.window)
        global_mean = x.mean(dim=2, keepdim=True)
        global_std = x.std(dim=2, keepdim=True).clamp_min(1e-5)
        norm = ((patch_flat - global_mean.repeat_interleave(n, dim=0)) / global_std.repeat_interleave(n, dim=0))
        diff = F.pad(norm.diff(dim=2), (0, 1))
        raw_emb = self.raw_conv(norm).squeeze(-1)
        diff_emb = self.diff_conv(diff).squeeze(-1)
        stats = torch.stack([patches.mean(dim=-1), patches.std(dim=-1)], dim=-1).reshape(b * n, 2)
        out = self.proj(torch.cat([raw_emb, diff_emb, self.stat(stats)], dim=-1))
        return out.reshape(b, n, -1)


class SameConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = self.kernel_size - 1
        left = total // 2
        right = total - left
        return self.conv(F.pad(x, (left, right)))


class InceptionModule1d(nn.Module):
    """UniShape-style two-path Inception block scaled to the configured hidden dim."""

    def __init__(self, in_channels: int, branch_channels: int):
        super().__init__()
        kernels = [39, 19, 9]
        self.bottleneck = nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False)
        self.convs = nn.ModuleList(
            [SameConv1d(branch_channels, branch_channels, kernel_size=k) for k in kernels]
        )
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
        )
        self.bn = nn.InstanceNorm1d(branch_channels * 4)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(x)
        branches = [conv(bottleneck) for conv in self.convs]
        branches.append(self.pool_conv(x))
        return self.act(self.bn(torch.cat(branches, dim=1)))


class HierarchicalEncoder(nn.Module):
    def __init__(
        self,
        input_length: int = 512,
        hidden_dim: int = 64,
        windows: list[int] | tuple[int, ...] = (64, 32, 16, 8, 4),
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_length = input_length
        self.hidden_dim = hidden_dim
        self.windows = list(windows)
        self.units = nn.ModuleList([ShapeTokenUnit(w, hidden_dim) for w in self.windows])
        branch_channels = max(1, hidden_dim // 4)
        self.inception_token = nn.Sequential(
            InceptionModule1d(hidden_dim, branch_channels),
            InceptionModule1d(hidden_dim, branch_channels),
        )
        self.scale_attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        max_tokens = input_length // min(self.windows) + 1
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_tokens, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        self.final_attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))

    def _attn_pool(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.inception_token(tokens.transpose(1, 2)).transpose(1, 2)
        logits = self.scale_attention(tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return pooled, weights

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_token = None
        last_shape_tokens = None
        for unit in self.units:
            tokens = unit(x)
            if cls_token is not None:
                tokens_for_pool = torch.cat([cls_token.unsqueeze(1), tokens], dim=1)
            else:
                tokens_for_pool = tokens
            cls_token, _ = self._attn_pool(tokens_for_pool)
            last_shape_tokens = tokens
        assert last_shape_tokens is not None
        seq = torch.cat([cls_token.unsqueeze(1), last_shape_tokens], dim=1)
        seq = self.pos_encoder(seq)
        refined = self.transformer(seq)
        cls_refined = refined[:, 0]
        shape_refined = refined[:, 1:]
        shape_logits = self.final_attention(shape_refined).squeeze(-1)
        shape_weights = torch.softmax(shape_logits, dim=1)
        return cls_refined, shape_refined, shape_weights


class IntervalPredictionModule(nn.Module):
    def __init__(self, hidden_dim: int, intervals: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, intervals),
        )

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.net(cls)


class G3Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        pre_cfg = config["pretrain"]
        hidden = int(model_cfg["hidden_dim"])
        self.encoder = HierarchicalEncoder(
            input_length=int(model_cfg["input_length"]),
            hidden_dim=hidden,
            windows=model_cfg["windows"],
            transformer_layers=int(model_cfg["transformer_layers"]),
            transformer_heads=int(model_cfg["transformer_heads"]),
            dropout=float(model_cfg["dropout"]),
        )
        self.ipm = IntervalPredictionModule(hidden, int(pre_cfg["intervals"]))
        self.readout = model_cfg.get("readout", "concat")
        head_in = hidden * 2 if self.readout == "concat" else hidden
        self.regression_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.Linear(hidden, 1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def regression_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cls, shapes, attn = self.encode(x)
        shape_pool = torch.sum(shapes * attn.unsqueeze(-1), dim=1)
        if self.readout == "class":
            feat = cls
        elif self.readout == "shape":
            feat = shape_pool
        else:
            feat = torch.cat([cls, shape_pool], dim=-1)
        return feat, cls, shapes, attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, _, _, _ = self.regression_features(x)
        return self.regression_head(feat).squeeze(-1)


class PrototypeBank(nn.Module):
    def __init__(self, intervals: int, hidden_dim: int):
        super().__init__()
        proto = torch.randn(intervals, hidden_dim)
        self.register_buffer("prototypes", F.normalize(proto, dim=1))

    @torch.no_grad()
    def ema_update(self, features: torch.Tensor, labels: torch.Tensor, beta: float) -> None:
        for label in labels.unique():
            mask = labels == label
            if not torch.any(mask):
                continue
            center = features[mask].mean(dim=0)
            center = F.normalize(center, dim=0)
            idx = int(label.item())
            self.prototypes[idx] = F.normalize(beta * self.prototypes[idx] + (1.0 - beta) * center, dim=0)


def augment(x: torch.Tensor, jitter_sigma: float, scale_low: float, scale_high: float) -> torch.Tensor:
    sample_std = x.std(dim=2, keepdim=True).clamp_min(1e-5)
    noise = torch.randn_like(x) * jitter_sigma * sample_std
    scale = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(scale_low, scale_high)
    return x * scale + noise


def interval_contrastive_loss(
    reps: torch.Tensor,
    probs: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    intervals = prototypes.shape[0]
    reps = F.normalize(reps, dim=-1)
    protos = F.normalize(prototypes, dim=-1)
    sim = torch.matmul(reps, protos.t()) / temperature
    j = torch.arange(intervals, device=reps.device).view(1, -1)
    weights = 1.0 - (labels.view(-1, 1) - j).abs().float() / max(intervals - 1, 1)
    weights = weights.clamp_min(0.0)
    exp_sim = torch.exp(sim)
    denom = (weights * exp_sim).sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_prob = sim - torch.log(denom)
    return -(probs * log_prob).sum(dim=1).mean()


def top_shape_loss(
    shapes: torch.Tensor,
    attn: torch.Tensor,
    probs: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
    top_ratio: float,
) -> torch.Tensor:
    b, n, d = shapes.shape
    k = max(1, int(n * top_ratio))
    idx = torch.topk(attn, k=k, dim=1).indices
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, d)
    top = torch.gather(shapes, 1, gather_idx).reshape(b * k, d)
    probs_rep = probs.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)
    labels_rep = labels.unsqueeze(1).expand(-1, k).reshape(b * k)
    return interval_contrastive_loss(top, probs_rep, labels_rep, prototypes, temperature)

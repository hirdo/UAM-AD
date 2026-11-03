"""
OrEdge (arXiv 2608.00309, theamrzaki/MicroService_Twin_Original; main model = FREQ_DOMAIN "FITS_Legendre") in plain
torch. Structure follows src/model.py + src/inner_models/FITS_Legendre.py of the authors' repository:
  per service node and modality (metric, log, trace): linear embedding + positional encoding -> projection to 10
  channels -> ONE shared FITS-Legendre temporal encoder (RevIN, Legendre basis of degree 5, NormLin frequency mixing,
  low-pass, linear frequency up-sampler) -> linear attention across the metric/log tokens -> projection back and
  dense layers that reconstruct the raw inputs; error = lambda_time * squared error + lambda_aux * squared error of
  the Legendre coefficients (Legendre-style loss); the error vector of the last step of every node is fed to a small
  classifier ("show": D -> 128 -> 2) that says whether the node is the culprit.
Training loss (semi-supervised, 3 node states: normal / labelled anomalous / unknown):
  loss = (1 - para) * weighted BCE(show, labelled nodes) + para * rec_loss, para = max(1 / (epoch + 1), 0.01),
  rec_loss = mean error of normal nodes + relu(mean error - 1) of labelled anomalous nodes
             + label_weight * mean error of unknown nodes.
Differences from the released code:
  * Legendre coefficients are taken over the time axis (the released legendre_encode reshapes [B, T, C] to
    [B * C, T] without a permute, which mixes time and channel values);
  * AdaBelief (adabelief-pytorch) is re-implemented below (rectified, decoupled weight decay) to avoid a new dependency;
  * only the Legendre/LPF/linear-attention configuration of the authors' run script is kept.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import legendre


class AdaBelief(torch.optim.Optimizer):
    """adabelief-pytorch defaults: eps 1e-16, rectify, decoupled weight decay, degenerated to SGD momentum."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["step"], st["m"], st["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                p.mul_(1 - g["lr"] * g["weight_decay"])
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                r = p.grad - st["m"]
                st["v"].mul_(b2).addcmul_(r, r, value=1 - b2)
                rho_inf = 2 / (1 - b2) - 1
                rho_t = rho_inf - 2 * t * b2 ** t / (1 - b2 ** t)
                if rho_t > 4:
                    rect = math.sqrt((1 - b2 ** t) * (rho_t - 4) * (rho_t - 2) * rho_inf
                                     / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                    denom = st["v"].add(g["eps"]).sqrt().add_(g["eps"])
                    p.addcdiv_(st["m"], denom, value=-g["lr"] * rect / (1 - b1 ** t))
                else:
                    p.add_(st["m"], alpha=-g["lr"] / (1 - b1 ** t))


class Embed(nn.Module):
    def __init__(self, raw_dim, dim, max_len=1000):
        super().__init__()
        self.linear = nn.Linear(raw_dim, dim)
        pos = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        pe = torch.zeros((1, max_len, dim))
        pe[:, :, 0::2], pe[:, :, 1::2] = torch.sin(pos), torch.cos(pos)
        self.register_buffer("pe", pe.unsqueeze(2))

    def forward(self, x):  # [B, T, N, raw]
        x = self.linear(x)
        return x + self.pe[:, :x.shape[1]]


class LinearAttention(nn.Module):
    def __init__(self, enc_in, dim):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(enc_in, dim, bias=False) for _ in range(3))
        self.out = nn.Linear(dim, enc_in)
        self.norm = nn.LayerNorm(enc_in)

    def forward(self, z):  # [B*N, M, D]
        q, k, v = F.elu(self.q(z)) + 1, F.elu(self.k(z)) + 1, self.v(z)
        kv = torch.einsum("bmd,bme->bde", k, v)
        denom = torch.einsum("bmd,bd->bm", q, k.sum(1)).unsqueeze(-1) + 1e-6
        return self.norm(self.out(torch.einsum("bmd,bde->bme", q, kv) / denom) + z)


class FitsLegendre(nn.Module):
    """FITS_Legendre.Model for pred_len = 0 (window reconstruction), basis 'legendre', filter 'LPF', NormLin."""

    def __init__(self, seq_len, degree=5):
        super().__init__()
        self.degree = degree
        self.normlin_W = nn.Parameter(torch.empty(degree, degree))
        self.up = nn.Linear(degree, degree)
        t = np.linspace(-1, 1, seq_len)
        self.register_buffer("basis", torch.tensor(np.array([legendre(i)(t) for i in range(degree)]),
                                                   dtype=torch.float32))  # [degree, T]

    def forward(self, x):  # [N, T, C]
        mean, std = x.mean(1, keepdim=True), x.std(1, keepdim=True) + 1e-5
        x = (x - mean) / std
        spec = x.transpose(1, 2) @ self.basis.T                          # [N, C, degree]
        w = F.softplus(self.normlin_W)
        spec = spec @ (w / (w.sum(1, keepdim=True) + 1e-8)).T            # NormLin
        spec = torch.cat([spec[..., :self.degree // 2], torch.zeros_like(spec[..., self.degree // 2:])], -1)  # LPF
        return (self.up(spec) @ self.basis).transpose(1, 2) * std + mean  # [N, T, C]


class OrEdge(nn.Module):
    def __init__(self, raw_node, log_len, raw_edge, window=10, feature_node=8, feature_log=32, feature_edge=8,
                 enc_in=10, degree=5, linear_attn_dim=10, rec_lambda=1.0, auxi_lambda=0.2, label_weight=1e-2):
        super().__init__()
        self.rec_lambda, self.auxi_lambda, self.label_weight = rec_lambda, auxi_lambda, label_weight
        self.node_emb, self.log_emb, self.edge_emb = (Embed(raw_node, feature_node), Embed(log_len, feature_log),
                                                      Embed(raw_edge, feature_edge))
        self.dense_node, self.dense_log, self.dense_edge = (nn.Linear(feature_node, raw_node),
                                                            nn.Linear(feature_log, log_len),
                                                            nn.Linear(feature_edge, raw_edge))
        self.fits = FitsLegendre(window, degree)
        self.attn = LinearAttention(enc_in, linear_attn_dim)
        self.proj = nn.ModuleDict({"node": nn.Linear(feature_node, enc_in), "log": nn.Linear(feature_log, enc_in),
                                   "edge": nn.Linear(feature_edge, enc_in)})
        self.proj_out = nn.ModuleDict({"node": nn.Linear(enc_in, feature_node), "log": nn.Linear(enc_in, feature_log),
                                       "edge": nn.Linear(enc_in, feature_edge)})
        self.show = nn.Sequential(nn.Linear(raw_node + log_len + raw_edge, 128), nn.LeakyReLU(inplace=True),
                                  nn.Linear(128, 2))

    def legendre_error(self, pred, true):  # [B, T, C] -> [B, C] mean squared error of the degree-5 coefficients
        b = self.fits.basis                                                # [degree, T]
        coef = lambda z: torch.einsum("dt,btc->bcd", b, z) / z.shape[1] * 2 * (
            (2 * torch.arange(b.shape[0], device=z.device) + 1) / 2)
        return (coef(pred) - coef(true)).square().mean(-1)

    def reconstruction_error(self, node, log, edge):
        """-> error tensor [B, T, N, raw_node + log_len + raw_edge]."""
        B, T, N, _ = node.shape
        seq = lambda z: z.permute(0, 2, 1, 3).reshape(B * N, T, -1)          # [B*N, T, F]
        h = {k: self.fits(self.proj[k](seq(e(x)))) for k, e, x in
             (("node", self.node_emb, node), ("log", self.log_emb, log), ("edge", self.edge_emb, edge))}
        tok = self.attn(torch.stack([h["node"].mean(1), h["log"].mean(1)], 1))
        h["node"], h["log"] = (tok[:, i].unsqueeze(1).expand(-1, T, -1) for i in (0, 1))
        errs = []
        for k, dense, x in (("node", self.dense_node, node), ("log", self.dense_log, log), ("edge", self.dense_edge, edge)):
            r = dense(self.proj_out[k](h[k])).reshape(B, N, T, -1).permute(0, 2, 1, 3)     # [B, T, N, raw]
            lt = (r - x).square()
            ll = self.legendre_error(r.flatten(2), x.flatten(2)).reshape(B, 1, N, -1).expand(-1, T, -1, -1)
            errs.append(self.rec_lambda * lt + self.auxi_lambda * ll)
        return torch.cat(errs, -1)

    def forward(self, node, log, edge):
        """-> logits [B, N, 2] of 'node is the culprit' and the node reconstruction error [B, N, D]."""
        rec = self.reconstruction_error(node, log, edge)[:, -1]
        return self.show(rec), rec

    def loss(self, logits, rec, state, para, class_weight):
        """state [B, N] in {0 normal, 1 labelled anomalous, 2 unknown}."""
        node_rec = rec.mean(-1)
        rec_loss = ((node_rec * (state == 0)).sum() + (F.relu(node_rec - 1.0) * (state == 1)).sum()
                    + (self.label_weight * node_rec * (state == 2)).sum()) / state.numel()
        keep = state != 2
        if keep.any():
            target = F.one_hot(state[keep], 2).float()
            cls_loss = F.binary_cross_entropy_with_logits(logits[keep], target, weight=class_weight)
        else:
            cls_loss = logits.sum() * 0
        return (1 - para) * cls_loss + para * rec_loss

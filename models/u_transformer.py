# --- filename: models/u_transformer.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============= Rotary Embeddings (RoPE) ============= #
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len: Optional[int] = None):
        # x: (..., seq_len, dim)
        if seq_len is None:
            seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq, dim]
        cos = emb.cos()[None, None, :, :]  # [1,1,seq,dim]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def apply_rotary(q, k, cos, sin):
    # q, k: [B, H, T, D]
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


# ============= RMSNorm ============= #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # norm = x.norm(2, dim=-1, keepdim=True)
        # rms = norm * (1.0 / math.sqrt(x.size(-1)))
        # x_norm = x / (rms + self.eps)
        # return self.weight * x_norm
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


# ============= SwiGLU ============= #
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_mult=4.0):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ============= Multi-Query Attention (GQA-ish) ============= #
class MultiQueryAttention(nn.Module):
    def __init__(self, dim, n_heads, head_dim=None, kv_heads=1, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim or (dim // n_heads)
        assert self.head_dim * n_heads == dim
        # Projections
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, return_attn=False):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,D]
        k = self.wk(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)  # [B,KV,T,D]
        v = self.wv(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q, T)
        # broadcast cos/sin to kv heads by repeat
        q, k = apply_rotary(q, k, cos, sin)

        if past_kv is not None:
            pk, pv = past_kv  # [B,KV,Tpast,D]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        # Expand k,v from KV heads to full heads
        if self.kv_heads == 1:
            k = k.expand(B, self.n_heads, k.size(2), self.head_dim)
            v = v.expand(B, self.n_heads, v.size(2), self.head_dim)
        else:
            # tile to n_heads
            repeat = self.n_heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # causal mask
        if attn_mask is not None:
            att = att + attn_mask  # mask should be additive (-inf on masked positions)
        weights = torch.softmax(att, dim=-1)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = torch.matmul(att, v)  # [B,H,T,D]
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        y = self.wo(y)
        if return_attn:
            return y, weights  # return attention map for debugging
        return y, (k, v)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, kv_heads=1, mlp_mult=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiQueryAttention(dim, n_heads, kv_heads=kv_heads, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_mult=mlp_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, past_kv=None):
        a, new_kv = self.attn(self.norm1(x), attn_mask=attn_mask, past_kv=past_kv)
        x = x + a
        m = self.mlp(self.norm2(x))
        x = x + self.dropout(m)
        return x, new_kv


@dataclass
class UTransformerConfig:
    vocab_size: int
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    kv_heads: int = 1
    mlp_mult: float = 4.0
    dropout: float = 0.0
    max_seq_len: int = 2048
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = -1  # NEW


class UTransformerLM(nn.Module):
    def __init__(self, cfg: UTransformerConfig):
        super().__init__()
        self.cfg = cfg
        # If we use an external PAD id at vocab_size, extend embedding/output by 1
        emb_vocab = cfg.vocab_size + (1 if cfg.pad_id == cfg.vocab_size else 0)
        self.tok_emb = nn.Embedding(
            emb_vocab, cfg.d_model,
            padding_idx=(cfg.pad_id if cfg.pad_id >= 0 else None)
        )
        # Output layer must match embedding vocab for correct class count
        self.lm_head = nn.Linear(cfg.d_model, emb_vocab, bias=False)
        # Tie weights if shapes match
        if self.lm_head.weight.shape == self.tok_emb.weight.shape:
            self.lm_head.weight = self.tok_emb.weight
        self.layers = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, kv_heads=cfg.kv_heads, mlp_mult=cfg.mlp_mult, dropout=cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm_f = RMSNorm(cfg.d_model)

        # Causal mask precomputed up to max_seq_len
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", (1.0 - mask) * -1e4)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, segment_ids: Optional[torch.Tensor] = None):
        # idx: [B,T]
        B, T = idx.size()
        if T > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.tok_emb(idx)
        attn_mask = self.causal_mask[:, :, :T, :T]
        # Block attention to PAD columns (keys)
        pad_id = getattr(self.cfg, "pad_id", -1)
        if pad_id is not None and pad_id >= 0:
            # shape [B,1,1,T] → broadcast across heads and query positions
            key_pad = (idx == pad_id).unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask + key_pad * (-1e4)  # fp16-safe large negative
        # Block attention across segments if provided (segment_ids: [B,T])
        if segment_ids is not None:
            # allow attention only within same segment id
            # seg_eq: [B,1,T,T] True when in same segment; False when crossing segments
            seg = segment_ids[:, :T]
            seg_eq = (seg.unsqueeze(1).unsqueeze(3) == seg.unsqueeze(1).unsqueeze(2))
            seg_mask = (~seg_eq).to(attn_mask.dtype) * (-1e4)
            attn_mask = attn_mask + seg_mask
        # (attention debug handled in training loop)

        new_kv = []
        kv = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, kv_i = layer(x, attn_mask=attn_mask, past_kv=kv[i])
            new_kv.append(kv_i)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # Use provided targets so caller can mask pad positions with -100
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                targets[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def score(self, idx: torch.Tensor) -> torch.Tensor:
        """Return average negative log-likelihood (nats) for each sequence in batch."""
        self.eval()
        B, T = idx.size()
        logits, _ = self.forward(idx)
        logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
        tgt = idx[:, 1:]
        nll = -logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        # mask out padding if any
        pad_id = self.cfg.pad_id if self.cfg.pad_id >= 0 else None
        if pad_id is not None:
            lengths = (tgt != pad_id).sum(dim=1).clamp(min=1)
        else:
            lengths = (tgt != 0).sum(dim=1).clamp(min=1)
        return (nll.sum(dim=1) / lengths)







"""
Qenya-5B: собственная LLM-архитектура (decoder-only transformer).

Фишки профессионального уровня (всё реализовано здесь, без внешних API):
  - RMSNorm, SwiGLU MLP, RoPE, QK-norm, tied embeddings
  - Multi-Query Attention + KV-cache для быстрого инференса
  - Мультимодальность: vision-патчи вливаются в последовательность через
    токен <|image|> ... <|image_end|> (proxy-tuning LLaVA-стиля)
  - Конфигурация QENYA_5B — 5 млрд параметров; на слабых машинах
    автоматически используется QENYA_TINY (тот же код, меньше слоёв).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QenyaConfig:
    vocab_size: int = 4096
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 2          # MQA/GQA
    ffn_mult: float = 8/3
    max_seq_len: int = 2048
    rope_theta: float = 1_000_000.0
    rms_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    image_tokens: int = 64         # сколько vision-токенов даёт картинка
    img_start_id: int = 3
    img_end_id: int = 4
    im_end_id: int = 1

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads


# Конфиг «взрослой» модели на 5B параметров (для обучения на кластере/GPU-ферме)
QENYA_5B = dict(vocab_size=65536, hidden_size=4096, num_layers=40, num_heads=32,
                num_kv_heads=8, ffn_mult=8/3, max_seq_len=8192)


def count_params(cfg: QenyaConfig) -> int:
    """Точный аналитический подсчёт параметров (без аллокации памяти)."""
    D = cfg.hidden_size
    Dh = D // cfg.num_heads
    ffn = int(D * cfg.ffn_mult / 2 / 8) * 8
    per_layer = (D * cfg.num_heads * Dh          # q
                 + 2 * D * cfg.num_kv_heads * Dh # k,v
                 + cfg.num_heads * Dh * D        # o
                 + 3 * D * ffn                   # swiglu
                 + 2 * Dh                        # qk-norm
                 + 2 * D)                        # rmsnorms
    emb = cfg.vocab_size * D                     # tied embeddings
    return emb + per_layer * cfg.num_layers
# Практичная конфигурация того же кода для локальной машины
QENYA_TINY = dict(vocab_size=4096, hidden_size=512, num_layers=8, num_heads=8,
                  num_kv_heads=2, ffn_mult=8/3, max_seq_len=1024)


def pick_config(prefer: str = "auto") -> QenyaConfig:
    """auto: если хватает RAM и есть CUDA — полная 5B, иначе tiny."""
    if prefer == "5b":
        return QenyaConfig(**QENYA_5B)
    if prefer == "tiny":
        return QenyaConfig(**QENYA_TINY)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().available / 1e9
    except Exception:
        ram_gb = 30.0 if torch.cuda.is_available() else 2.0
    if torch.cuda.is_available() and ram_gb >= 14:
        return QenyaConfig(**QENYA_5B)
    return QenyaConfig(**QENYA_TINY)


# --------------------------------------------------------------- блоки ------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def precompute_freqs_cis(dim: int, end: int, theta: float, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary(xq, xk, freqs_cis):
    # xq/xk: [B,H,S,Dh]
    cq = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    ck = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    f = freqs_cis.unsqueeze(0).unsqueeze(0)  # [1,1,S,Dh/2]
    oq = torch.view_as_real(cq * f).flatten(-2).type_as(xq)
    ok = torch.view_as_real(ck * f).flatten(-2).type_as(xk)
    return oq, ok


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: QenyaConfig):
        super().__init__()
        self.cfg = cfg
        self.nh, self.nkv = cfg.num_heads, cfg.num_kv_heads
        self.hd = cfg.head_dim
        self.q = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=False)
        self.k = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)

    def forward(self, x, freqs_cis, kv_cache=None, mask=None):
        B, S, D = x.shape
        qh = self.q_norm(self.q(x).view(B, S, self.nh, self.hd)).transpose(1, 2)
        kh = self.k_norm(self.k(x).view(B, S, self.nkv, self.hd)).transpose(1, 2)
        vh = self.v(x).view(B, S, self.nkv, self.hd).transpose(1, 2)

        cur_pos = kv_cache["pos"] if kv_cache is not None else 0
        qh_r, kh_r = apply_rotary(qh, kh, freqs_cis[cur_pos:cur_pos + S])

        if kv_cache is not None:
            if kv_cache["k"] is not None:
                kh_r = torch.cat([kv_cache["k"], kh_r], dim=2)
                vh = torch.cat([kv_cache["v"], vh], dim=2)
            kv_cache["k"] = kh_r
            kv_cache["v"] = vh
        T = kh_r.shape[2]

        # repeat_kv для GQA
        rep = self.nh // self.nkv
        kfull = kh_r.repeat_interleave(rep, dim=1) if rep > 1 else kh_r
        vfull = vh.repeat_interleave(rep, dim=1) if rep > 1 else vh

        out = F.scaled_dot_product_attention(qh_r, kfull, vfull,
                                             attn_mask=mask, is_causal=(mask is None))
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: QenyaConfig):
        super().__init__()
        hid = int(cfg.hidden_size * cfg.ffn_mult / 2 / 8) * 8
        self.gate = nn.Linear(cfg.hidden_size, hid, bias=False)
        self.up = nn.Linear(cfg.hidden_size, hid, bias=False)
        self.down = nn.Linear(hid, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: QenyaConfig):
        super().__init__()
        self.att_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.att = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, freqs_cis, kv_cache=None, mask=None):
        x = x + self.att(self.att_norm(x), freqs_cis, kv_cache, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class QenyaLLM(nn.Module):
    def __init__(self, cfg: QenyaConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.register_buffer("freqs_cis",
                             precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta),
                             persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self) -> int:
        seen = set()
        tot = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p)); tot += p.numel()
        return tot

    def forward(self, ids, labels=None, extra_embed=None, extra_mask=None,
                kv_cache=None):
        """ids: [B,S]. extra_embed: [B,S,D] — vision-векторы (заменяют эмбеддинги
        картинок там, где extra_mask==True)."""
        x = self.tok_emb(ids)
        if extra_embed is not None:
            x = torch.where(extra_mask.unsqueeze(-1), extra_embed.to(x.dtype), x)
        S = x.shape[1]
        if kv_cache is None:
            causal = torch.full((S, S), float("-inf"), dtype=x.dtype)
            causal = torch.triu(causal, diagonal=1)
            mask = causal.to(x.device)
            caches = [None] * len(self.layers)
        else:
            first = kv_cache[0]
            past = first["k"].shape[2] if (S == 1 and first["k"] is not None) else 0
            if past == 0:
                mask = None                      # prefill: обычный causal
            else:
                mask = torch.zeros(1, 1, S, past + S, device=x.device, dtype=x.dtype)
                mask[..., past + 1:] = float("-inf")  # decode-step
            caches = kv_cache
        for li, layer in enumerate(self.layers):
            x = layer(x, self.freqs_cis, caches[li], mask)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def new_cache(self):
        return [{"k": None, "v": None, "pos": 0} for _ in range(self.cfg.num_layers)]

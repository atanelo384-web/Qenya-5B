"""
Qenya-Vision: собственный ViT-энкодер изображений + проектор в пространство LLM.

Схема (как в профессиональных мультимодальных моделях):
  image -> patchify (16x16) -> N_layers трансформера (RoPE-free, learned pos-emb)
        -> mean-pool/patch-projection MLP -> [image_tokens, D_llm]
В LLM картинки вставляются как <|image|> v_1..v_K <|image_end|>, где v_i —
проекционные векторы, замещающие эмбеддинги токенов (extra_embed).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

IMG_SIZE = 128          # входной размер (128/16 = 8x8 = 64 патча)
PATCH = 16


class PatchEmbed(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=PATCH, stride=PATCH)
        self.n_patches = (IMG_SIZE // PATCH) ** 2
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, dim) * 0.02)

    def forward(self, x):                    # x: [B,3,S,S]
        t = self.proj(x)                     # [B,dim,h,w]
        t = t.flatten(2).transpose(1, 2)     # [B,np,dim]
        return t + self.pos


class VisionBlock(nn.Module):
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return x


class QenyaVision(nn.Module):
    """ViT + MLP-проектор к hidden_size языковой модели."""

    def __init__(self, vdim=256, layers=6, heads=8, llm_dim=512, image_tokens=64):
        super().__init__()
        self.patch = PatchEmbed(vdim)
        self.blocks = nn.ModuleList([VisionBlock(vdim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(vdim)
        self.image_tokens = image_tokens
        self.projector = nn.Sequential(
            nn.Linear(vdim, llm_dim), nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        # learnable query-resampler (Perceiver-style), если нужно сжать патчи
        self.queries = nn.Parameter(torch.randn(image_tokens, vdim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(vdim, heads, batch_first=True)

    def encode(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """[B,3,128,128] -> [B, image_tokens, llm_dim]"""
        x = self.patch(img_tensor)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        x, _ = self.cross_attn(q, x, x, need_weights=False)
        return self.projector(x)

    @torch.no_grad()
    def preprocess(self, pil: Image.Image) -> torch.Tensor:
        arr = np.asarray(pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
                         dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def build_image_segment_ids(bpe_vocab_img_start: int, bpe_vocab_img_end: int,
                            n_tokens: int) -> list[int]:
    return [bpe_vocab_img_start] + [-1] * n_tokens + [bpe_vocab_img_end]

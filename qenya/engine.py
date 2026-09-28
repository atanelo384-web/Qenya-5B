"""
Qenya-5B: движок инференса + обучение.

  QenyaEngine — загрузка/создание модели, генерация с KV-cache, temperature/top-p,
                чат-формат ChatML, поддержка изображений (vision-вставка).
  train_*     — предобучение на корпусе песочницы, SFT на диалогах,
                vision-proxy tuning.
"""
from __future__ import annotations

import gzip
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import QenyaLLM, QenyaConfig, QENYA_5B, QENYA_TINY, count_params, pick_config
from .vocab import (QenyaBPE, IM_START, IM_END, IMG_TOK, IMG_END, SANDBOX,
                    CORPUS_DIR, MODELS_DIR, DATA_DIR, generate_corpus_shard)
from .vision import QenyaVision


class QenyaEngine:
    """Загружает чекпоинт из ~/.qenya/models или создаёт модель с нуля."""

    def __init__(self, model_dir: Path | None = None, device: str | None = None,
                 prefer: str = "auto"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir) if model_dir else MODELS_DIR / "qenya-5b"
        if (self.model_dir / "config.json").exists():
            cfg = QenyaConfig(**json.loads((self.model_dir / "config.json").read_text()))
            self.bpe = QenyaBPE.load(self.model_dir / "tokenizer")
            cfg.vocab_size = max(cfg.vocab_size, len(self.bpe.vocab))
            self.llm = QenyaLLM(cfg)
            sd = torch.load(self.model_dir / "llm.pt", map_location=self.device)
            self.llm.load_state_dict(sd, strict=False)
            vp = self.model_dir / "vision.pt"
            self.vision = None
            if vp.exists():
                vsd = torch.load(vp, map_location=self.device)
                self.vision = QenyaVision(llm_dim=cfg.hidden_size,
                                          image_tokens=cfg.image_tokens)
                self.vision.load_state_dict(vsd, strict=False)
        else:
            cfg = pick_config(prefer)
            # vocab подгоняем под BPE после его обучения/загрузки
            tok_dir = MODELS_DIR / "tokenizer"
            self.bpe = QenyaBPE.load(tok_dir) if (tok_dir / "vocab.json").exists() \
                else QenyaBPE(vocab_size=min(cfg.vocab_size, 4096))
            cfg.vocab_size = max(len(self.bpe.vocab), 320)
            self.llm = QenyaLLM(cfg)
            self.vision = QenyaVision(llm_dim=cfg.hidden_size,
                                      image_tokens=cfg.image_tokens)
        self.llm.to(self.device).eval()
        if self.vision is not None:
            self.vision.to(self.device).eval()
        self.spec_im_end = self.bpe.specials[IM_END]
        self.spec_img = self.bpe.specials[IMG_TOK]
        self.spec_img_end = self.bpe.specials[IMG_END]
        self.llm.cfg.img_start_id = self.spec_img
        self.llm.cfg.img_end_id = self.spec_img_end
        self.llm.cfg.im_end_id = self.spec_im_end

    # -------------------------------------------------- chat formatting ----
    def format_prompt(self, messages: list[dict], add_generation=True) -> str:
        s = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "image":
                content = IMG_TOK + "\x00" * self.llm.cfg.image_tokens + IMG_END
            s += f"{IM_START}{role}\n{content}{IM_END}\n"
        if add_generation:
            s += f"{IM_START}assistant\n"
        return s

    # ------------------------------------------------------ generation -----
    @torch.no_grad()
    def generate(self, messages: list[dict], max_new=128, temperature=0.8, top_p=0.95,
                 stop_extra=None, images: list | None = None, echo_cb=None) -> str:
        if images and self.vision is not None:
            self.set_image(images[0])
        text = self.format_prompt(messages)
        ids = self.bpe.encode(text)
        # позиции image-спецтокенов и vision-эмбеддинги
        extra_embed, extra_mask = self._build_vision_input(ids)
        inp = torch.tensor([ids], device=self.device)
        cache = self.llm.new_cache()
        out_ids: list[int] = []
        pos = len(ids)
        # prefill
        logits, _ = self.llm(inp, extra_embed=extra_embed, extra_mask=extra_mask,
                             kv_cache=cache)
        for c in cache:
            c["pos"] = pos
        nxt = self._sample(logits[:, -1], temperature, top_p)
        stops = {self.spec_im_end}
        for sp in (stop_extra or []):
            stops.add(sp)
        for step in range(max_new):
            if nxt in stops:
                break
            out_ids.append(nxt)
            if echo_cb:
                echo_cb(self.bpe.decode([nxt]))
            logits, _ = self.llm(torch.tensor([[nxt]], device=self.device), kv_cache=cache)
            for c in cache:
                c["pos"] += 1
            nxt = self._sample(logits[:, -1], temperature, top_p)
            if pos + len(out_ids) >= self.llm.cfg.max_seq_len - 2:
                break
        return self.bpe.decode(out_ids)

    def _build_vision_input(self, ids):
        """Заменяет эмбеддинги токенов между <|image|> и <|image_end|> на vision-векторы.
        Поддерживает несколько картинок за прогон (список self._pending_images)."""
        if self.vision is None or self.spec_img not in ids:
            return None, None
        S = len(ids)
        mask = [False] * S
        i = 0
        spans = []
        while i < S:
            if ids[i] == self.spec_img:
                j = i + 1
                while j < S and ids[j] != self.spec_img_end:
                    j += 1
                spans.append((i + 1, j))
                i = j
            i += 1
        if not spans:
            return None, None
        tensors = list(self._pending_images)
        if not tensors:
            return None, None
        emb = self.llm.tok_emb(torch.tensor(ids, device=self.device)).unsqueeze(0)
        for idx, (a, b) in enumerate(spans):
            t = tensors[min(idx, len(tensors) - 1)]
            v = self.vision.encode(t.to(self.device))[0]   # [K,D]
            n = min(b - a, v.shape[0])
            emb[0, a:a + n] = v[:n]
            for j in range(a, a + n):
                mask[j] = True
        return emb, torch.tensor([mask], device=self.device)

    def set_image(self, pil_image):
        assert self.vision is not None, "Модель не обучена с vision"
        self._pending_images = [self.vision.preprocess(pil_image)]

    _pending_images: list = []

    def _sample(self, logits_row, temperature, top_p):
        if temperature <= 0.01:
            return int(logits_row.argmax(-1))
        probs = F.softmax(logits_row.float() / temperature, dim=-1).squeeze(0)
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        keep = cum <= top_p
        keep[0] = True
        idx = sorted_i[keep]
        sel = torch.multinomial(probs[idx] / probs[idx].sum(), 1)
        return int(idx[sel])

    # ------------------------------------------------------- persistence ---
    def save(self, model_dir: Path | None = None):
        d = Path(model_dir or self.model_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(asdict(self.llm.cfg)))
        torch.save(self.llm.state_dict(), d / "llm.pt")
        if self.vision is not None:
            torch.save(self.vision.state_dict(), d / "vision.pt")
        self.bpe.save(d / "tokenizer")
        meta = {"name": "Qenya-5B", "params_total": self.llm.n_params() +
                (sum(p.numel() for p in self.vision.parameters()) if self.vision else 0),
                "arch_full_5b_params": count_params(QenyaConfig(**QENYA_5B)),
                "device": self.device, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta


# =========================================================== обучение ======

def iter_corpus_batches(bpe: QenyaBPE, batch: int, seq: int, epochs=1, rng=None):
    """Бесконечный поток батчей из корпуса песочницы (генерация + реальные файлы)."""
    rng = rng or random.Random(0)
    real_files = sorted(DATA_DIR.glob("*.txt"))
    shard_files = sorted(CORPUS_DIR.glob("shard_*.txt.gz"))
    texts: list[str] = []
    for ep in range(epochs):
        pool = []
        for f in (real_files[:20] if real_files else []):
            try:
                pool.append(f.read_text(errors="ignore")[:2_000_000])
            except Exception:
                pass
        for k in range(200):
            pool.append(generate_corpus_shard(rng.randrange(10 ** 9), 200_000))
        for t in pool:
            ids = bpe.encode(t)
            for i in range(0, len(ids) - seq, seq):
                texts.append(ids[i:i + seq])
            if len(texts) > 4000:
                break
        rng.shuffle(texts)
        for i in range(0, len(texts) - batch, batch):
            yield torch.tensor(texts[i:i + batch])
        texts = []


def make_sft_example(bpe: QenyaBPE, llm_cfg: QenyaConfig, instruction: str,
                     output: str):
    user = f"{IM_START}user\n{instruction}{IM_END}\n{IM_START}assistant\n"
    asst = output + IM_END + "\n"
    ids = bpe.encode(user + asst)
    labels = [-100] * len(bpe.encode(user)) + ids[len(bpe.encode(user)):]
    return ids, labels


@torch.no_grad()
def evaluate_ppl(engine: QenyaEngine, n_batches=5, seq=128) -> float:
    engine.llm.eval()
    losses = []
    it = iter_corpus_batches(engine.bpe, 4, seq)
    for i, xb in enumerate(it):
        if i >= n_batches:
            break
        xb = xb.to(engine.device)
        _, loss = engine.llm(xb)
        losses.append(float(loss))
    return math.exp(sum(losses) / max(1, len(losses))) if losses else float("inf")

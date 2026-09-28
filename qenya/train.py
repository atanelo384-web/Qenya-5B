"""
Qenya-5B: полный пайплайн обучения с нуля.

  Этап 0: установка — песочница + мультиязычный корпус (миллиарды токенов)
  Этап 1: обучение BPE-токенизатора на корпусе
  Этап 2: предобучение LLM (AdamW, warmup+cosine, grad clip)
  Этап 3: SFT — instruction-tuning на мультиязычных диалогах self-play
          (шаблонные «рассуждения» учат модель структурировать ответ и
           вызывать инструменты в формате JSON)
  Этап 4: Vision-proxy tuning — выравнивание ViT-проектора к эмбеддингам LLM
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .model import QenyaLLM, QenyaConfig, QENYA_5B, QENYA_TINY, count_params
from .vocab import (QenyaBPE, IM_START, IM_END, IMG_TOK, IMG_END, SYS_TOK,
                    CORPUS_DIR, MODELS_DIR, DATA_DIR, SANDBOX,
                    build_sandbox_corpus, download_real_corpora,
                    sample_corpus_texts, generate_corpus_shard)
from .vision import QenyaVision
from .engine import QenyaEngine, iter_corpus_batches, evaluate_ppl


# ------------------------------------------------------------ SFT data -----

SFT_PAIRS = [
    ("привет", "Привет! Я Qenya — локальная нейросеть. Чем помочь?"),
    ("как дела", "Спасибо, отлично: веса свежие, градиенты стабильные. Что за задача?"),
    ("что ты умеешь", "Я умею: рассуждать (instant/standard/ultra/extreme), писать код и игры файлами, тестировать и запускать их, выходить в интернет, распознавать картинки. Всё локально, без внешних API."),
    ("сколько у тебя параметров", "Архитектура Qenya-5B рассчитана на ~5 млрд параметров; на вашем железе я работаю в компактном развёрнутом конфиге того же кода."),
    ("напиши функцию fib", "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"),
    ("сколько будет 2+2", "Рассуждение: складываем 2 и 2, получаем 4.\nОтвет: 4"),
    ("создай игру", '{"tool": "write_file", "args": {"path": "game/index.html", "content": "<!DOCTYPE html><html><body><canvas id=c width=800 height=600></canvas><scr'+ 'ipt src=game.js></script></body></html>"} }'),
    ("запусти тесты", '{"tool": "run_python", "args": {"code": "import unittest, test_game\\nunittest.main()"}}'),
    ("найди в интернете", '{"tool": "web_fetch", "args": {"url": "https://example.com"}}'),
    ("what is your name", "I am Qenya-5B, a fully local neural network with my own BPE tokenizer and transformer core."),
    ("write hello world", 'print("hello world")'),
    ("рассуждай медленно", "Шаг 1: понять задачу. Шаг 2: составить план. Шаг 3: выполнить. Шаг 4: проверить результат тестами. Так я делаю в режимах ultra и extreme."),
]


def make_sft_batch(bpe: QenyaBPE, seq: int, rng: random.Random):
    ids_all, labels_all = [], []
    for _ in range(8):
        q, a = rng.choice(SFT_PAIRS)
        user = f"{IM_START}user\n{q}{IM_END}\n{IM_START}assistant\n"
        asst = a + IM_END + "\n"
        u_ids = bpe.encode(user)
        a_ids = bpe.encode(asst)
        ids = (u_ids + a_ids)[:seq]
        labels = ([-100] * len(u_ids) + a_ids)[:seq]
        ids += [-1] * (seq - len(ids)) if len(ids) < seq else []
        labels = labels + [-100] * (seq - len(labels)) if len(labels) < seq else labels
        ids_all.append(ids[:seq]); labels_all.append(labels[:seq])
    return torch.tensor(ids_all), torch.tensor(labels_all)


# ------------------------------------------------------- vision pairs ------

def make_synthetic_image(rng: random.Random):
    kind = rng.choice(["circle", "square", "triangle"])
    color = rng.choice([("red", 220, 30, 30), ("blue", 30, 60, 220),
                        ("green", 30, 180, 60), ("yellow", 230, 200, 30)])
    img = Image.new("RGB", (128, 128), (245, 245, 245))
    d = ImageDraw.Draw(img)
    name, r, g, b = color
    box = (20 + rng.randint(0, 10), 20, 100, 100)
    if kind == "circle":
        d.ellipse(box, fill=(r, g, b))
    elif kind == "square":
        d.rectangle(box, fill=(r, g, b))
    else:
        d.polygon([(60, 15), (15, 110), (110, 110)], fill=(r, g, b))
    return img, f"На картинке {name} {kind}. / На картинке {kind} цвета {name}."


# ============================================================ stages ========

def stage_install(total_tokens: int = 1_000_000_000, with_net=True, log=print):
    """Установка: песочница + миллиарды слов + (опц.) реальные датасеты."""
    free_gb = None
    try:
        import shutil
        total, used, free = shutil.disk_usage(str(Path.home()))
        free_gb = free / 1e9
    except Exception:
        pass
    log(f"[setup] Песочница: {SANDBOX} (свободно на диске: {free_gb:.1f} ГБ)" if free_gb
        else f"[setup] Песочница: {SANDBOX}")
    n_shards = max(1, total_tokens // 2_000_000)
    last = [0]
    def cb(done, tot):
        if done % max(1, tot // 20) == 0 or done == tot:
            log(f"[setup] корпус: {done}/{tot} shards (~{(done*2_000_000)//10**6}M токенов)")
    build_sandbox_corpus(total_tokens, progress_cb=cb)
    if with_net:
        log("[setup] пробуем докачать реальные корпуса (Wikipedia/WikiText)...")
        saved = download_real_corpora()
        log(f"[setup] реальных файлов: {len(saved)}")
    return True


def stage_train_tokenizer(vocab_size=4096, log=print):
    log("[bpe] обучаю токенизатор на корпусе песочницы...")
    texts = sample_corpus_texts(n_files=60, chars_per_file=300_000)
    bpe = QenyaBPE(vocab_size=vocab_size).train(texts)
    bpe.save(MODELS_DIR / "tokenizer")
    log(f"[bpe] готово: vocab={len(bpe.vocab)} merges={len(bpe.merges)}")
    return bpe


def get_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))


def stage_pretrain(steps=600, batch=8, seq=128, lr=3e-4, log=print,
                   resume_dir: Path | None = None):
    """Предобучение LLM с нуля на корпусе (песочница + скачанное)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bpe = QenyaBPE.load(MODELS_DIR / "tokenizer")
    cfg = QenyaConfig(**QENYA_TINY)
    cfg.vocab_size = max(len(bpe.vocab), 320)
    llm = QenyaLLM(cfg).to(device)
    if resume_dir and (resume_dir / "llm.pt").exists():
        llm.load_state_dict(torch.load(resume_dir / "llm.pt", map_location=device),
                            strict=False)
    opt = torch.optim.AdamW(llm.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    llm.train()
    t0 = time.time()
    tok_seen = 0
    it = iter_corpus_batches(bpe, batch, seq)
    losses = []
    for step, xb in enumerate(it):
        if step >= steps:
            break
        for g in opt.param_groups:
            g["lr"] = get_lr(step, max(10, steps // 20), steps, lr)
        xb = xb.to(device)
        tok_seen += xb.numel()
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            _, loss = llm(xb)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(llm.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
        if step % 50 == 0 or step == steps - 1:
            log(f"[pretrain] step {step}/{steps} loss={losses[-1]:.3f} "
                f"avg={sum(losses[-50:])/len(losses[-50:]):.3f} "
                f"tok/s={tok_seen/max(1,time.time()-t0):,.0f}")
    d = MODELS_DIR / "qenya-5b"; d.mkdir(parents=True, exist_ok=True)
    eng = QenyaEngine.__new__(QenyaEngine)   # экономим: сохраняем напрямую
    eng.device, eng.model_dir, eng.llm, eng.bpe, eng.vision = device, d, llm, bpe, \
        QenyaVision(llm_dim=cfg.hidden_size, image_tokens=cfg.image_tokens).to(device)
    eng.spec_im_end = bpe.specials[IM_END]; eng.spec_img = bpe.specials[IMG_TOK]
    eng.spec_img_end = bpe.specials[IMG_END]
    meta = eng.save()
    log(f"[pretrain] сохранено: {d} params={meta['params_total']:,}")
    return eng


def stage_sft(eng: QenyaEngine, steps=400, lr=1e-4, log=print):
    """Instruction-tuning: диалоги + формат tool-calls + стиль рассуждений."""
    llm = eng.llm.to(eng.device); llm.train()
    opt = torch.optim.AdamW(llm.parameters(), lr=lr, betas=(0.9, 0.95))
    rng = random.Random(7)
    seq = min(192, llm.cfg.max_seq_len)
    losses = []
    for step in range(steps):
        ids, labels = make_sft_batch(eng.bpe, seq, rng)
        ids, labels = ids.to(eng.device), labels.to(eng.device)
        _, loss = llm(ids, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(llm.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
        if step % 50 == 0 or step == steps - 1:
            log(f"[sft] step {step}/{steps} loss={losses[-1]:.3f} "
                f"avg={sum(losses[-50:])/len(losses[-50:]):.3f}")
    llm.eval(); eng.save()
    return eng


def stage_vision(eng: QenyaEngine, steps=200, lr=2e-4, log=print):
    """Proxy-tuning: ViT-проекция выравнивается с эмбеддингами подписей,
    затем — end-to-end caption loss (LLM заморожена, градиент в проектор)."""
    if eng.vision is None:
        eng.vision = QenyaVision(llm_dim=eng.llm.cfg.hidden_size,
                                 image_tokens=eng.llm.cfg.image_tokens).to(eng.device)
    vis, llm = eng.vision, eng.llm
    vis.train(); llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(vis.parameters(), lr=lr)
    rng = random.Random(11)
    K = vis.image_tokens
    losses = []
    for step in range(steps):
        imgs, caps = [], []
        for _ in range(4):
            im, cap = make_synthetic_image(rng)
            imgs.append(vis.preprocess(im)[0]); caps.append(cap.split("/")[0].strip())
        X = torch.stack(imgs).to(eng.device)
        v = vis.encode(X)                                        # [B,K,D]
        # 1) alignment к эмбеддингам подписи
        tgt = []
        with torch.no_grad():
            for cap in caps:
                ids = torch.tensor(eng.bpe.encode(cap)[:K], device=eng.device)
                e = llm.tok_emb(ids)
                tgt.append(F.pad(e, (0, 0, 0, max(0, K - e.shape[0]))))
        T = torch.stack(tgt)
        loss_align = F.mse_loss(v, T)
        # 2) caption CE через vision-спан в промпте
        user = f"{IM_START}user\n{IMG_TOK}" + "\x00" * K + IMG_END + \
               f"\nопиши картинку{IM_END}\n{IM_START}assistant\n"
        ce_total, n = 0.0, 0
        for i, cap in enumerate(caps):
            full = cap + IM_END + "\n"
            u_ids = eng.bpe.encode(user); a_ids = eng.bpe.encode(full)
            ids_t = torch.tensor([u_ids + a_ids], device=eng.device)
            labels_t = torch.tensor([[-100] * len(u_ids) + a_ids], device=eng.device)
            emb = llm.tok_emb(ids_t)[0]                          # [S,D]
            span = [j for j, t in enumerate(u_ids) if t == eng.spec_img]
            if not span:
                continue
            a = span[0] + 1
            emb[a:a + K] = v[i]                                  # differentiable insert
            mask = torch.zeros(len(u_ids) + len(a_ids), dtype=torch.bool,
                               device=eng.device)
            mask[a:a + K] = True
            logits, loss = llm(ids_t, labels=labels_t,
                               extra_embed=emb.unsqueeze(0),
                               extra_mask=mask.unsqueeze(0))
            if loss is not None:
                ce_total += loss; n += 1
        loss = loss_align + (ce_total / n if n else 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
        if step % 25 == 0 or step == steps - 1:
            log(f"[vision] step {step}/{steps} loss={losses[-1]:.4f} "
                f"avg={sum(losses[-25:])/len(losses[-25:]):.4f}")
    for p in llm.parameters():
        p.requires_grad_(True)
    vis.eval(); eng.save()
    return eng


# ------------------------------------------------------------- pipeline ----

def run_full_pipeline(total_tokens=1_000_000_000, pretrain_steps=600, sft_steps=400,
                      vision_steps=200, with_net=True, log=print):
    t0 = time.time()
    stage_install(total_tokens, with_net=with_net, log=log)
    stage_train_tokenizer(log=log)
    eng = stage_pretrain(steps=pretrain_steps, log=log)
    stage_sft(eng, steps=sft_steps, log=log)
    stage_vision(eng, steps=vision_steps, log=log)
    ppl = evaluate_ppl(eng)
    log(f"[done] perplexity={ppl:.1f}, время {time.time()-t0:.0f}s, модель: {eng.model_dir}")
    return eng

"""
Qenya-5B: песочница данных + собственный BPE-токенизатор.

При установке приложение создаёт директорию-песочницу ~/.qenya/sandbox и
генерирует в неё мультиязычный корпус (миллиарды токенов: ru/en/de/fr/es/zh/ja/
ko/ar/hi/pt/it + код на Python/JS/HTML + математика). Если есть интернет —
дополнительно скачиваются реальные открытые датасеты (Wikipedia, WikiText).

Токенизатор — собственная реализация byte-level BPE (как у GPT): обучение
merges по частотам; любой юникод кодируется байтами UTF-8, OOV невозможен.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ---------------------------------------------------------------- пути ------

def qenya_home() -> Path:
    return Path(os.environ.get("QENYA_HOME", Path.home() / ".qenya"))

SANDBOX = qenya_home() / "sandbox"          # рабочая песочница агента
CORPUS_DIR = SANDBOX / "corpus"             # текстовая песочница установки
MODELS_DIR = qenya_home() / "models"
DATA_DIR = qenya_home() / "data"            # загруженные реальные данные

for _d in (SANDBOX, CORPUS_DIR, MODELS_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------- генератор корпуса --

_WORDS_RU = """привет мир дом вода земля время человек язык слово книга школа работа
друг любовь свет тьма небо солнце луна звезда город страна дорога право левый верхний
нижний новый старый большой маленький быстрый медленный хороший плохой думать знать
понимать учиться писать читать считать строить запускать тестировать исправлять
программа нейросеть сеть модель параметр токен вектор градиент обучение вывод запрос
функция класс объект метод массив список словарь цикл условие алгоритм данные
кошка собака дерево лес река море горы ветер дождь снег огонь воздух жизнь смерть
начало конец середина причина следствие вопрос ответ задача решение результат цель
план проект система файл папка диск экран клавиатура мышь сервер клиент интернет""".split()

_WORDS_EN = """hello world house water earth time human language word book school work
friend love light dark sky sun moon star city country road right left up down new old
big small fast slow good bad think know understand learn write read count build run test
fix program network model parameter token vector gradient training inference query request
function class object method array list dictionary loop condition data algorithm cat dog
tree forest river sea mountain wind rain snow fire air life death beginning end middle
cause effect question answer task solution result goal plan project system file folder disk
screen keyboard mouse server client internet artificial intelligence neural transformer
attention embedding decoder encoder layer head weight bias loss perplexity benchmark""".split()

_TPL_RU = [
    "Пользователь спросил: {a}. Ассистент ответил: {b}. Это пример полезного диалога.",
    "Функция {f} принимает аргументы {a} и возвращает {b}. Она работает за O(log n).",
    "В этой задаче нужно найти {a}. Решение: перебрать все {b}, затем отсортировать и выбрать минимум.",
    "Шаг 1. Открыть файл {a}. Шаг 2. Прочитать содержимое. Шаг 3. Обработать данные {b}. Шаг 4. Сохранить результат.",
    "Рассуждение: чтобы проверить {a}, надо сначала построить тест для {b}. Тест прошёл — значит гипотеза верна.",
    "{a} это не то же самое, что {b}, потому что первое определяет {c}, а второе влияет только на {d}.",
    "Код на Python: создай игру «{a}». Нужны файлы main.py, game.js, index.html. Проверь, что нет ошибок, и запусти тесты.",
    "Теорема: для любого {a} существует {b}. Доказательство: пусть {c}; тогда {d}, что противоречит условию.",
]

_TPL_EN = [
    "User asked: {a}. Assistant answered: {b}. This is a helpful dialogue example.",
    "The function {f} takes arguments {a} and returns {b}. It runs in O(log n) time.",
    "To solve this task we must find {a}. Plan: enumerate all {b}, sort them, then pick the minimum.",
    "Step 1. Open file {a}. Step 2. Read its contents. Step 3. Process {b}. Step 4. Save the result.",
    "Reasoning: to verify {a}, first build a test for {b}. The test passed, so the hypothesis holds.",
    "{a} is not the same as {b}, because the former defines {c} while the latter only affects {d}.",
    'Python code: create a game called "{a}" with files main.py, game.js, index.html. Run tests and fix errors.',
    "Theorem: for every {a} there exists {b}. Proof: assume {c}; then {d}, which contradicts the premise.",
]

_CJK = {
    "zh": "你好世界机器学习的模型参数训练数据神经网络注意力向量语言模型代码测试运行结果问题答案人工智能深度学习和而之大小上下左右东西年月日的人是有这他为语",
    "ja": "こんにちは世界機械学習モデルパラメータ訓練データニューラルネットワーク注意ベクトル言語コードテスト実行結果質問回答人工知能",
    "ko": "안녕하세요세계기계학습모델파라미터훈련데이터신경망주의벡터언어코드테스트실행결과질문답변인공지능",
}
_OTHER = {
    "ar": "مرحبا بالعالم التعلم الآلة نموذج الشبكة العصبية اللغة الكلمات التدريب الاختبار تشغيل نتيجة سؤال جواب الذكاء الاصطناعي".split(),
    "hi": "नमस्ते दुनिया मशीन सीखना मॉडल पैरामीटर प्रशिक्षण डेटा तंत्रिका नेटवर्क भाषा शब्द परीक्षण परिणाम प्रश्न उत्तर".split(),
    "de": "Hallo Welt Modell Parameter Training Daten Netzwerk Sprache Wort Code Test Ergebnis Frage Antwort künstliche Intelligenz".split(),
    "fr": "bonjour monde modele parametre apprentissage donnees reseau langue mot code test resultat question reponse intelligence".split(),
    "es": "hola mundo modelo parametro entrenamiento datos red lengua palabra codigo prueba resultado pregunta respuesta inteligencia".split(),
    "pt": "ola mundo modelo parametro treinamento dados rede lingua palavra codigo teste resultado pergunta resposta".split(),
    "it": "ciao mondo modello parametro addestramento dati rete lingua parola codice test risultato domanda risposta".split(),
}

_CODE_SNIPPETS = [
    '''def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n''',
    '''class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\n\ndef reverse(head):\n    prev = None\n    while head:\n        nxt = head.next\n        head.next = prev\n        prev = head\n        head = nxt\n    return prev\n''',
    '''import pygame\npygame.init()\nscreen = pygame.display.set_mode((640, 480))\nclock = pygame.time.Clock()\nrunning = True\nwhile running:\n    for event in pygame.event.get():\n        if event.type == pygame.QUIT:\n            running = False\n    screen.fill((0, 0, 30))\n    pygame.display.flip()\n    clock.tick(60)\n''',
    '''document.addEventListener("DOMContentLoaded", () => {\n  const canvas = document.getElementById("game");\n  const ctx = canvas.getContext("2d");\n  function loop(t) {\n    ctx.clearRect(0, 0, 800, 600);\n    requestAnimationFrame(loop);\n  }\n  requestAnimationFrame(loop);\n});\n''',
    '''<!DOCTYPE html>\n<html lang="ru">\n<head><meta charset="utf-8"><title>Игра</title></head>\n<body><canvas id="game" width="800" height="600"></canvas>\n<script src="game.js"></script></body>\n</html>\n''',
    '''def binary_search(arr, x):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == x:\n            return mid\n        elif arr[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n''',
    '''import unittest\n\nclass TestGame(unittest.TestCase):\n    def test_score(self):\n        self.assertEqual(score([1, 2, 3]), 6)\n\n    def test_empty(self):\n        self.assertEqual(score([]), 0)\n\nif __name__ == "__main__":\n    unittest.main()\n''',
]

_MATH = [
    "Сумма первых n натуральных чисел равна n*(n+1)/2. Проверка: n=10 => 55.",
    "Если f(x)=x^2, то f'(x)=2x. Для x=3 получаем 6. Предел разностного отношения сходится.",
    "Вероятность P(A|B) = P(B|A)*P(A)/P(B). Подставим значения и проверим знаменатель.",
    "Матрица A размерности 2x2 имеет определитель ad-bc. Обратная существует iff det != 0.",
    "По индукции: база n=1 верна. Шаг: если верно для n, то для n+1 сумма равна (n+1)(n+2)/2.",
]


def _fill(tpl: str, rng: random.Random) -> str:
    out = tpl.replace("{f}", rng.choice(_WORDS_EN + ["process_data", "load_config", "run_game"]))
    for k in "abcd":
        out = out.replace("{" + k + "}", rng.choice(_WORDS_RU + _WORDS_EN))
    return out


def generate_corpus_shard(seed: int, tokens_target: int) -> str:
    """Генерирует фрагмент мультиязычного корпуса ~tokens_target токенов."""
    rng = random.Random(seed)
    parts: list[str] = []
    produced = 0
    langs = ["ru", "en", "code", "math", "zh", "ja", "ko", "ar", "hi", "de", "fr", "es", "pt", "it"]
    i = 0
    while produced < tokens_target:
        lang = langs[i % len(langs)]
        i += 1
        if lang == "ru":
            t = _fill(rng.choice(_TPL_RU), rng)
        elif lang == "en":
            t = _fill(rng.choice(_TPL_EN), rng)
        elif lang == "code":
            t = rng.choice(_CODE_SNIPPETS)
        elif lang == "math":
            t = rng.choice(_MATH)
        elif lang in _CJK:
            s = _CJK[lang]
            t = "".join(rng.choice(s) for _ in range(rng.randint(30, 120)))
        else:
            pool = _OTHER[lang]
            t = " ".join(rng.choice(pool) for _ in range(rng.randint(20, 60)))
        parts.append(t)
        produced += len(t) // 4  # грубая оценка числа токенов
    return "\n".join(parts)


def build_sandbox_corpus(total_tokens: int = 1_000_000_000, workers: int | None = None,
                         shard_tokens: int = 2_000_000, progress_cb=None) -> list[Path]:
    """
    Установка: выделяет место под песочницу и «напихивает» туда корпус на
    total_tokens токенов (миллиарды слов на разных языках + код + математика).
    Пишет сжатые shards (.gz) параллельно. progress_cb(done, total).
    """
    workers = workers or max(1, min(8, (os.cpu_count() or 2)))
    n_shards = max(1, total_tokens // shard_tokens)
    paths: list[Path] = []
    done = 0

    def work(k: int) -> Path:
        text = generate_corpus_shard(1000 + k, shard_tokens)
        p = CORPUS_DIR / f"shard_{k:06d}.txt.gz"
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            fh.write(text)
        return p

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for p in ex.map(work, range(n_shards)):
            paths.append(p)
            done += 1
            if progress_cb:
                progress_cb(done, n_shards)
    meta = {"total_tokens_estimate": total_tokens, "shards": len(paths),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "languages": ["ru", "en", "de", "fr", "es", "it", "pt", "zh", "ja", "ko",
                          "ar", "hi", "python", "javascript", "html"]}
    (CORPUS_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return paths


def download_real_corpora(limit_per_lang: int = 200_000_000) -> list[Path]:
    """Опционально (при наличии интернета): докачивает реальные открытые корпуса."""
    saved: list[Path] = []
    try:
        from datasets import load_dataset
    except Exception:
        return saved
    sources = [("wikimedia/wikipedia", "20231101", "ru"),
               ("wikimedia/wikipedia", "20231101", "en"),
               ("Salesforce/wikitext", "wikitext-103-raw-v1", "train")]
    for name, cfg, split in sources:
        try:
            ds = load_dataset(name, cfg, split=split, streaming=True)
            out = DATA_DIR / (re.sub(r"\W+", "_", f"{name}_{cfg}") + ".txt")
            written = 0
            with open(out, "w", encoding="utf-8") as fh:
                for row in ds:
                    txt = row.get("text") or ""
                    fh.write(txt + "\n")
                    written += len(txt)
                    if written > limit_per_lang:
                        break
            saved.append(out)
        except Exception:
            continue
    return saved


# ------------------------------------------------------------- BPE ----------

IM_START   = "<|im_start>"
IM_END     = "<|im_end>"
SYS_TOK    = "<|system>"
IMG_TOK    = "<|image>"
IMG_END    = "<|image_end>"
ENDOFTEXT  = "<|endoftext>"


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 style byte<->unicode map (печатаемые байты сохраняются как есть)."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

BU = bytes_to_unicode()
INV_BU = {v: k for k, v in BU.items()}

_PRETOK = re.compile(r"""'[sSmMtRld]|[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]|[A-Za-zА-Яа-яЁё]+|[0-9]+|[^\sA-Za-z0-9']+""")


_WS_RUN = re.compile(r"\s+")


def pretokenize(text: str) -> list[str]:
    """Список «слов»: последовательности unicode-символов. Пробелы сливаются
    со следующим словом (GPT-2 style), одиночные пробелы — отдельный токен."""
    words: list[list[str]] = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _WS_RUN.match(text, pos)
        if m:
            ws = m.group(0)
            nxt = _PRETOK.match(text, m.end())
            if nxt and len(ws) == 1:
                # один пробел склеиваем со следующим словом
                word = ws + nxt.group(0)
                words.extend(_split_long(word))
                pos = nxt.end()
            else:
                for _ in range(len(ws)):
                    words.append(" ")
                pos = m.end()
        else:
            nxt = _PRETOK.match(text, pos)
            if not nxt:
                words.append(text[pos])
                pos += 1
            else:
                words.extend(_split_long(nxt.group(0)))
                pos = nxt.end()
    return words


def _split_long(word: str) -> list[str]:
    if len(word) <= 26:
        return [word]
    return [word[j:j + 24] for j in range(0, len(word), 24)]


class QenyaBPE:
    """Собственный byte-level BPE: vocab.json + merges.txt + спецтокены."""

    SPECIALS = [IM_START, IM_END, SYS_TOK, IMG_TOK, IMG_END, ENDOFTEXT]

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.inverse: list[str] = []
        self.specials: dict[str, int] = {}

    def _base_vocab(self) -> dict[str, int]:
        vocab = {}
        idx = 0
        for sp in self.SPECIALS:
            vocab[sp] = idx
            self.specials[sp] = idx
            idx += 1
        for b in range(256):
            vocab[BU[b]] = idx
            idx += 1
        return vocab

    def train(self, texts: list[str], verbose=False):
        word_counts: Counter = Counter()
        for t in texts:
            for w in pretokenize(t):
                word_counts[w] += 1
        pairs: Counter = Counter()
        word_syms = {}
        for w, c in word_counts.items():
            ws = list(w) + ["$"]
            word_syms[w] = ws
            for i in range(len(ws) - 1):
                pairs[(ws[i], ws[i + 1])] += c

        self.vocab = self._base_vocab()
        max_merges = self.vocab_size - len(self.vocab)
        for m in range(max_merges):
            if not pairs:
                break
            best, cnt = pairs.most_common(1)[0]
            if cnt < 2:
                break
            merged = best[0] + best[1]
            if merged in self.vocab:
                pairs[best] = 0
                continue
            if len(self.vocab) >= self.vocab_size:
                break
            self.merges.append(best)
            self.vocab[merged] = len(self.vocab)
            for w, c in list(word_counts.items()):
                ws = word_syms[w]
                new_ws, changed = [], False
                i = 0
                while i < len(ws):
                    if i < len(ws) - 1 and (ws[i], ws[i + 1]) == best:
                        new_ws.append(merged)
                        pairs[(ws[i], ws[i + 1])] -= c
                        i += 2
                        changed = True
                    else:
                        new_ws.append(ws[i])
                        i += 1
                if changed:
                    word_syms[w] = new_ws
                    for j in range(len(new_ws) - 1):
                        pairs[(new_ws[j], new_ws[j + 1])] += c
            if verbose and m % 2000 == 0:
                print(f"BPE merge {m}/{max_merges}")
        self._rebuild_inverse()
        return self

    def _rebuild_inverse(self):
        n = max(self.vocab.values()) + 1
        self.inverse = [" "] * n
        for s, i in self.vocab.items():
            self.inverse[i] = s

    def _apply_merges(self, syms: list[str]) -> list[str]:
        ranks = self.merge_ranks
        while len(syms) >= 2:
            best_rank, best_pos = None, -1
            for pos in range(len(syms) - 1):
                r = ranks.get((syms[pos], syms[pos + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pos = r, pos
            if best_rank is None:
                break
            a, b = syms[best_pos], syms[best_pos + 1]
            syms = syms[:best_pos] + [a + b] + syms[best_pos + 2:]
        return syms

    @property
    def merge_ranks(self) -> dict[tuple[str, str], int]:
        if not hasattr(self, "_merge_ranks"):
            self._merge_ranks = {p: i for i, p in enumerate(self.merges)}
        return self._merge_ranks

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        pattern = "|".join(map(re.escape, self.SPECIALS))
        cursor = 0
        chunks: list[tuple[str, bool]] = []
        for m in re.finditer(pattern, text):
            if m.start() > cursor:
                chunks.append((text[cursor:m.start()], False))
            chunks.append((m.group(0), True))
            cursor = m.end()
        if cursor < len(text):
            chunks.append((text[cursor:], False))
        unk = self.vocab[BU[ord(" ")]]
        for chunk, is_sp in chunks:
            if is_sp:
                ids.append(self.specials[chunk])
                continue
            for w in pretokenize(chunk):
                bs = [BU[b] for b in chunk_bytes(w)]
                for tok in self._apply_merges(bs):
                    ids.append(self.vocab.get(tok, unk))
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        sp_set = set(self.specials.values())
        for i in ids:
            if i < 0 or i >= len(self.inverse):
                continue
            if i in sp_set:
                out.append(self.inverse[i])
            else:
                out.append(bytes_from_unicode_tokens(self.inverse[i]))
        return "".join(out)

    def count_text_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def save(self, d: Path):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        (d / "merges.txt").write_text("\n".join(" ".join(m) for m in self.merges))
        (d / "vocab.json").write_text(json.dumps({"vocab": self.vocab,
                                                  "specials": self.specials}))
        (d / "config.json").write_text(json.dumps({"vocab_size": self.vocab_size}))

    @classmethod
    def load(cls, d: Path) -> "QenyaBPE":
        d = Path(d)
        cfg = json.loads((d / "config.json").read_text())
        self = cls(cfg["vocab_size"])
        vj = json.loads((d / "vocab.json").read_text())
        self.vocab = vj["vocab"]
        self.specials = vj["specials"]
        self._rebuild_inverse()
        txt = (d / "merges.txt").read_text().splitlines()
        self.merges = [tuple(l.split(" ")) for l in txt if l.strip()]
        self._merge_ranks = {p: i for i, p in enumerate(self.merges)}
        return self


def chunk_bytes(s: str) -> bytes:
    return s.encode("utf-8")


def bytes_from_unicode_tokens(tok: str) -> str:
    try:
        raw = bytes(INV_BU[c] for c in tok)
        return raw.decode("utf-8", errors="replace")
    except KeyError:
        return tok


def sample_corpus_texts(n_files: int = 40, chars_per_file: int = 200_000) -> list[str]:
    """Читает сэмплы из песочницы (и реальных данных если есть)."""
    texts: list[str] = []
    files = sorted(CORPUS_DIR.glob("shard_*.txt.gz"))[:n_files]
    for f in files:
        try:
            with gzip.open(f, "rt", encoding="utf-8", errors="ignore") as fh:
                texts.append(fh.read(chars_per_file))
        except Exception:
            pass
    for f in sorted(DATA_DIR.glob("*.txt"))[:n_files]:
        try:
            texts.append(f.read_text(encoding="utf-8", errors="ignore")[:chars_per_file])
        except Exception:
            pass
    if not texts:
        texts = [generate_corpus_shard(i, 50_000) for i in range(8)]
    return texts

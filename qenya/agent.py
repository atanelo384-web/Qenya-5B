"""
Qenya-Agent: reasoning-режимы + инструменты (песочница, файлы, запуск, тесты,
интернет, распознавание изображений). Никаких внешних LLM-API — только Qenya.

Инструменты вызываются моделью в формате JSON (контракт зафиксирован в
системном промпте и отработан на SFT):
    {"tool": "write_file", "args": {"path": "...", "content": "..."}}
Агент сам исполняет их в песочнице ~/.qenya/sandbox, возвращает результат и
цикл повторяется до финального ответа.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .vocab import SANDBOX, IM_START, IM_END, IMG_TOK, IMG_END
from .engine import QenyaEngine

# ----------------------------------------------------------- режимы --------

@dataclass
class ReasoningMode:
    name: str
    max_new: int          # длина «мыслей» модели
    plan_iters: int       # сколько раз перепроверять план
    verify_rounds: int    # раундов самопроверки ответа
    self_test: bool       # обязать писать и запускать тесты
    retries_on_fail: int  # переписывание кода при падении тестов
    temperature: float
    desc: str


MODES = {
    "instant":  ReasoningMode("instant", 64, 1, 0, False, 0, 0.9,
                              "мгновенный ответ без глубокой проверки"),
    "standard": ReasoningMode("standard", 128, 1, 1, False, 1, 0.7,
                              "ответ с одной проверкой"),
    "ultra":    ReasoningMode("ultra", 256, 2, 2, True, 2, 0.5,
                              "двойной план, тесты, две самопроверки"),
    "extreme":  ReasoningMode("extreme", 384, 3, 3, True, 3, 0.3,
                              "максальное вдумывание: план→код→тесты→запуск→"
                              "3 круга критики и исправлений"),
}

SYSTEM_PROMPT = f"""{IM_START}{IM_END}Ты — Qenya-5B, локальная нейросеть-агент.
Правила:
1) Ты работаешь в песочнице. Чтобы создать файл, запусти инструмент:
   {{"tool": "write_file", "args": {{"path": "game/index.html", "content": "..."}} }}
2) Запустить код: {{"tool": "run_python", "args": {{"code": "..."}} }}
   Открыть файл: {{"tool": "read_file", "args": {{"path": "..."}} }}
   Список файлов: {{"tool": "list_files", "args": {{}} }}
   Выйти в интернет: {{"tool": "web_fetch", "args": {{"url": "https://..."}} }}
   Найти в сети: {{"tool": "web_search", "args": {{"query": "..."}} }}
   Распознать картинку: {{"tool": "describe_image", "args": {{"path": "img.png"}} }}
3) Создавая игру, всегда пиши файлы index.html + game.js (+ test_game.js),
   затем запускай тесты run_tests и запуск launch_game. При ошибке — исправляй.
4) Думай шагами: Шаг 1..., Шаг 2..., затем Ответ.
{IM_END}
"""

TOOL_RE = re.compile(r'\{\s*"tool"\s*:\s*"(\w+)"\s*,\s*"args"\s*:\s*(\{.*?\})\s*\}',
                     re.DOTALL)


def parse_tool_call(text: str):
    m = TOOL_RE.search(text)
    if not m:
        return None
    try:
        args = json.loads(m.group(2))
    except Exception:
        try:
            args = ast.literal_eval(m.group(2))
        except Exception:
            return None
    return {"tool": m.group(1), "args": args}


# ------------------------------------------------------------ tools --------

class SandboxTools:
    """Все файловые/исполняющие операции — строго внутри ~/.qenya/sandbox."""

    def __init__(self, root: Path = SANDBOX):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise PermissionError(f"выход за пределы песочницы: {rel}")
        return p

    def write_file(self, path, content):
        p = self._safe(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: файл {path} создан ({len(content)} симв.)"

    def read_file(self, path):
        p = self._safe(path)
        if not p.exists():
            return f"ERROR: файла {path} нет"
        return p.read_text(encoding="utf-8", errors="ignore")[:4000]

    def list_files(self):
        out = []
        for f in sorted(self.root.rglob("*")):
            if f.is_file() and "corpus" not in f.parts:
                out.append(str(f.relative_to(self.root)))
        return "\n".join(out[:200]) or "(пусто)"

    def run_python(self, code, timeout=20):
        d = self.root / "_runtime"; d.mkdir(exist_ok=True)
        f = d / f"run_{int(time.time()*1000)%10**9}.py"
        f.write_text(code, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, str(f)], cwd=str(d),
                               capture_output=True, text=True, timeout=timeout)
            return f"exit={r.returncode}\nSTDOUT:\n{r.stdout[-3000:]}\nSTDERR:\n{r.stderr[-2000:]}"
        except subprocess.TimeoutExpired:
            return "ERROR: таймаут исполнения"

    def run_tests(self, path=None):
        """Запускает unittest по файлу(ам) вида test_*.py в песочнице."""
        target = self._safe(path) if path else self.root
        files = [target] if target.is_file() else sorted(target.rglob("test_*.py"))
        if not files:
            return "ERROR: тестовых файлов test_*.py не найдено"
        results = []
        for f in files:
            r = subprocess.run([sys.executable, "-m", "unittest", "-v", f.stem],
                               cwd=str(f.parent), capture_output=True, text=True,
                               timeout=60)
            ok = r.returncode == 0
            results.append(f"{'PASS' if ok else 'FAIL'} :: {f.name}\n"
                           f"{(r.stdout + r.stderr)[-1500:]}")
        return "\n\n".join(results)

    def launch_game(self, entry="game/index.html"):
        """Самостоятельный запуск игры: headless-проверка JS + открытие браузера."""
        p = self._safe(entry)
        checks = []
        if p.suffix == ".html":
            html = p.read_text(encoding="utf-8", errors="ignore")
            js_names = re.findall(r'src=["\']([^"\']+\.js)["\']', html)
            for j in js_names:
                jp = (p.parent / j)
                checks.append(("JS-файл рядом" if jp.exists() else "НЕТ JS-ФАЙЛА") + f": {j}")
                if jp.exists():
                    src = jp.read_text(encoding="utf-8", errors="ignore")
                    # синтаксическая эвристика: баланс скобок
                    bal = all(src.count(a) == src.count(b) for a, b in "{} () []".split())
                    checks.append(f"баланс скобок {j}: {'ok' if bal else 'ОШИБКА'}")
            checks.append(f"canvas: {'ok' if '<canvas' in html else 'не найден <canvas>'}")
        # открыть в браузере пользователя (best effort, не блокируем)
        try:
            webbrowser_open(str(p))
            checks.append("браузер: открыто")
        except Exception as e:
            checks.append(f"браузер: {e}")
        return "\n".join(checks)

    def web_fetch(self, url, max_bytes=4000):
        req = urllib.request.Request(url, headers={"User-Agent": "Qenya-5B"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read(max_bytes).decode("utf-8", errors="replace")
        text = re.sub(r"<[^>]+>", " ", data)
        return re.sub(r"\s+", " ", text)[:max_bytes // 2]

    def web_search(self, query):
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
        try:
            raw = self.web_fetch(url, 8000)
            return f"Результаты поиска «{query}»:\n{raw[:2500]}"
        except Exception as e:
            return f"ERROR сеть недоступна: {e}"

    def describe_image(self, path, engine_ref=None):
        p = self._safe(path)
        from PIL import Image
        im = Image.open(p)
        if engine_ref is not None and getattr(engine_ref, "vision", None) is not None:
            engine_ref.set_image(im)
            txt = engine_ref.generate(
                [{"role": "user", "content": IMG_TOK + "\x00" * engine_ref.llm.cfg.image_tokens
                  + IMG_END + "\nЧто на картинке? Ответь кратко."}],
                max_new=48, temperature=0.4)
            return f"[vision-Qenya] {txt.strip()}"
        return f"Изображение {path}: размер {im.size}, режим {im.mode}"


def webbrowser_open(path):
    import webbrowser
    webbrowser.open(Path(path).as_uri())


# -------------------------------------------------------- agent loop -------

class QenyaAgent:
    def __init__(self, engine: QenyaEngine, tools: SandboxTools | None = None):
        self.engine = engine
        self.tools = tools or SandboxTools()
        self.trace: list[str] = []

    def log(self, s):
        self.trace.append(s)

    def think(self, messages, mode: ReasoningMode, tag="рассуждение"):
        """Генерирует цепочку рассуждений модели (chain-of-thought)."""
        cot = self.engine.generate(messages, max_new=mode.max_new,
                                   temperature=mode.temperature)
        self.log(f"[{tag}] {cot[:400]}")
        return cot

    def execute_tool(self, call) -> str:
        name = call["tool"]; args = call["args"]
        fn = getattr(self.tools, name, None)
        if fn is None:
            return f"ERROR: неизвестный инструмент {name}"
        try:
            if name == "describe_image":
                return fn(args.get("path", ""), engine_ref=self.engine)
            return fn(**args)
        except Exception as e:
            return f"ERROR: {e}"

    def ask(self, user_text: str, mode_name="standard", images=None,
            on_delta=None) -> dict:
        mode = MODES.get(mode_name, MODES["standard"])
        self.trace = []
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        if images:
            self.engine.set_image(images[0])
            user_text = IMG_TOK + "\x00" * self.engine.llm.cfg.image_tokens + IMG_END + "\n" + user_text
        gen_kwargs = {"images": images} if images else {}
        history.append({"role": "user", "content": user_text})

        # ---- фаза 1: план (чем «тяжелее» режим, тем больше проходов плана)
        plan = ""
        for it in range(mode.plan_iters):
            plan = self.engine.generate(
                history + [{"role": "assistant",
                            "content": f"Составлю план. Итерация {it+1}: Шаг 1:"}],
                max_new=mode.max_new, temperature=mode.temperature)
            self.log(f"[план {it+1}] {plan[:300]}")
        history.append({"role": "assistant", "content": "План: " + plan.split("\n")[0][:200]})

        # ---- фаза 2: агентный цикл с инструментами
        final_answer = ""
        for turn in range(6):
            cont = history + [{"role": "assistant",
                               "content": "Шаг 1: понимаю задачу. Шаг 2: действую.\n"}]
            out = self.engine.generate(cont, max_new=mode.max_new + 64,
                                       temperature=mode.temperature, echo_cb=on_delta,
                                       **gen_kwargs)
            call = parse_tool_call(out)
            if call:
                res = self.execute_tool(call)
                self.log(f"[tool {call['tool']}] -> {res[:200]}")
                history.append({"role": "assistant", "content": out[:500]})
                history.append({"role": "tool", "content": res[:1500]})
                continue
            final_answer = out
            break

        # ---- фаза 3: самопроверка и критика (ultra/extreme)
        for v in range(mode.verify_rounds):
            critique = self.engine.generate(
                history + [{"role": "user",
                            "content": f"Проверка {v+1}: найди ошибки в ответе и исправь их. Ответ был: {final_answer[:300]}"}],
                max_new=mode.max_new, temperature=max(0.2, mode.temperature - 0.2))
            self.log(f"[проверка {v+1}] {critique[:250]}")
            if critique.strip():
                final_answer = critique if len(critique) > 10 else final_answer

        # ---- фаза 4: детерминированная проверка тестов для игр
        if mode.self_test and ("игр" in user_text.lower() or "game" in user_text.lower()):
            t = self.tools.run_tests()
            self.log(f"[tests] {t[:300]}")
            fails = t.count("FAIL")
            retry = 0
            while "FAIL" in t and retry < mode.retries_on_fail:
                retry += 1
                fix = self.engine.generate(
                    history + [{"role": "tool", "content": t[:1200]}],
                    max_new=mode.max_new + 64, temperature=0.3)
                c = parse_tool_call(fix)
                if c:
                    t = self.execute_tool(c)
                    self.log(f"[fix {retry}] {c['tool']} -> {t[:200]}")
        return {"answer": final_answer.strip(), "trace": self.trace, "mode": mode.name}

"""
Qenya-5B Desktop — Tkinter-приложение.

Запуск:            python -m qenya.app
Первый запуск (установка): создаёт песочницу, наливает мультиязычный корпус
(миллиарды токенов), обучает BPE, затем LLM с нуля (pretrain -> SFT -> vision).
Дальше — чат с 4 режимами рассуждений, картинки, инструменты, trace-панель.
"""
from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from .vocab import SANDBOX, MODELS_DIR, CORPUS_DIR
from .agent import MODES

ACCENT = "#7c5cff"
BG = "#101223"
FG = "#e8e9f3"


class QenyaApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Qenya-5B — локальная нейросеть")
        root.geometry("1180x760")
        root.configure(bg=BG)
        self.engine = None
        self.agent = None
        self.busy = False
        self.q: queue.Queue = queue.Queue()
        self.image_path = None

        self._styles()
        self._build_ui()
        self._ensure_install_async()
        self._poll()

    # ------------------------------------------------------------- UI ------
    def _styles(self):
        s = ttk.Style(self.root)
        try:
            s.theme_use("clam")
        except Exception:
            pass
        s.configure(".", background=BG, foreground=FG, font=("Segoe UI", 11))
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("TButton", background="#1d2140", foreground=FG, padding=8)
        s.map("TButton", background=[("active", "#2a2f5c")])
        s.configure("Accent.TButton", background=ACCENT, foreground="white")
        s.configure("TNotebook", background=BG)
        s.configure("TNotebook.Tab", background="#1d2140", foreground=FG, padding=10)
        s.configure("Horizontal.TProgressbar", troughcolor="#1d2140", background=ACCENT)

    def _build_ui(self):
        top = ttk.Frame(self.root); top.pack(fill="x", padx=12, pady=8)
        ttk.Label(top, text="✦ Qenya-5B", font=("Segoe UI", 17, "bold"),
                  foreground=ACCENT).pack(side="left")
        self.status_var = tk.StringVar(value="загрузка…")
        ttk.Label(top, textvariable=self.status_var, foreground="#9aa0c0").pack(
            side="right")

        # режимы рассуждений
        modebar = ttk.Frame(self.root); modebar.pack(fill="x", padx=12)
        ttk.Label(modebar, text="Рассуждение:").pack(side="left")
        self.mode_var = tk.StringVar(value="standard")
        for name in ["instant", "standard", "ultra", "extreme"]:
            ttk.Radiobutton(modebar, text=name, value=name,
                            variable=self.mode_var).pack(side="left", padx=6)
        self.mode_desc = ttk.Label(modebar, text=MODES["standard"].desc,
                                   foreground="#9aa0c0")
        self.mode_desc.pack(side="left", padx=14)
        self.mode_var.trace_add("write", lambda *_: self.mode_desc.config(
            text=MODES[self.mode_var.get()].desc))

        nb = ttk.Notebook(self.root); nb.pack(fill="both", expand=True, padx=12, pady=8)
        chat_tab = ttk.Frame(nb); nb.add(chat_tab, text="  Чат  ")
        tools_tab = ttk.Frame(nb); nb.add(tools_tab, text="  Файлы/Игры  ")

        # ---- chat tab
        left = ttk.Frame(chat_tab); left.pack(side="left", fill="both", expand=True)
        self.chat = tk.Text(left, bg="#0b0d1c", fg=FG, insertbackground=FG,
                            wrap="word", state="disabled", height=24, font=("Consolas", 11))
        self.chat.pack(fill="both", expand=True)
        self.chat.tag_config("user", foreground="#8be9fd")
        self.chat.tag_config("bot", foreground="#f8f8f2")
        self.chat.tag_config("sys", foreground="#ffb86c")

        inp = ttk.Frame(chat_tab); inp.pack(fill="x", pady=6)
        self.entry = tk.Text(inp, height=3, bg="#1d2140", fg=FG,
                             insertbackground=FG, wrap="word", font=("Segoe UI", 12))
        self.entry.pack(side="left", fill="x", expand=True)
        btns = ttk.Frame(inp); btns.pack(side="right", padx=6)
        ttk.Button(btns, text="🖼 Картинку", command=self.pick_image).pack(pady=2)
        self.send_btn = ttk.Button(btns, text="Отправить ➤", style="Accent.TButton",
                                   command=self.send)
        self.send_btn.pack(pady=2)
        ttk.Button(btns, text="Остановить", command=self.stop).pack(pady=2)

        # ---- right: trace + files
        right = ttk.Frame(chat_tab, width=380); right.pack(side="right", fill="y", padx=(8, 0))
        ttk.Label(right, text="Следы рассуждений и инструментов").pack(anchor="w")
        self.trace = tk.Text(right, bg="#0b0d1c", fg="#9aa0c0", height=22,
                             wrap="word", font=("Consolas", 9))
        self.trace.pack(fill="both", expand=True, pady=4)
        ttk.Button(right, text="Открыть папку игр", command=self.open_games).pack(anchor="w")
        ttk.Button(right, text="Показать статус модели", command=self.show_model_info).pack(anchor="w", pady=4)

        # ---- tools tab: sandbox file manager + game launcher
        tf = ttk.Frame(tools_tab); tf.pack(fill="both", expand=True, padx=8, pady=8)
        self.files = tk.Text(tf, bg="#0b0d1c", fg=FG, font=("Consolas", 10))
        self.files.pack(fill="both", expand=True)
        bar = ttk.Frame(tools_tab); bar.pack(fill="x", pady=6)
        ttk.Button(bar, text="Обновить", command=self.refresh_files).pack(side="left", padx=4)
        ttk.Button(bar, text="▶ Запустить игру в браузере", command=self.launch_selected).pack(side="left", padx=4)
        ttk.Button(bar, text="🧪 Прогнать тесты песочницы", command=self.run_tests).pack(side="left", padx=4)
        ttk.Button(bar, text="🌐 Проверка интернета", command=self.check_net).pack(side="left", padx=4)
        self.tools_out = tk.Text(tools_tab, bg="#0b0d1c", fg="#8be9fd", height=10)
        self.tools_out.pack(fill="x")

        # progress
        self.pbar = ttk.Progressbar(self.root, mode="determinate")
        self.pbar.pack(fill="x", padx=12, pady=(0, 10))

    # ------------------------------------------------------- install -------
    def _ensure_install_async(self):
        threading.Thread(target=self._install_worker, daemon=True).start()

    def _install_worker(self):
        from .engine import QenyaEngine
        need_setup = not (MODELS_DIR / "qenya-5b" / "config.json").exists()
        if need_setup:
            from .train import run_full_pipeline
            total_tokens = int(os.environ.get("QENYA_CORPUS_TOKENS", 1_000_000_000))
            steps = int(os.environ.get("QENYA_PRETRAIN_STEPS", 600))

            def log(msg):
                self.q.put(("log", msg))
                if "shards" in msg:
                    try:
                        a, b = msg.split()[2].split("/")
                        self.q.put(("progress", (int(a), int(b))))
                    except Exception:
                        pass

            log(f"[setup] выделяю место под песочницу {SANDBOX} …")
            run_full_pipeline(total_tokens=total_tokens, pretrain_steps=steps,
                              sft_steps=int(os.environ.get("QENYA_SFT_STEPS", 400)),
                              vision_steps=int(os.environ.get("QENYA_VISION_STEPS", 200)),
                              with_net=True, log=log)
        self.q.put(("log", "[model] загружаю веса…"))
        eng = QenyaEngine()
        self.q.put(("ready", eng))

    # ------------------------------------------------------------ chat -----
    def pick_image(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if p:
            self.image_path = p
            self._append("sys", f"[приложена картинка: {p}]")

    def send(self):
        if self.busy or self.agent is None:
            return
        text = self.entry.get("1.0", "end").strip()
        if not text:
            return
        self.entry.delete("1.0", "end")
        self._append("user", "Вы: " + text)
        self.busy = True
        self.send_btn.state(["disabled"])
        images = None
        if self.image_path:
            from PIL import Image
            images = [Image.open(self.image_path)]
            self._append("sys", f"[режим распознавания: {self.image_path}]")
            self.image_path = None
        mode = self.mode_var.get()
        threading.Thread(target=self._ask_worker, args=(text, mode, images),
                         daemon=True).start()

    def _ask_worker(self, text, mode, images):
        t0 = time.time()
        try:
            res = self.agent.ask(text, mode_name=mode, images=images,
                                 on_delta=lambda d: self.q.put(("delta", d)))
            res["_secs"] = time.time() - t0
            self.q.put(("answer", res))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.q.put(("error", str(e)))

    def stop(self):
        self.busy = False  # генерация идёт в потоке; мягкая остановка по флагу
        self._append("sys", "[остановка запрошена]")

    # ------------------------------------------------------------ poll -----
    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.status_var.set(payload[:90])
                    self._append("sys", payload)
                elif kind == "progress":
                    a, b = payload
                    self.pbar["value"] = 100 * a / b
                elif kind == "ready":
                    self.engine = payload
                    from .agent import QenyaAgent
                    self.agent = QenyaAgent(self.engine)
                    meta = Path(self.engine.model_dir) / "meta.json"
                    info = ""
                    if meta.exists():
                        import json
                        m = json.loads(meta.read_text())
                        info = (f"{m['params_total']:,} локальных параметров | "
                                f"конфиг 5B = {m['arch_full_5b_params']/1e9:.2f}B")
                    self.status_var.set(f"готово ✓ ({self.engine.device}) {info}")
                    self.pbar["value"] = 100
                    self.refresh_files()
                elif kind == "delta":
                    self._stream(payload)
                elif kind == "answer":
                    self.busy = False
                    self.send_btn.state(["!disabled"])
                    ans = payload["answer"] or "(модель вернула пусто)"
                    self._finalize_stream(ans)
                    for line in payload["trace"]:
                        self.trace.insert("end", line + "\n")
                    self.trace.see("end")
                    self._append("sys", f"[режим {payload['mode']} · {payload['_secs']:.1f}s]")
                elif kind == "error":
                    self.busy = False
                    self.send_btn.state(["!disabled"])
                    self._append("sys", f"[ошибка] {payload}")
        except queue.Empty:
            pass
        self.root.after(60, self._poll)

    _streaming = False

    def _stream(self, d):
        if not self._streaming:
            self._streaming = True
            self.chat.configure(state="normal")
            self.chat.insert("end", "\nQenya: ", "bot")
        self.chat.insert("end", d, "bot")
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _finalize_stream(self, full):
        if self._streaming:
            # поток уже показал текст; просто добавим хвост если он длиннее
            self._streaming = False
            self.chat.configure(state="normal")
            self.chat.insert("end", "\n", "bot")
            self.chat.configure(state="disabled")
        else:
            self._append("bot", "Qenya: " + full)

    def _append(self, tag, text):
        self.chat.configure(state="normal")
        self.chat.insert("end", text.rstrip() + "\n\n", tag)
        self.chat.see("end")
        self.chat.configure(state="disabled")

    # ----------------------------------------------------------- tools -----
    def refresh_files(self):
        from .agent import SandboxTools
        self.files.delete("1.0", "end")
        self.files.insert("end", SandboxTools().list_files())

    def open_games(self):
        import webbrowser
        g = SANDBOX / "game"
        g.mkdir(parents=True, exist_ok=True)
        self._append("sys", f"папка песочницы: {SANDBOX}")

    def launch_selected(self):
        from .agent import SandboxTools
        sel = self.files.get("insert linestart", "insert lineend").strip()
        out = SandboxTools().launch_game(sel or "game/index.html")
        self.tools_out.delete("1.0", "end"); self.tools_out.insert("end", out)

    def run_tests(self):
        from .agent import SandboxTools
        out = SandboxTools().run_tests()
        self.tools_out.delete("1.0", "end"); self.tools_out.insert("end", out)

    def check_net(self):
        from .agent import SandboxTools
        try:
            r = SandboxTools().web_fetch("https://example.com", 600)
            self.tools_out.delete("1.0", "end")
            self.tools_out.insert("end", "Интернет есть: " + r[:300])
        except Exception as e:
            self.tools_out.delete("1.0", "end")
            self.tools_out.insert("end", f"Нет сети: {e}")

    def show_model_info(self):
        if not self.engine:
            return
        cfg = self.engine.llm.cfg
        from .model import QENYA_5B, QenyaConfig, count_params
        msg = (f"Локально развёрнуто: {self.engine.llm.n_params():,} парам.\n"
               f"Vision: {sum(p.numel() for p in self.engine.vision.parameters()):, } парам.\n"
               f"Архитектура Qenya-5B (полный конфиг): "
               f"{count_params(QenyaConfig(**QENYA_5B))/1e9:.2f}B парам.\n"
               f"Словарь BPE: {len(self.engine.bpe.vocab)}\n"
               f"Корпус-песочница: {CORPUS_DIR}\nУстройство: {self.engine.device}")
        messagebox.showinfo("Qenya-5B", msg)


def main():
    root = tk.Tk()
    QenyaApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

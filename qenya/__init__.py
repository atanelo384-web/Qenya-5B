"""
Qenya-5B — локальная нейросеть + desktop-агент.

Модули:
  qenya.model      — архитектура трансформера (LLM), конфигурации Qenya-5B / Qenya-Tiny
  qenya.vocab      — BPE-токенизатор + мультиязычная песочница-корпус
  qenya.train      — обучение с нуля (синтетика self-play, реальные данные при наличии сети)
  qenya.vision     — ViT-энкодер изображений + проекция в пространство LLM
  qenya.agent      — reasoning-режимы, инструменты (файлы/запуск/тесты/интернет/песочница)
  qenya.app        — Tkinter desktop-приложение (точка входа: python -m qenya.app)
"""
__version__ = "1.0.0"

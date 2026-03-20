"""
make_plots.py
=============
Построение графиков по результатам экспериментов.

Читает файлы:
  - out/results.csv
  - out/traces.csv
  - out/experiment_meta.json  (необязательно; для согласованного порога dx)

Создаёт папку out/plots/ и сохраняет PNG-файлы:
  1. convergence_mean.png  — средняя кривая best_f(iter) ± std по вариантам
  2. boxplot_best_f.png    — ящик с усами финального best_f по вариантам
  3. boxplot_dx.png        — ящик с усами dx по вариантам
  4. success_rate.png      — столбчатая диаграмма success_rate
  5. scatter_endpoints.png — scatter финальных точек (best_x, best_y) по вариантам

Запуск (открывает GUI-окно)::

    python make_plots.py
"""

import json
import os
import queue as _queue
import threading

import matplotlib
matplotlib.use("Agg")          # безголовый бэкенд (без GUI)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from objective import TRUE_MIN, TRUE_F

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------
RESULTS_CSV  = os.path.join("out", "results.csv")
TRACES_CSV   = os.path.join("out", "traces.csv")
META_JSON    = os.path.join("out", "experiment_meta.json")
PLOTS_DIR    = os.path.join("out", "plots")

# Значение по умолчанию; будет перезаписано из experiment_meta.json если файл есть
_DEFAULT_DX_THRESHOLD = 1.0

# Цветовая схема — по одному цвету на вариант
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f",
]


# ===========================================================================
# Загрузка порога из метаданных эксперимента
# ===========================================================================

def _load_threshold() -> float:
    """
    Попытаться загрузить SUCCESS_DX_THRESHOLD из experiment_meta.json.
    Если файл отсутствует или не содержит ключа — вернуть _DEFAULT_DX_THRESHOLD.
    """
    if os.path.exists(META_JSON):
        try:
            with open(META_JSON, encoding="utf-8") as fh:
                meta = json.load(fh)
            threshold = float(meta.get("SUCCESS_DX_THRESHOLD", _DEFAULT_DX_THRESHOLD))
            print(f"Загружен порог dx из {META_JSON}: {threshold}")
            return threshold
        except Exception as err:
            print(f"Предупреждение: не удалось прочитать {META_JSON}: {err}")
    else:
        print(f"Файл {META_JSON} не найден; используется порог по умолчанию "
              f"{_DEFAULT_DX_THRESHOLD}.")
    return _DEFAULT_DX_THRESHOLD


# ===========================================================================
# Вспомогательные утилиты
# ===========================================================================

def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузить results.csv и traces.csv; завершить с ошибкой, если нет файлов."""
    if not os.path.exists(RESULTS_CSV):
        sys.exit(
            f"Файл '{RESULTS_CSV}' не найден. "
            "Сначала запустите run_experiments.py"
        )
    if not os.path.exists(TRACES_CSV):
        print(
            f"Предупреждение: файл '{TRACES_CSV}' не найден. "
            "Графики сходимости строиться не будут."
        )
        results = pd.read_csv(RESULTS_CSV)
        return results, pd.DataFrame()

    results = pd.read_csv(RESULTS_CSV)
    traces  = pd.read_csv(TRACES_CSV)
    return results, traces


def _color_map(variants: list[str]) -> dict:
    """Назначить цвет каждому варианту."""
    return {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(variants)}


def _save_fig(fig: plt.Figure, name: str) -> None:
    """Сохранить фигуру в PLOTS_DIR/name.png и закрыть."""
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Сохранён: {path}")


# ===========================================================================
# 1. Кривые сходимости
# ===========================================================================

def plot_convergence(traces: pd.DataFrame, colors: dict) -> None:
    """
    Построить средние кривые сходимости best_f(iter) с полосой ±std.

    По оси X — номер итерации/поколения.
    По оси Y — лучшее найденное значение best_f.
    """
    variants = traces["variant"].unique()
    fig, ax  = plt.subplots(figsize=(10, 6))

    for variant in variants:
        sub = traces[traces["variant"] == variant]
        # Агрегировать по iter: mean и std
        agg = sub.groupby("iter")["best_f"].agg(["mean", "std"]).reset_index()
        x   = agg["iter"].values
        y   = agg["mean"].values
        s   = agg["std"].values

        color = colors.get(variant, "#333333")
        ax.plot(x, y, label=variant, color=color, linewidth=1.8)
        ax.fill_between(x, y - s, y + s, alpha=0.15, color=color)

    # Горизонтальная линия — истинный минимум
    ax.axhline(TRUE_F, color="black", linestyle="--", linewidth=1.2,
               label=f"f* = {TRUE_F:.2f}")

    ax.set_xlabel("Итерация / Поколение", fontsize=12)
    ax.set_ylabel("Лучшее значение f", fontsize=12)
    ax.set_title("Средняя кривая сходимости (± std)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, "convergence_mean.png")


# ===========================================================================
# 2. Boxplot финального best_f
# ===========================================================================

def plot_boxplot_best_f(results: pd.DataFrame, colors: dict) -> None:
    """
    Ящик с усами финального значения best_f по вариантам.

    Позволяет сравнить качество решения и разброс между запусками.
    """
    variants = results["variant"].unique()
    data     = [results[results["variant"] == v]["best_f"].values for v in variants]

    fig, ax  = plt.subplots(figsize=(max(6, len(variants) * 1.4), 6))
    bp       = ax.boxplot(data, patch_artist=True, notch=False,
                          medianprops=dict(color="black", linewidth=2))

    for patch, variant in zip(bp["boxes"], variants):
        patch.set_facecolor(colors.get(variant, "#aaaaaa"))
        patch.set_alpha(0.7)

    ax.axhline(TRUE_F, color="red", linestyle="--", linewidth=1.2,
               label=f"f* = {TRUE_F:.2f}")
    ax.set_xticks(range(1, len(variants) + 1))
    ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Финальное best_f", fontsize=12)
    ax.set_title("Распределение финального значения f по вариантам", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    _save_fig(fig, "boxplot_best_f.png")


# ===========================================================================
# 3. Boxplot dx
# ===========================================================================

def plot_boxplot_dx(results: pd.DataFrame, colors: dict, threshold: float) -> None:
    """
    Ящик с усами расстояния до истинного минимума (dx) по вариантам.
    """
    variants = results["variant"].unique()
    data     = [results[results["variant"] == v]["dx"].values for v in variants]

    fig, ax  = plt.subplots(figsize=(max(6, len(variants) * 1.4), 6))
    bp       = ax.boxplot(data, patch_artist=True, notch=False,
                          medianprops=dict(color="black", linewidth=2))

    for patch, variant in zip(bp["boxes"], variants):
        patch.set_facecolor(colors.get(variant, "#aaaaaa"))
        patch.set_alpha(0.7)

    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
               label=f"Порог успеха dx < {threshold}")
    ax.set_xticks(range(1, len(variants) + 1))
    ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("dx (расстояние до глобального минимума)", fontsize=12)
    ax.set_title("Точность нахождения глобального минимума (dx)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    _save_fig(fig, "boxplot_dx.png")


# ===========================================================================
# 4. Столбчатая диаграмма success_rate
# ===========================================================================

def plot_success_rate(results: pd.DataFrame, colors: dict, threshold: float) -> None:
    """
    Столбчатая диаграмма доли успешных запусков (dx < порог) по вариантам.
    """
    variants = results["variant"].unique()
    rates    = [
        (results[results["variant"] == v]["dx"] < threshold).mean()
        for v in variants
    ]

    fig, ax = plt.subplots(figsize=(max(6, len(variants) * 1.4), 5))
    bar_colors = [colors.get(v, "#aaaaaa") for v in variants]
    bars = ax.bar(variants, rates, color=bar_colors, edgecolor="black",
                  linewidth=0.8, alpha=0.8)

    # Подписать значения над столбцами
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{rate:.2%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.12)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(f"Доля успехов (dx < {threshold})", fontsize=12)
    ax.set_title("Вероятность нахождения глобального минимума", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    _save_fig(fig, "success_rate.png")


# ===========================================================================
# 5. Scatter финальных точек
# ===========================================================================

def plot_scatter_endpoints(results: pd.DataFrame, colors: dict) -> None:
    """
    Scatter-диаграмма финальных точек (best_x, best_y) по вариантам.

    Позволяет визуально оценить, насколько алгоритмы находят
    глобальный минимум (отмечен звёздочкой).
    """
    variants = results["variant"].unique()
    fig, ax  = plt.subplots(figsize=(9, 7))

    for variant in variants:
        sub   = results[results["variant"] == variant]
        color = colors.get(variant, "#aaaaaa")
        ax.scatter(sub["best_x"], sub["best_y"],
                   label=variant, color=color, alpha=0.6, s=40, edgecolors="none")

    # Истинный минимум
    ax.scatter(*TRUE_MIN, marker="*", s=300, color="red", zorder=5,
               label=f"f* ({TRUE_MIN[0]}, {TRUE_MIN[1]:.2f})")

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Финальные точки по вариантам (все повторения)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    _save_fig(fig, "scatter_endpoints.png")


# ===========================================================================
# GUI
# ===========================================================================

def main():
    """Точка входа при запуске `python make_plots.py` — открывает GUI-окно."""
    q: _queue.Queue = _queue.Queue()

    root = tk.Tk()
    root.title("Построение графиков")
    root.resizable(False, False)

    # ---- Файлы ввода ----
    ff = ttk.LabelFrame(root, text="Файлы данных", padding=8)
    ff.grid(row=0, column=0, padx=10, pady=8, sticky="ew")

    def _file_row(parent, row, label, default):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w",
                                           padx=4, pady=2)
        var = tk.StringVar(value=default)
        ttk.Entry(parent, textvariable=var, width=38).grid(row=row, column=1,
                                                            sticky="w", padx=4)
        def _browse():
            path = filedialog.askopenfilename(
                title=f"Выбрать {label}",
                filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")],
                initialfile=default,
            )
            if path:
                var.set(path)
        ttk.Button(parent, text="…", width=3,
                   command=_browse).grid(row=row, column=2, padx=2)
        return var

    results_var = _file_row(ff, 0, "results.csv:", RESULTS_CSV)
    traces_var  = _file_row(ff, 1, "traces.csv:",  TRACES_CSV)

    # ---- Настройки ----
    sf = ttk.LabelFrame(root, text="Настройки", padding=8)
    sf.grid(row=1, column=0, padx=10, pady=4, sticky="ew")

    ttk.Label(sf, text="Порог успеха dx:").grid(row=0, column=0, sticky="w",
                                                 padx=4, pady=2)
    dx_var = tk.StringVar(value=str(_load_threshold()))
    ttk.Entry(sf, textvariable=dx_var, width=12).grid(row=0, column=1,
                                                       sticky="w", padx=4)

    # ---- Управление ----
    cf = ttk.Frame(root, padding=8)
    cf.grid(row=2, column=0, sticky="ew")

    build_btn = ttk.Button(cf, text="Построить графики")
    build_btn.grid(row=0, column=0, padx=4, pady=4)

    status_var = tk.StringVar(value="Готово")
    ttk.Label(cf, textvariable=status_var,
              font=("", 10, "bold")).grid(row=1, column=0, sticky="w", padx=4)

    # ---- Список файлов ----
    lf = ttk.LabelFrame(root, text="Созданные файлы", padding=8)
    lf.grid(row=3, column=0, padx=10, pady=8, sticky="nsew")

    files_txt = tk.Text(lf, height=6, width=56, state="disabled",
                        font=("Courier", 9))
    files_txt.grid(row=0, column=0, sticky="nsew")

    def _show_files(paths: list[str]):
        files_txt.config(state="normal")
        files_txt.delete("1.0", "end")
        for p in paths:
            files_txt.insert("end", p + "\n")
        files_txt.config(state="disabled")

    def _worker():
        try:
            threshold = float(dx_var.get())
        except ValueError as exc:
            q.put(("error", f"Ошибка в параметрах: {exc}"))
            return

        results_path = results_var.get()
        traces_path  = traces_var.get()

        if not os.path.exists(results_path):
            q.put(("error",
                   f"Файл '{results_path}' не найден.\n"
                   "Сначала запустите run_experiments.py"))
            return

        try:
            results = pd.read_csv(results_path)
            traces  = (pd.read_csv(traces_path)
                       if os.path.exists(traces_path) else pd.DataFrame())

            os.makedirs(PLOTS_DIR, exist_ok=True)
            variants = list(results["variant"].unique())
            colors   = _color_map(variants)

            if not traces.empty:
                plot_convergence(traces, colors)
            plot_boxplot_best_f(results, colors)
            plot_boxplot_dx(results, colors, threshold)
            plot_success_rate(results, colors, threshold)
            plot_scatter_endpoints(results, colors)

            saved = sorted(
                os.path.join(PLOTS_DIR, f)
                for f in os.listdir(PLOTS_DIR)
                if f.endswith(".png")
            )
            q.put(("done", saved))
        except Exception as exc:
            q.put(("error", str(exc)))

    def _start():
        build_btn.config(state="disabled")
        status_var.set("Выполняется…")
        _show_files([])
        threading.Thread(target=_worker, daemon=True).start()
        root.after(100, _poll)

    def _poll():
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == "done":
                    status_var.set("Готово")
                    _show_files(msg[1])
                    build_btn.config(state="normal")
                    return
                elif msg[0] == "error":
                    status_var.set("Ошибка")
                    messagebox.showerror("Ошибка", msg[1])
                    build_btn.config(state="normal")
                    return
        except _queue.Empty:
            pass
        root.after(100, _poll)

    build_btn.config(command=_start)
    root.mainloop()


if __name__ == "__main__":
    main()


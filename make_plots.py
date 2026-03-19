"""
make_plots.py
=============
Построение графиков по результатам экспериментов.

Читает файлы:
  - out/results.csv
  - out/traces.csv

Создаёт папку out/plots/ и сохраняет PNG-файлы:
  1. convergence_mean.png  — средняя кривая best_f(iter) ± std по вариантам
  2. boxplot_best_f.png    — ящик с усами финального best_f по вариантам
  3. boxplot_dx.png        — ящик с усами dx по вариантам
  4. success_rate.png      — столбчатая диаграмма success_rate
  5. scatter_endpoints.png — scatter финальных точек (best_x, best_y) по вариантам

Запуск::

    python make_plots.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")          # безголовый бэкенд (без GUI)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from objective import TRUE_MIN, TRUE_F

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------
RESULTS_CSV  = os.path.join("out", "results.csv")
TRACES_CSV   = os.path.join("out", "traces.csv")
PLOTS_DIR    = os.path.join("out", "plots")

# Порог «успеха» должен совпадать с run_experiments.py
SUCCESS_DX_THRESHOLD = 1.0

# Цветовая схема — по одному цвету на вариант
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f",
]


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
        sys.exit(
            f"Файл '{TRACES_CSV}' не найден. "
            "Сначала запустите run_experiments.py"
        )

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

def plot_boxplot_dx(results: pd.DataFrame, colors: dict) -> None:
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

    ax.axhline(SUCCESS_DX_THRESHOLD, color="red", linestyle="--", linewidth=1.2,
               label=f"Порог успеха dx < {SUCCESS_DX_THRESHOLD}")
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

def plot_success_rate(results: pd.DataFrame, colors: dict) -> None:
    """
    Столбчатая диаграмма доли успешных запусков (dx < порог) по вариантам.
    """
    variants = results["variant"].unique()
    rates    = [
        (results[results["variant"] == v]["dx"] < SUCCESS_DX_THRESHOLD).mean()
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
    ax.set_ylabel(f"Доля успехов (dx < {SUCCESS_DX_THRESHOLD})", fontsize=12)
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
# Точка входа
# ===========================================================================

def main():
    """Загрузить данные и построить все графики."""

    print("=== Построение графиков ===")

    results, traces = _load_data()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Собрать список уникальных вариантов в порядке появления
    variants = list(results["variant"].unique())
    colors   = _color_map(variants)

    print(f"Найдено {len(results)} запусков, {len(variants)} вариантов:")
    for v in variants:
        n = (results["variant"] == v).sum()
        print(f"  {v}: {n} запусков")
    print()

    plot_convergence(traces, colors)
    plot_boxplot_best_f(results, colors)
    plot_boxplot_dx(results, colors)
    plot_success_rate(results, colors)
    plot_scatter_endpoints(results, colors)

    print("\nВсе графики сохранены в:", PLOTS_DIR)


if __name__ == "__main__":
    main()

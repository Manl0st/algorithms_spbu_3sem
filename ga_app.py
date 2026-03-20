"""
ga_app.py
=========
Генетический алгоритм (GA) для минимизации функции Eggholder.

Поддерживает два способа кодирования генотипа:
  - binary  : вещественные переменные кодируются цепочкой битов
  - real    : вещественные переменные хранятся напрямую

Как пользоваться
----------------
Запустите GUI::

    python ga_app.py

В окне настройте параметры и нажмите «Запустить».
Результаты сохраняются в папке out/:
  - ga_last_result.json — параметры и итоговые метрики
  - ga_last_trace.csv   — трасса сходимости по поколениям (с русскими заголовками)

Для использования из run_experiments.py доступна функция::

    result, trace_df = run_ga(params: dict)
"""

import json
import math
import os
import queue as _queue
import threading
import time

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

from objective import BOUNDS, TRUE_F, TRUE_MIN, CounterObjective

# ---------------------------------------------------------------------------
# Вспомогательные функции общего назначения
# ---------------------------------------------------------------------------

def _clip(value: float, lo: float, hi: float) -> float:
    """Ограничить value отрезком [lo, hi]."""
    return max(lo, min(hi, value))


def _euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    """Евклидово расстояние между двумя точками 2D."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ===========================================================================
# Бинарное кодирование
# ===========================================================================

def _encode_binary(x: float, y: float, bits: int) -> np.ndarray:
    """
    Закодировать пару (x, y) как двоичную хромосому длиной 2*bits.

    Вещественное значение v ∈ [lo, hi] отображается в целое число
    i = round((v - lo) / (hi - lo) * (2^bits - 1)), затем записывается
    в виде bits бит (MSB first).

    Для переменной x используются BOUNDS[0], для y — BOUNDS[1].
    """
    chromosome = np.empty(2 * bits, dtype=np.int8)
    max_int = (1 << bits) - 1

    # Перебираем переменные: idx=0 → x (BOUNDS[0]), idx=1 → y (BOUNDS[1])
    for var_idx, v in enumerate((x, y)):
        lo, hi = BOUNDS[var_idx]
        i = round((v - lo) / (hi - lo) * max_int)
        i = max(0, min(max_int, i))
        for bit_idx in range(bits):
            chromosome[var_idx * bits + (bits - 1 - bit_idx)] = (i >> bit_idx) & 1

    return chromosome


def _decode_binary(chromosome: np.ndarray, bits: int) -> tuple[float, float]:
    """
    Декодировать двоичную хромосому обратно в пару (x, y).

    Для переменной x используются BOUNDS[0], для y — BOUNDS[1].
    """
    max_int = (1 << bits) - 1
    result = []

    for var_idx in range(2):
        lo, hi = BOUNDS[var_idx]
        segment = chromosome[var_idx * bits : (var_idx + 1) * bits]
        integer = 0
        for bit in segment:
            integer = (integer << 1) | int(bit)
        v = lo + (hi - lo) * integer / max_int
        result.append(v)

    return result[0], result[1]


# ===========================================================================
# Популяция
# ===========================================================================

def _init_population(pop_size: int, encoding: str, bits: int,
                     rng: np.random.Generator) -> list:
    """
    Создать начальную популяцию.

    Для binary — случайные двоичные векторы.
    Для real   — случайные вещественные пары (x, y) в BOUNDS, раздельно по каждой переменной.
    """
    population = []

    for _ in range(pop_size):
        if encoding == "binary":
            chrom = rng.integers(0, 2, size=2 * bits, dtype=np.int8)
        else:  # real — FIX: использовать границы каждой переменной отдельно
            x_val = rng.uniform(BOUNDS[0][0], BOUNDS[0][1])
            y_val = rng.uniform(BOUNDS[1][0], BOUNDS[1][1])
            chrom = np.array([x_val, y_val], dtype=float)
        population.append(chrom)

    return population


def _phenotype(chromosome: np.ndarray, encoding: str, bits: int) -> tuple[float, float]:
    """
    Получить фенотип (x, y) из хромосомы.
    """
    if encoding == "binary":
        return _decode_binary(chromosome, bits)
    else:
        return float(chromosome[0]), float(chromosome[1])


# ===========================================================================
# Оценка популяции
# ===========================================================================

def _evaluate(population: list, encoding: str, bits: int,
               obj: CounterObjective) -> np.ndarray:
    """
    Вычислить значения целевой функции для всей популяции.

    Возвращает массив fitness значений (одно на особь).
    """
    fitness = np.empty(len(population))
    for i, chrom in enumerate(population):
        x, y = _phenotype(chrom, encoding, bits)
        fitness[i] = obj(x, y)
    return fitness


# ===========================================================================
# Турнирная селекция
# ===========================================================================

def _tournament_select(fitness: np.ndarray, k: int,
                        rng: np.random.Generator) -> int:
    """
    Выбрать индекс лучшей особи из k случайных кандидатов (минимизация).

    Параметры
    ----------
    fitness : ndarray
        Массив значений приспособленности.
    k : int
        Размер турнира.
    rng : Generator
        Генератор псевдослучайных чисел.

    Возвращает
    ----------
    int
        Индекс победителя турнира.
    """
    candidates = rng.choice(len(fitness), size=k, replace=False)
    best = candidates[np.argmin(fitness[candidates])]
    return int(best)


# ===========================================================================
# Операторы кроссинговера
# ===========================================================================

def _crossover_binary(parent1: np.ndarray, parent2: np.ndarray,
                      ctype: str, rate: float,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Бинарный кроссинговер.

    ctype ∈ {"one_point", "two_point", "uniform"}
    """
    if rng.random() > rate:
        # нет кроссинговера — возвращаем копии родителей
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    c1, c2 = parent1.copy(), parent2.copy()

    if ctype == "one_point":
        # одна точка разрыва
        pt = int(rng.integers(1, n))
        c1[pt:], c2[pt:] = parent2[pt:].copy(), parent1[pt:].copy()

    elif ctype == "two_point":
        # две точки разрыва
        pts = sorted(rng.choice(range(1, n), size=2, replace=False))
        a, b = pts[0], pts[1]
        c1[a:b], c2[a:b] = parent2[a:b].copy(), parent1[a:b].copy()

    elif ctype == "uniform":
        # каждый бит обменивается с вероятностью 0.5
        mask = rng.integers(0, 2, size=n, dtype=bool)
        c1[mask], c2[mask] = parent2[mask].copy(), parent1[mask].copy()

    return c1, c2


def _crossover_real(parent1: np.ndarray, parent2: np.ndarray,
                    rate: float,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Арифметический кроссинговер для вещественного кодирования.

    Потомки:
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
    где alpha ~ Uniform(0, 1).
    """
    if rng.random() > rate:
        return parent1.copy(), parent2.copy()

    alpha = rng.random()
    c1 = alpha * parent1 + (1.0 - alpha) * parent2
    c2 = (1.0 - alpha) * parent1 + alpha * parent2
    return c1, c2


# ===========================================================================
# Операторы мутации
# ===========================================================================

def _mutate_binary(chromosome: np.ndarray, rate: float,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Побитовая мутация (bit-flip): каждый бит инвертируется с вероятностью rate.
    """
    chrom = chromosome.copy()
    mask = rng.random(len(chrom)) < rate
    chrom[mask] = 1 - chrom[mask]
    return chrom


def _mutate_real(chromosome: np.ndarray, rate: float, sigma: float,
                 rng: np.random.Generator) -> np.ndarray:
    """
    Гауссова мутация для вещественного кодирования.

    Каждый ген мутирует с вероятностью rate, добавляя Gaussian(0, sigma).
    После мутации значения клипируются в соответствующие BOUNDS каждой переменной.
    """
    chrom = chromosome.copy()
    for i in range(len(chrom)):
        lo, hi = BOUNDS[i]  # FIX: раздельные границы для x (i=0) и y (i=1)
        if rng.random() < rate:
            chrom[i] += rng.normal(0.0, sigma)
            chrom[i] = _clip(chrom[i], lo, hi)
    return chrom


# ===========================================================================
# Основной алгоритм ГА
# ===========================================================================

def run_ga(params: dict, callback=None) -> tuple[dict, pd.DataFrame]:
    """
    Запустить генетический алгоритм с заданными параметрами.

    Параметры (ключи словаря params)
    ---------------------------------
    encoding        : str   — "binary" или "real"
    bits_per_var    : int   — биты на переменную (только для binary)
    pop_size        : int   — размер популяции
    generations     : int   — число поколений
    crossover_type  : str   — тип кроссинговера
    crossover_rate  : float — вероятность кроссинговера
    mutation_rate   : float — вероятность мутации (на бит или на ген)
    tournament_k    : int   — размер турнира
    elitism         : int   — число элитных особей
    seed            : int   — начальное значение ГСЧ
    stop_eps_f      : float — останов, если улучшение best_f < eps_f (порог по функции)
    stop_eps_dx     : float — останов, если best_dx < eps_dx (0 = отключён)
    no_improve_patience : int — останов, если нет улучшения N поколений
    viz_every       : int   — передавать позиции в callback каждые N поколений (0 = никогда)
    callback : callable, optional
        Вызывается на каждом поколении:
          callback(gen, total_gens, best_f, positions_xy=None, best_xy=None)
        positions_xy — np.array (pop_size, 2) фенотипов (только при viz_every)
        best_xy      — np.array [best_x, best_y]
        Используется GUI для отображения прогресса и визуализации.

    Возвращает
    ----------
    result : dict
        Итоговые метрики: best_x, best_y, best_f, dx, df, evals, time_sec и т.д.
    trace_df : pd.DataFrame
        Статистика по поколениям (внутренние имена столбцов).
    """

    # -- параметры с дефолтами -----------------------------------------------
    enc          = params.get("encoding",            "binary")
    bits         = int(params.get("bits_per_var",    20))
    pop_size     = int(params.get("pop_size",        50))
    generations  = int(params.get("generations",     300))
    cx_type      = params.get("crossover_type",      "one_point" if enc == "binary" else "arithmetic")
    cx_rate      = float(params.get("crossover_rate",0.9))
    mut_rate     = float(params.get("mutation_rate", 0.02))
    tourn_k      = int(params.get("tournament_k",    3))
    elitism      = int(params.get("elitism",         1))
    seed         = int(params.get("seed",            42))
    eps_f        = float(params.get("stop_eps_f",    1e-6))   # порог улучшения функции
    eps_dx       = float(params.get("stop_eps_dx",   0.0))    # порог dx (0 = отключён)
    patience     = int(params.get("no_improve_patience", 80))
    viz_every    = int(params.get("viz_every",       0))      # 0 = не передавать позиции

    # Для вещественного кодирования поддерживается только арифметический кроссинговер
    if enc == "real" and cx_type != "arithmetic":
        cx_type = "arithmetic"

    # sigma для гауссовой мутации (real): 5% от диапазона переменной x
    lo0, hi0 = BOUNDS[0]
    sigma = (hi0 - lo0) * 0.05

    # -- инициализация --------------------------------------------------------
    rng = np.random.default_rng(seed)
    obj = CounterObjective()
    t_start = time.perf_counter()

    population = _init_population(pop_size, enc, bits, rng)
    fitness    = _evaluate(population, enc, bits, obj)

    # -- история --------------------------------------------------------------
    trace_rows = []          # строки для CSV-трассы
    best_f_prev = np.inf     # для критерия останова по патциенции
    no_improve_cnt = 0       # счётчик поколений без улучшения

    # Границы для клиппинга вещественных потомков (per-variable)
    _lo_arr = np.array([b[0] for b in BOUNDS])
    _hi_arr = np.array([b[1] for b in BOUNDS])

    # -- главный цикл ---------------------------------------------------------
    for gen in range(generations):

        # 1. Статистика текущего поколения
        best_idx = int(np.argmin(fitness))
        best_f   = float(fitness[best_idx])
        mean_f   = float(np.mean(fitness))
        bx, by   = _phenotype(population[best_idx], enc, bits)
        best_dx  = _euclidean(bx, by, TRUE_MIN[0], TRUE_MIN[1])

        trace_rows.append({
            "gen":          gen,
            "best_f":       best_f,
            "mean_f":       mean_f,
            "best_dx":      best_dx,
            "best_x":       bx,
            "best_y":       by,
            "evals_so_far": obj.evals,
        })

        # Подготовить данные для визуализации (только при viz_every)
        positions_xy = None
        best_xy_arr  = None
        if callback is not None and viz_every > 0 and gen % viz_every == 0:
            positions_xy = np.array([_phenotype(p, enc, bits) for p in population])
            best_xy_arr  = np.array([bx, by])

        if callback is not None:
            callback(gen, generations, best_f, positions_xy, best_xy_arr)

        # 2. Критерий останова по улучшению функции.
        if best_f < best_f_prev - eps_f:
            best_f_prev    = best_f
            no_improve_cnt = 0
        else:
            no_improve_cnt += 1

        if no_improve_cnt >= patience:
            break

        # 3. Критерий останова по близости к истинному минимуму (если включён).
        if eps_dx > 0.0 and best_dx < eps_dx:
            break

        # 4. Элитизм: сохранить лучших особей
        elite_idxs = np.argsort(fitness)[:elitism]
        elites     = [population[i].copy() for i in elite_idxs]

        # 5. Формирование нового поколения
        new_population = []

        while len(new_population) < pop_size - elitism:
            # --- турнирная селекция ---
            idx1 = _tournament_select(fitness, tourn_k, rng)
            idx2 = _tournament_select(fitness, tourn_k, rng)
            p1   = population[idx1]
            p2   = population[idx2]

            # --- кроссинговер ---
            if enc == "binary":
                c1, c2 = _crossover_binary(p1, p2, cx_type, cx_rate, rng)
            else:
                c1, c2 = _crossover_real(p1, p2, cx_rate, rng)
                c1 = np.clip(c1, _lo_arr, _hi_arr)
                c2 = np.clip(c2, _lo_arr, _hi_arr)

            # --- мутация ---
            if enc == "binary":
                c1 = _mutate_binary(c1, mut_rate, rng)
                c2 = _mutate_binary(c2, mut_rate, rng)
            else:
                c1 = _mutate_real(c1, mut_rate, sigma, rng)
                c2 = _mutate_real(c2, mut_rate, sigma, rng)

            new_population.append(c1)
            if len(new_population) < pop_size - elitism:
                new_population.append(c2)

        # 6. Добавить элиты в начало нового поколения
        population = elites + new_population
        fitness    = _evaluate(population, enc, bits, obj)

    # -- финальный результат --------------------------------------------------
    best_idx = int(np.argmin(fitness))
    bx, by   = _phenotype(population[best_idx], enc, bits)
    best_f   = float(fitness[best_idx])
    best_dx  = _euclidean(bx, by, TRUE_MIN[0], TRUE_MIN[1])
    best_df  = best_f - TRUE_F
    elapsed  = time.perf_counter() - t_start

    result = {
        # параметры
        "encoding":       enc,
        "bits_per_var":   bits,
        "pop_size":       pop_size,
        "generations":    generations,
        "crossover_type": cx_type,
        "crossover_rate": cx_rate,
        "mutation_rate":  mut_rate,
        "tournament_k":   tourn_k,
        "elitism":        elitism,
        "seed":           seed,
        # результаты
        "best_x":    bx,
        "best_y":    by,
        "best_f":    best_f,
        "dx":        best_dx,
        "df":        best_df,
        "evals":     obj.evals,
        "gens_done": len(trace_rows),
        "time_sec":  elapsed,
    }

    trace_df = pd.DataFrame(trace_rows)
    return result, trace_df


# ===========================================================================
# GUI
# ===========================================================================

# Словари для перевода между человекочитаемыми именами и внутренними значениями
_ENC_LABELS     = {"Бинарное": "binary", "Вещественное": "real"}
_ENC_LABELS_INV = {v: k for k, v in _ENC_LABELS.items()}

_CX_BIN_LABELS  = {
    "Одноточечный":  "one_point",
    "Двухточечный":  "two_point",
    "Равномерный":   "uniform",
}
_CX_BIN_INV = {v: k for k, v in _CX_BIN_LABELS.items()}

_CX_REAL_LABELS = {"Арифметический": "arithmetic"}
_CX_REAL_INV    = {v: k for k, v in _CX_REAL_LABELS.items()}

# Размер холста визуализации (пиксели)
_VIZ_SIZE = 420


def _world_to_canvas(x: float, y: float, size: int = _VIZ_SIZE) -> tuple[int, int]:
    """Перевести мировые координаты (x, y) в координаты холста."""
    bx_lo, bx_hi = BOUNDS[0]
    by_lo, by_hi = BOUNDS[1]
    cx = int((x - bx_lo) / (bx_hi - bx_lo) * (size - 1))
    cy = int((1.0 - (y - by_lo) / (by_hi - by_lo)) * (size - 1))  # Y перевёрнут
    return max(0, min(size - 1, cx)), max(0, min(size - 1, cy))


def main():
    """Точка входа при запуске `python ga_app.py` — открывает GUI-окно."""
    q: _queue.Queue = _queue.Queue()

    root = tk.Tk()
    root.title("ГА — Eggholder")
    root.resizable(False, False)

    # =========================================================================
    # Переменные параметров (только ключевые)
    # =========================================================================
    enc_var = tk.StringVar(value="Бинарное")
    cx_var  = tk.StringVar(value="Одноточечный")
    tk_var  = tk.StringVar(value="3")
    eli_var = tk.StringVar(value="1")
    pop_var = tk.StringVar(value="50")
    gen_var = tk.StringVar(value="300")
    cxr_var = tk.StringVar(value="0.9")
    mut_var = tk.StringVar(value="0.02")

    # =========================================================================
    # Основной контейнер (2 колонки: параметры + визуализация)
    # =========================================================================
    main_frame = ttk.Frame(root)
    main_frame.grid(row=0, column=0, sticky="nsew")

    # ----- Левая колонка: параметры -----
    pf = ttk.LabelFrame(main_frame, text="Параметры ГА", padding=8)
    pf.grid(row=0, column=0, padx=(10, 5), pady=8, sticky="nw")

    def _field(parent, row, label, var, choices=None, width=18):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=4, pady=3)
        if choices:
            wgt = ttk.Combobox(parent, textvariable=var, values=choices,
                               state="readonly", width=width)
        else:
            wgt = ttk.Entry(parent, textvariable=var, width=width + 2)
        wgt.grid(row=row, column=1, sticky="w", padx=4, pady=3)
        return wgt

    _field(pf, 0, "Тип кодирования:",    enc_var, list(_ENC_LABELS.keys()))
    cx_cb = _field(pf, 1, "Тип кроссинговера:", cx_var,
                   list(_CX_BIN_LABELS.keys()))
    _field(pf, 2, "Размер турнира:",      tk_var)
    _field(pf, 3, "Элитизм:",             eli_var)
    _field(pf, 4, "Размер популяции:",    pop_var)
    _field(pf, 5, "Число поколений:",     gen_var)
    _field(pf, 6, "Вер. кроссинговера:",  cxr_var)
    _field(pf, 7, "Вер. мутации:",        mut_var)

    def _on_enc(*_):
        if enc_var.get() == "Вещественное":
            cx_var.set("Арифметический")
            cx_cb.config(state="disabled", values=list(_CX_REAL_LABELS.keys()))
        else:
            cx_cb.config(state="readonly", values=list(_CX_BIN_LABELS.keys()))
            if cx_var.get() not in _CX_BIN_LABELS:
                cx_var.set("Одноточечный")

    enc_var.trace_add("write", _on_enc)

    # ----- Правая колонка: визуализация -----
    vf = ttk.LabelFrame(main_frame, text="Визуализация популяции", padding=4)
    vf.grid(row=0, column=1, padx=(5, 10), pady=8, sticky="n")

    viz_canvas = tk.Canvas(vf, width=_VIZ_SIZE, height=_VIZ_SIZE,
                           bg="#1a1a2e", highlightthickness=1,
                           highlightbackground="#555555")
    viz_canvas.grid(row=0, column=0, sticky="nsew")

    # Истинный минимум (рисуется один раз)
    _TRUE_CX, _TRUE_CY = _world_to_canvas(TRUE_MIN[0], TRUE_MIN[1])
    viz_canvas.create_text(_TRUE_CX, _TRUE_CY, text="★",
                            fill="#ff4444", font=("", 14, "bold"), tags="true_min")

    # Легенда (минимальная)
    ttk.Label(vf,
              text="● Синие: особи   ● Оранжевая: лучший   ★ Красная: минимум",
              font=("", 8), foreground="#555555").grid(
        row=1, column=0, sticky="w", padx=2, pady=(2, 0))

    # Живая статистика
    info_var = tk.StringVar(value="")
    ttk.Label(vf, textvariable=info_var, font=("Courier", 9),
              foreground="#225599").grid(row=2, column=0, sticky="w",
                                         padx=4, pady=(2, 4))

    # Буфер следа лучшей особи (последние N точек)
    _best_trail: list = []
    _TRAIL_LEN = 30

    def _redraw_viz(gen: int, total: int, best_f: float,
                    positions_xy: np.ndarray, best_xy: np.ndarray):
        """Перерисовать холст с текущей популяцией, следом и статистикой."""
        viz_canvas.delete("pop")

        if best_xy is not None:
            tx, ty = _world_to_canvas(best_xy[0], best_xy[1])
            _best_trail.append((tx, ty))
            if len(_best_trail) > _TRAIL_LEN:
                del _best_trail[:-_TRAIL_LEN]

        n = len(_best_trail)
        for i, (tx, ty) in enumerate(_best_trail):
            frac = (i + 1) / max(n, 1)
            g = int(60 + frac * 155)
            trail_color = f"#00{g:02x}44"
            r_t = max(1, round(frac * 3))
            viz_canvas.create_oval(tx - r_t, ty - r_t, tx + r_t, ty + r_t,
                                   fill=trail_color, outline="", tags="pop")

        if positions_xy is not None:
            for xi, yi in positions_xy:
                cx, cy = _world_to_canvas(xi, yi)
                viz_canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                                       fill="#4488cc", outline="", tags="pop")

        if best_xy is not None:
            bx_c, by_c = _world_to_canvas(best_xy[0], best_xy[1])
            viz_canvas.create_oval(bx_c - 6, by_c - 6, bx_c + 6, by_c + 6,
                                   fill="#ff8800", outline="#ffffff", width=1,
                                   tags="pop")

        viz_canvas.tag_raise("true_min")

        if best_xy is not None:
            info_var.set(
                f"Поколение: {gen + 1} / {total}\n"
                f"Лучшее f: {best_f:.4f}\n"
                f"Лучшая точка: ({best_xy[0]:.2f}, {best_xy[1]:.2f})"
            )

    def _clear_viz():
        _best_trail.clear()
        viz_canvas.delete("pop")
        info_var.set("")

    # =========================================================================
    # Управление
    # =========================================================================
    cf = ttk.Frame(root, padding=8)
    cf.grid(row=1, column=0, sticky="ew")

    run_btn = ttk.Button(cf, text="▶  Запустить")
    run_btn.grid(row=0, column=0, padx=4, pady=4)

    status_var = tk.StringVar(value="Готово")
    ttk.Label(cf, textvariable=status_var,
              font=("", 10, "bold")).grid(row=0, column=1, sticky="w", padx=8)

    pb_var = tk.IntVar(value=0)
    ttk.Progressbar(cf, variable=pb_var, maximum=100,
                    length=400, mode="determinate").grid(
        row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=2)

    pb_lbl = tk.StringVar(value="")
    ttk.Label(cf, textvariable=pb_lbl, font=("", 8)).grid(
        row=2, column=0, columnspan=2, sticky="w", padx=4)

    # =========================================================================
    # Результаты
    # =========================================================================
    rf = ttk.LabelFrame(root, text="Результаты", padding=8)
    rf.grid(row=2, column=0, padx=10, pady=8, sticky="nsew")

    res_txt = tk.Text(rf, height=7, width=70, state="disabled",
                      font=("Courier", 9))
    res_txt.grid(row=0, column=0, sticky="nsew")

    def _show(r):
        res_txt.config(state="normal")
        res_txt.delete("1.0", "end")
        res_txt.insert("end",
                       f"Лучшая точка : x = {r['best_x']:.6f},  "
                       f"y = {r['best_y']:.6f}\n")
        res_txt.insert("end", f"f(x, y)       = {r['best_f']:.6f}\n")
        res_txt.insert("end", f"dx до истины  = {r['dx']:.6f}\n")
        res_txt.insert("end", f"df = f − f*   = {r['df']:.6f}\n")
        res_txt.insert("end", f"Вычислений f  : {r['evals']}\n")
        res_txt.insert("end", f"Поколений     : {r['gens_done']}\n")
        res_txt.insert("end", f"Время         : {r['time_sec']:.3f} с\n")
        res_txt.config(state="disabled")

    # Трасса ГА с русскими заголовками столбцов
    _TRACE_RU_COLUMNS = {
        "gen":          "поколение",
        "best_f":       "лучшее_f",
        "mean_f":       "среднее_f",
        "best_dx":      "dx_лучшей",
        "best_x":       "лучший_x",
        "best_y":       "лучший_y",
        "evals_so_far": "вычислений_всего",
    }

    # =========================================================================
    # Рабочий поток
    # =========================================================================
    def _worker():
        try:
            enc_internal = _ENC_LABELS.get(enc_var.get(), "binary")
            cx_label = cx_var.get()
            if enc_internal == "binary":
                cx_internal = _CX_BIN_LABELS.get(cx_label, "one_point")
            else:
                cx_internal = _CX_REAL_LABELS.get(cx_label, "arithmetic")

            params = {
                "encoding":       enc_internal,
                "pop_size":       int(pop_var.get()),
                "generations":    int(gen_var.get()),
                "crossover_type": cx_internal,
                "crossover_rate": float(cxr_var.get()),
                "mutation_rate":  float(mut_var.get()),
                "tournament_k":   int(tk_var.get()),
                "elitism":        int(eli_var.get()),
                "viz_every":      10,
            }
        except ValueError as exc:
            q.put(("error", f"Ошибка в параметрах: {exc}"))
            return

        def cb(gen, gens, best_f, positions_xy=None, best_xy=None):
            pct = int(gen / max(gens, 1) * 100)
            q.put(("progress", gen, gens, best_f, pct, positions_xy, best_xy))

        try:
            result, trace_df = run_ga(params, callback=cb)
            q.put(("done", result, trace_df))
        except Exception as exc:
            q.put(("error", str(exc)))

    def _start():
        run_btn.config(state="disabled")
        pb_var.set(0)
        pb_lbl.set("")
        status_var.set("Выполняется…")
        _clear_viz()
        threading.Thread(target=_worker, daemon=True).start()
        root.after(50, _poll)

    def _poll():
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == "progress":
                    _, gen, gens, best_f, pct, positions_xy, best_xy = msg
                    pb_var.set(pct)
                    pb_lbl.set(f"Поколение {gen}/{gens},  f = {best_f:.4f}")
                    if positions_xy is not None:
                        _redraw_viz(gen, gens, best_f, positions_xy, best_xy)
                elif msg[0] == "done":
                    result, trace_df = msg[1], msg[2]
                    status_var.set("Готово")
                    pb_var.set(100)
                    _show(result)
                    os.makedirs("out", exist_ok=True)
                    with open(os.path.join("out", "ga_last_result.json"),
                              "w", encoding="utf-8") as fh:
                        json.dump(result, fh, ensure_ascii=False, indent=2)
                    trace_ru = trace_df.rename(columns=_TRACE_RU_COLUMNS)
                    trace_ru.to_csv(
                        os.path.join("out", "ga_last_trace.csv"), index=False,
                        encoding="utf-8")
                    pb_lbl.set(
                        "Сохранено: out/ga_last_result.json, "
                        "out/ga_last_trace.csv")
                    run_btn.config(state="normal")
                    return
                elif msg[0] == "error":
                    status_var.set("Ошибка")
                    messagebox.showerror("Ошибка", msg[1])
                    run_btn.config(state="normal")
                    return
        except _queue.Empty:
            pass
        root.after(50, _poll)

    run_btn.config(command=_start)
    root.mainloop()


if __name__ == "__main__":
    main()

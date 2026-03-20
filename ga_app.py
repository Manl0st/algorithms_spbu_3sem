"""
ga_app.py
=========
Генетический алгоритм (GA) для минимизации функции Eggholder.

Поддерживает два способа кодирования генотипа:
  - binary  : вещественные переменные кодируются цепочкой битов
  - real    : вещественные переменные хранятся напрямую

Запуск (открывает GUI-окно)::

    python ga_app.py

Для использования из run_experiments.py доступна функция::

    result, trace_df = run_ga(params: dict)
"""

import json
import math
import os
import queue as _queue
import subprocess
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
    Для real   — случайные вещественные пары (x, y) в BOUNDS.
    """
    population = []
    lo, hi = BOUNDS[0]

    for _ in range(pop_size):
        if encoding == "binary":
            chrom = rng.integers(0, 2, size=2 * bits, dtype=np.int8)
        else:  # real
            chrom = rng.uniform(lo, hi, size=2).astype(float)
        population.append(chrom)

    return population


def _phenotype(chromosome, encoding: str, bits: int) -> tuple[float, float]:
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
    После мутации значения клипируются в BOUNDS.
    """
    chrom = chromosome.copy()
    lo, hi = BOUNDS[0]
    for i in range(len(chrom)):
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
    callback : callable, optional
        Вызывается на каждом поколении: callback(gen, total_gens, best_f).
        Используется GUI для отображения прогресса.

    Возвращает
    ----------
    result : dict
        Итоговые метрики: best_x, best_y, best_f, dx, df, evals, time_sec и т.д.
    trace_df : pd.DataFrame
        Статистика по поколениям.
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

    # Для вещественного кодирования поддерживается только арифметический кроссинговер
    if enc == "real" and cx_type != "arithmetic":
        print(f"Предупреждение: для вещественного кодирования поддерживается только "
              f"'arithmetic'. Тип '{cx_type}' заменён на 'arithmetic'.")
        cx_type = "arithmetic"

    # sigma для гауссовой мутации (real): 5% от диапазона
    lo, hi = BOUNDS[0]
    sigma = (hi - lo) * 0.05

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

    # -- главный цикл ---------------------------------------------------------
    for gen in range(generations):

        # 1. Статистика текущего поколения
        best_idx = int(np.argmin(fitness))
        best_f   = float(fitness[best_idx])
        mean_f   = float(np.mean(fitness))
        bx, by   = _phenotype(population[best_idx], enc, bits)
        best_dx  = _euclidean(bx, by, TRUE_MIN[0], TRUE_MIN[1])

        trace_rows.append({
            "gen":         gen,
            "best_f":      best_f,
            "mean_f":      mean_f,
            "best_dx":     best_dx,
            "best_x":      bx,
            "best_y":      by,
            "evals_so_far": obj.evals,
        })

        if callback is not None:
            callback(gen, generations, best_f)

        # 2. Критерий останова по улучшению функции.
        #    Улучшением считается уменьшение best_f более чем на eps_f.
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

        # Границы для клиппинга вещественных потомков
        _lo_arr = np.array([b[0] for b in BOUNDS])
        _hi_arr = np.array([b[1] for b in BOUNDS])

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
                # Клиппинг после кроссинговера (арифметический кроссинговер не выходит
                # за пределы, но страховка нужна при возможных числовых погрешностях)
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

def main():
    """Точка входа при запуске `python ga_app.py` — открывает GUI-окно."""
    q: _queue.Queue = _queue.Queue()

    root = tk.Tk()
    root.title("ГА — Eggholder")
    root.resizable(False, False)

    # ---- Параметры ----
    pf = ttk.LabelFrame(root, text="Параметры", padding=8)
    pf.grid(row=0, column=0, padx=10, pady=8, sticky="nsew")

    def _row(parent, row, label, default, choices=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w",
                                           padx=4, pady=2)
        var = tk.StringVar(value=str(default))
        if choices:
            w = ttk.Combobox(parent, textvariable=var, values=choices,
                             state="readonly", width=22)
        else:
            w = ttk.Entry(parent, textvariable=var, width=24)
        w.grid(row=row, column=1, sticky="w", padx=4, pady=2)
        return var, w

    enc_var,   enc_cb  = _row(pf,  0, "Кодирование:",           "binary",
                              ["binary", "real"])
    bits_var,  bits_e  = _row(pf,  1, "Бит на переменную:",     "20")
    pop_var,   _       = _row(pf,  2, "Размер популяции:",      "50")
    gen_var,   _       = _row(pf,  3, "Поколений:",              "300")
    cx_var,    cx_cb   = _row(pf,  4, "Кроссинговер:",           "one_point",
                              ["one_point", "two_point", "uniform"])
    cxr_var,   _       = _row(pf,  5, "Вер. кроссинговера:",    "0.9")
    mut_var,   _       = _row(pf,  6, "Вер. мутации:",           "0.02")
    tk_var,    _       = _row(pf,  7, "Размер турнира:",         "3")
    eli_var,   _       = _row(pf,  8, "Элитизм:",                "1")
    seed_var,  _       = _row(pf,  9, "Seed:",                   "42")
    epsf_var,  _       = _row(pf, 10, "stop_eps_f:",              "1e-6")
    epsdx_var, _       = _row(pf, 11, "stop_eps_dx (0=откл.):",  "0.0")
    pat_var,   _       = _row(pf, 12, "Терпение:",                "80")

    def _on_enc(*_):
        if enc_var.get() == "real":
            bits_e.config(state="disabled")
            cx_var.set("arithmetic")
            cx_cb.config(state="disabled", values=["arithmetic"])
        else:
            bits_e.config(state="normal")
            cx_cb.config(state="readonly",
                         values=["one_point", "two_point", "uniform"])
            if cx_var.get() == "arithmetic":
                cx_var.set("one_point")

    enc_var.trace_add("write", _on_enc)

    # ---- Управление ----
    cf = ttk.Frame(root, padding=8)
    cf.grid(row=1, column=0, sticky="ew")

    run_btn = ttk.Button(cf, text="Запустить")
    run_btn.grid(row=0, column=0, padx=4, pady=4)

    def _open_out():
        folder = os.path.abspath("out")
        os.makedirs(folder, exist_ok=True)
        if os.name == "nt":
            os.startfile(folder)
        else:
            subprocess.Popen(["xdg-open", folder])

    ttk.Button(cf, text="Открыть папку out",
               command=_open_out).grid(row=0, column=1, padx=4)

    status_var = tk.StringVar(value="Готово")
    ttk.Label(cf, textvariable=status_var,
              font=("", 10, "bold")).grid(row=1, column=0, columnspan=2,
                                          sticky="w", padx=4)

    pb_var = tk.IntVar(value=0)
    ttk.Progressbar(cf, variable=pb_var, maximum=100,
                    length=350, mode="determinate").grid(
        row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=2)

    pb_lbl = tk.StringVar(value="")
    ttk.Label(cf, textvariable=pb_lbl).grid(row=3, column=0, columnspan=2,
                                             sticky="w", padx=4)

    # ---- Результаты ----
    rf = ttk.LabelFrame(root, text="Результаты", padding=8)
    rf.grid(row=2, column=0, padx=10, pady=8, sticky="nsew")

    res_txt = tk.Text(rf, height=7, width=54, state="disabled",
                      font=("Courier", 9))
    res_txt.grid(row=0, column=0, sticky="nsew")

    def _show(r):
        res_txt.config(state="normal")
        res_txt.delete("1.0", "end")
        res_txt.insert("end",
                       f"Лучшая точка : x = {r['best_x']:.6f},  "
                       f"y = {r['best_y']:.6f}\n")
        res_txt.insert("end", f"f(x*, y*)     = {r['best_f']:.6f}\n")
        res_txt.insert("end", f"dx до истины  = {r['dx']:.6f}\n")
        res_txt.insert("end", f"df = f − f*   = {r['df']:.6f}\n")
        res_txt.insert("end", f"Вычислений f  : {r['evals']}\n")
        res_txt.insert("end", f"Поколений     : {r['gens_done']}\n")
        res_txt.insert("end", f"Время         : {r['time_sec']:.3f} с\n")
        res_txt.config(state="disabled")

    def _worker():
        try:
            params = {
                "encoding":            enc_var.get(),
                "bits_per_var":        int(bits_var.get()),
                "pop_size":            int(pop_var.get()),
                "generations":         int(gen_var.get()),
                "crossover_type":      cx_var.get(),
                "crossover_rate":      float(cxr_var.get()),
                "mutation_rate":       float(mut_var.get()),
                "tournament_k":        int(tk_var.get()),
                "elitism":             int(eli_var.get()),
                "seed":                int(seed_var.get()),
                "stop_eps_f":          float(epsf_var.get()),
                "stop_eps_dx":         float(epsdx_var.get()),
                "no_improve_patience": int(pat_var.get()),
            }
        except ValueError as exc:
            q.put(("error", f"Ошибка в параметрах: {exc}"))
            return

        def cb(gen, gens, best_f):
            pct = int(gen / max(gens, 1) * 100)
            q.put(("progress", gen, gens, best_f, pct))

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
        threading.Thread(target=_worker, daemon=True).start()
        root.after(50, _poll)

    def _poll():
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == "progress":
                    _, gen, gens, best_f, pct = msg
                    pb_var.set(pct)
                    pb_lbl.set(
                        f"Поколение {gen}/{gens},  best_f = {best_f:.4f}")
                elif msg[0] == "done":
                    result, trace_df = msg[1], msg[2]
                    status_var.set("Готово")
                    pb_var.set(100)
                    _show(result)
                    os.makedirs("out", exist_ok=True)
                    with open(os.path.join("out", "ga_last_result.json"),
                              "w", encoding="utf-8") as fh:
                        json.dump(result, fh, ensure_ascii=False, indent=2)
                    trace_df.to_csv(
                        os.path.join("out", "ga_last_trace.csv"), index=False)
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


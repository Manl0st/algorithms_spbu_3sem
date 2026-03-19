"""
ga_app.py
=========
Генетический алгоритм (GA) для минимизации функции Eggholder.

Поддерживает два способа кодирования генотипа:
  - binary  : вещественные переменные кодируются цепочкой битов
  - real    : вещественные переменные хранятся напрямую

Запуск в интерактивном режиме::

    python ga_app.py

Программа задаёт вопросы через input(); нажмите Enter, чтобы принять значение
по умолчанию.

Для использования из run_experiments.py доступна функция::

    result, trace_df = run_ga(params: dict)
"""

import json
import math
import os
import time

import numpy as np
import pandas as pd

from objective import BOUNDS, TRUE_F, TRUE_MIN, CounterObjective

# ---------------------------------------------------------------------------
# Вспомогательные функции общего назначения
# ---------------------------------------------------------------------------

def _ask(prompt: str, default):
    """
    Вывести подсказку с дефолтом и считать ввод пользователя.
    Если пользователь нажимает Enter (пустой ввод), возвращается default.
    """
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    return type(default)(raw)


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
    """
    chromosome = np.empty(2 * bits, dtype=np.int8)
    lo, hi = BOUNDS[0]
    max_int = (1 << bits) - 1

    for var_idx, v in enumerate((x, y)):
        # нормировать в [0, max_int]
        i = round((v - lo) / (hi - lo) * max_int)
        i = max(0, min(max_int, i))
        for bit_idx in range(bits):
            chromosome[var_idx * bits + (bits - 1 - bit_idx)] = (i >> bit_idx) & 1

    return chromosome


def _decode_binary(chromosome: np.ndarray, bits: int) -> tuple[float, float]:
    """
    Декодировать двоичную хромосому обратно в пару (x, y).
    """
    lo, hi = BOUNDS[0]
    max_int = (1 << bits) - 1
    result = []

    for var_idx in range(2):
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

def run_ga(params: dict) -> tuple[dict, pd.DataFrame]:
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
    stop_eps_dx     : float — останов, если изменение best_f < eps_dx
    no_improve_patience : int — останов, если нет улучшения N поколений

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
    eps_dx       = float(params.get("stop_eps_dx",   1e-3))
    patience     = int(params.get("no_improve_patience", 80))

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

        # 2. Критерий останова: нет улучшения в течение patience поколений.
        #    Улучшением считается уменьшение best_f более чем на eps_dx.
        if best_f < best_f_prev - eps_dx:
            best_f_prev    = best_f
            no_improve_cnt = 0
        else:
            no_improve_cnt += 1

        if no_improve_cnt >= patience:
            break

        # 3. Элитизм: сохранить лучших особей
        elite_idxs = np.argsort(fitness)[:elitism]
        elites     = [population[i].copy() for i in elite_idxs]

        # 4. Формирование нового поколения
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

        # 5. Добавить элиты в начало нового поколения
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
# Интерактивный запуск
# ===========================================================================

def _interactive_params() -> dict:
    """Спросить у пользователя параметры через input() с дефолтами."""

    print("\n=== Генетический алгоритм — Eggholder ===")
    print("Нажмите Enter для принятия значения по умолчанию.\n")

    enc = _ask("Кодирование (binary / real)", "binary")
    if enc not in ("binary", "real"):
        print(f"Неизвестное кодирование '{enc}', используется 'binary'.")
        enc = "binary"

    bits = 20
    if enc == "binary":
        bits = int(_ask("Bits per variable", 20))

    pop_size    = int(_ask("Размер популяции (pop_size)", 50))
    generations = int(_ask("Число поколений (generations)", 300))

    # тип кроссинговера зависит от кодирования
    if enc == "binary":
        cx_type = _ask("Тип кроссинговера (one_point / two_point / uniform)", "one_point")
    else:
        cx_type = "arithmetic"
        print(f"Кроссинговер для real: arithmetic (фиксировано)")

    cx_rate  = float(_ask("Вероятность кроссинговера (crossover_rate)", 0.9))
    mut_rate = float(_ask("Вероятность мутации (mutation_rate)", 0.02))
    tourn_k  = int(_ask("Размер турнира (tournament_k)", 3))
    elitism  = int(_ask("Число элитных особей (elitism)", 1))
    seed     = int(_ask("Начальный seed", 42))
    eps_dx   = float(_ask("Критерий останова eps_dx (stop_eps_dx)", 1e-3))
    patience = int(_ask("Терпение без улучшения (no_improve_patience)", 80))

    return {
        "encoding":            enc,
        "bits_per_var":        bits,
        "pop_size":            pop_size,
        "generations":         generations,
        "crossover_type":      cx_type,
        "crossover_rate":      cx_rate,
        "mutation_rate":       mut_rate,
        "tournament_k":        tourn_k,
        "elitism":             elitism,
        "seed":                seed,
        "stop_eps_dx":         eps_dx,
        "no_improve_patience": patience,
    }


def main():
    """Точка входа при запуске `python ga_app.py`."""

    params   = _interactive_params()
    result, trace_df = run_ga(params)

    # -- вывод результатов ---------------------------------------------------
    print("\n--- Результаты GA ---")
    print(f"  Лучшая точка  : x = {result['best_x']:.6f}, y = {result['best_y']:.6f}")
    print(f"  f(x*, y*)     : {result['best_f']:.6f}")
    print(f"  dx до истины  : {result['dx']:.6f}")
    print(f"  df = f - f*   : {result['df']:.6f}")
    print(f"  Вычислений f  : {result['evals']}")
    print(f"  Поколений     : {result['gens_done']}")
    print(f"  Время         : {result['time_sec']:.3f} с")

    # -- сохранение результатов ----------------------------------------------
    os.makedirs("out", exist_ok=True)

    # JSON с параметрами и итогами
    json_path = os.path.join("out", "ga_last_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nРезультат сохранён: {json_path}")

    # CSV с трассой по поколениям
    csv_path = os.path.join("out", "ga_last_trace.csv")
    trace_df.to_csv(csv_path, index=False)
    print(f"Трасса сохранена : {csv_path}")


if __name__ == "__main__":
    main()

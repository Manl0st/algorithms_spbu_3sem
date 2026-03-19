"""
run_experiments.py
==================
Пакетный запуск экспериментов для сравнения алгоритмов GA и PSO
на функции Eggholder.

Как использовать
----------------
1. При необходимости отредактируйте список EXPERIMENTS и значение REPEATS
   в начале файла.
2. Запустите::

       python run_experiments.py

Результаты будут сохранены в папке out/:
  - results.csv  — одна строка на один запуск
  - traces.csv   — динамика сходимости (все итерации всех запусков)
  - summary.csv  — агрегированная статистика по конфигурациям
"""

import json
import os
import time

import numpy as np
import pandas as pd

# Импортируем алгоритмы из соседних файлов
from ga_app  import run_ga
from pso_app import run_pso
from objective import TRUE_MIN

# ===========================================================================
# Настройки экспериментов — редактируйте этот раздел
# ===========================================================================

# Число независимых запусков для каждой конфигурации
REPEATS: int = 30

# Порог по dx для определения «успеха» (попал в окрестность глобального минимума)
SUCCESS_DX_THRESHOLD: float = 1.0   # единицы расстояния (в координатах x,y)

# ---------------------------------------------------------------------------
# Список конфигураций
# Каждый словарь — одна конфигурация алгоритма.
# Ключ "algo" : "GA" или "PSO"
# Ключ "variant" (опционально): человекочитаемое имя для сводных таблиц/графиков
# Все остальные ключи передаются напрямую в run_ga() или run_pso()
# ---------------------------------------------------------------------------
EXPERIMENTS: list[dict] = [

    # --- GA binary: one_point crossover ---
    {
        "algo":           "GA",
        "variant":        "GA_bin_one_point",
        "encoding":       "binary",
        "bits_per_var":   20,
        "pop_size":       50,
        "generations":    300,
        "crossover_type": "one_point",
        "crossover_rate": 0.9,
        "mutation_rate":  0.02,
        "tournament_k":   3,
        "elitism":        1,
        "no_improve_patience": 80,
    },

    # --- GA binary: two_point crossover ---
    {
        "algo":           "GA",
        "variant":        "GA_bin_two_point",
        "encoding":       "binary",
        "bits_per_var":   20,
        "pop_size":       50,
        "generations":    300,
        "crossover_type": "two_point",
        "crossover_rate": 0.9,
        "mutation_rate":  0.02,
        "tournament_k":   3,
        "elitism":        1,
        "no_improve_patience": 80,
    },

    # --- GA real: arithmetic crossover ---
    {
        "algo":           "GA",
        "variant":        "GA_real_arith",
        "encoding":       "real",
        "pop_size":       50,
        "generations":    300,
        "crossover_type": "arithmetic",
        "crossover_rate": 0.9,
        "mutation_rate":  0.05,
        "tournament_k":   3,
        "elitism":        1,
        "no_improve_patience": 80,
    },

    # --- PSO basic: малый инерционный вес ---
    {
        "algo":       "PSO",
        "variant":    "PSO_basic_w04",
        "mode":       "basic",
        "swarm_size": 30,
        "iters":      300,
        "w":          0.4,
        "c1":         1.5,
        "c2":         1.5,
    },

    # --- PSO basic: стандартный инерционный вес ---
    {
        "algo":       "PSO",
        "variant":    "PSO_basic_w07",
        "mode":       "basic",
        "swarm_size": 30,
        "iters":      300,
        "w":          0.7,
        "c1":         1.5,
        "c2":         1.5,
    },

    # --- PSO constriction: Clerc-Kennedy ---
    {
        "algo":       "PSO",
        "variant":    "PSO_constriction",
        "mode":       "constriction",
        "swarm_size": 30,
        "iters":      300,
        "c1":         2.05,
        "c2":         2.05,
    },
]

# ===========================================================================
# Вспомогательная функция: запустить один эксперимент
# ===========================================================================

def _run_one(config: dict, seed: int) -> tuple[dict, pd.DataFrame]:
    """
    Запустить один алгоритм с заданной конфигурацией и seed.

    Параметры
    ----------
    config : dict
        Конфигурация с ключами "algo", "variant" и параметрами алгоритма.
    seed : int
        Начальный seed для генератора псевдослучайных чисел.

    Возвращает
    ----------
    result : dict
        Метрики одного запуска.
    trace_df : pd.DataFrame
        Трасса сходимости.
    """
    # Подготовить параметры: убрать служебные ключи, добавить seed
    algo    = config["algo"]
    variant = config.get("variant", algo)
    run_params = {k: v for k, v in config.items()
                  if k not in ("algo", "variant")}
    run_params["seed"] = seed

    t0 = time.perf_counter()

    if algo == "GA":
        result, trace_df = run_ga(run_params)
        best_x   = result["best_x"]
        best_y   = result["best_y"]
        best_f   = result["best_f"]
        dx       = result["dx"]
        df       = result["df"]
        evals    = result["evals"]
        n_iters  = result["gens_done"]
    elif algo == "PSO":
        result, trace_df = run_pso(run_params)
        best_x   = result["gbest_x"]
        best_y   = result["gbest_y"]
        best_f   = result["best_f"]
        dx       = result["dx"]
        df       = result["df"]
        evals    = result["evals"]
        n_iters  = result["iters"]
    else:
        raise ValueError(f"Неизвестный алгоритм: {algo}")

    elapsed = time.perf_counter() - t0

    row = {
        "algo":        algo,
        "variant":     variant,
        "params_json": json.dumps(run_params, ensure_ascii=False),
        "seed":        seed,
        "best_x":      best_x,
        "best_y":      best_y,
        "best_f":      best_f,
        "dx":          dx,
        "df":          df,
        "evals":       evals,
        "iters_or_gens": n_iters,
        "time_sec":    elapsed,
    }
    return row, trace_df


# ===========================================================================
# Главная функция
# ===========================================================================

def main():
    """Прогнать все эксперименты и сохранить CSV-файлы в out/."""

    os.makedirs("out", exist_ok=True)

    all_results: list[dict]      = []   # строки для results.csv
    all_traces:  list[pd.DataFrame] = []  # трассы для traces.csv

    run_id = 0  # глобальный счётчик запусков

    total_runs = len(EXPERIMENTS) * REPEATS
    print(f"Запускаем {len(EXPERIMENTS)} конфигураций × {REPEATS} повторений"
          f" = {total_runs} запусков.")
    print("=" * 60)

    for exp_idx, config in enumerate(EXPERIMENTS):
        variant = config.get("variant", config.get("algo", "?"))
        algo    = config["algo"]
        print(f"\n[{exp_idx + 1}/{len(EXPERIMENTS)}] {algo} / {variant}")

        seed_base = config.get("seed_base", 1000)  # базовый seed для серии

        for rep in range(REPEATS):
            seed = seed_base + rep
            print(f"  Повторение {rep + 1:3d}/{REPEATS} (seed={seed})...", end=" ")

            try:
                row, trace_df = _run_one(config, seed)
            except Exception as exc:
                print(f"ОШИБКА: {exc}")
                continue

            row["run_id"] = run_id
            all_results.append(row)

            # Добавить run_id в трассу и сохранить
            trace_df = trace_df.copy()
            trace_df["run_id"]  = run_id
            trace_df["variant"] = variant
            trace_df["algo"]    = algo
            # Переименовать колонку итераций в единое имя "iter"
            if "gen" in trace_df.columns:
                trace_df.rename(columns={"gen": "iter"}, inplace=True)
            all_traces.append(trace_df)

            run_id += 1
            print(f"best_f={row['best_f']:.4f}, dx={row['dx']:.4f}, "
                  f"t={row['time_sec']:.2f}s")

    # -----------------------------------------------------------------------
    # Сохранить results.csv
    # -----------------------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    # Упорядочить столбцы
    cols_order = ["run_id", "algo", "variant", "params_json", "seed",
                  "best_x", "best_y", "best_f", "dx", "df",
                  "evals", "iters_or_gens", "time_sec"]
    results_df = results_df[cols_order]
    results_path = os.path.join("out", "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nСохранено: {results_path}  ({len(results_df)} строк)")

    # -----------------------------------------------------------------------
    # Сохранить traces.csv
    # -----------------------------------------------------------------------
    if all_traces:
        traces_df = pd.concat(all_traces, ignore_index=True)
        # Выбрать нужные столбцы
        trace_cols = ["run_id", "algo", "variant", "iter", "best_f", "mean_f", "best_dx"]
        # Оставить только те, которые есть в DataFrame
        trace_cols = [c for c in trace_cols if c in traces_df.columns]
        traces_df  = traces_df[trace_cols]
        traces_path = os.path.join("out", "traces.csv")
        traces_df.to_csv(traces_path, index=False)
        print(f"Сохранено: {traces_path}  ({len(traces_df)} строк)")

    # -----------------------------------------------------------------------
    # Агрегировать и сохранить summary.csv
    # -----------------------------------------------------------------------
    if not results_df.empty:
        summary_rows = []
        for variant, grp in results_df.groupby("variant", sort=False):
            algo_name = grp["algo"].iloc[0]
            n = len(grp)

            success_mask = grp["dx"] < SUCCESS_DX_THRESHOLD
            success_rate = float(success_mask.sum()) / n  # доля успешных запусков

            summary_rows.append({
                "algo":          algo_name,
                "variant":       variant,
                "repeats":       n,
                "mean_best_f":   grp["best_f"].mean(),
                "std_best_f":    grp["best_f"].std(),
                "median_best_f": grp["best_f"].median(),
                "mean_dx":       grp["dx"].mean(),
                "std_dx":        grp["dx"].std(),
                "success_rate":  success_rate,
                "mean_time_sec": grp["time_sec"].mean(),
                "mean_evals":    grp["evals"].mean(),
            })

        summary_df  = pd.DataFrame(summary_rows)
        summary_path = os.path.join("out", "summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Сохранено: {summary_path}  ({len(summary_df)} строк)")

        # Краткая сводка в консоли
        print("\n=== Сводка результатов ===")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(summary_df.to_string(index=False))

    print("\nГотово.")


if __name__ == "__main__":
    main()

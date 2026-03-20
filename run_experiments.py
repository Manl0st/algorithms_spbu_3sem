"""
run_experiments.py
==================
Пакетный запуск экспериментов для сравнения алгоритмов GA и PSO
на функции Eggholder (CLI, без GUI).

Как использовать
----------------
Запустите::

    python run_experiments.py

Результаты будут сохранены в папке out/:
  - results.csv          — одна строка на один запуск (русские заголовки)
  - traces.csv           — динамика сходимости (русские заголовки)
  - summary.csv          — агрегированная статистика по конфигурациям (русские заголовки)
  - experiment_meta.json — метаданные запуска (порог успеха, число повторений и т.д.)
"""

import datetime
import json
import os
import time

import numpy as np
import pandas as pd

# Импортируем алгоритмы из соседних файлов
from ga_app  import run_ga
from pso_app import run_pso
from objective import BOUNDS, TRUE_MIN, TRUE_F

# ===========================================================================
# Настройки экспериментов — редактируйте этот раздел
# ===========================================================================

# Число независимых запусков для каждой конфигурации (можно изменить через меню)
REPEATS: int = 30

# Порог по dx для определения «успеха» (попал в окрестность глобального минимума)
SUCCESS_DX_THRESHOLD: float = 1.0   # единицы расстояния (в координатах x,y)

# ---------------------------------------------------------------------------
# Человекочитаемые имена вариантов (для отображения в GUI)
# ---------------------------------------------------------------------------
_VARIANT_DISPLAY = {
    "GA_bin_one_point":  "ГА (бинарный, 1-точ. кроссинговер)",
    "GA_bin_two_point":  "ГА (бинарный, 2-точ. кроссинговер)",
    "GA_real_arith":     "ГА (вещественный, арифм. кроссинговер)",
    "PSO_basic_w04":     "PSO базовый (w=0.4)",
    "PSO_basic_w07":     "PSO базовый (w=0.7)",
    "PSO_constriction":  "PSO с коэф. сжатия (χ)",
}

# ---------------------------------------------------------------------------
# Список конфигураций
# Каждый словарь — одна конфигурация алгоритма.
# Ключ "algo" : "GA" или "PSO"
# Ключ "variant" (опционально): идентификатор для сводных таблиц/графиков
# Все остальные ключи передаются напрямую в run_ga() или run_pso()
# ---------------------------------------------------------------------------
EXPERIMENTS: list = [

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

    Возвращает (row: dict, trace_df: pd.DataFrame) — внутренние имена столбцов.
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
        n_iters  = result["iters_done"]  # FIX: фактически выполненных итераций
    else:
        raise ValueError(f"Неизвестный алгоритм: {algo}")

    elapsed = time.perf_counter() - t0

    row = {
        "run_id":        None,  # будет заполнено в _run_batch
        "algo":          algo,
        "variant":       variant,
        "params_json":   json.dumps(run_params, ensure_ascii=False),
        "seed":          seed,
        "best_x":        best_x,
        "best_y":        best_y,
        "best_f":        best_f,
        "dx":            dx,
        "df":            df,
        "evals":         evals,
        "iters_or_gens": n_iters,
        "time_sec":      elapsed,
    }
    return row, trace_df


# ===========================================================================
# Пакетный запуск (используется GUI)
# ===========================================================================

# Соответствие внутренних имён столбцов → русских заголовков для CSV
_RESULTS_RU = {
    "run_id":        "id_запуска",
    "algo":          "алгоритм",
    "variant":       "вариант",
    "params_json":   "параметры_json",
    "seed":          "seed",
    "best_x":        "лучший_x",
    "best_y":        "лучший_y",
    "best_f":        "лучшее_f",
    "dx":            "dx_до_истины",
    "df":            "df_ошибка",
    "evals":         "вычислений_f",
    "iters_or_gens": "итераций",
    "time_sec":      "время_сек",
}

_TRACES_RU = {
    "run_id":  "id_запуска",
    "algo":    "алгоритм",
    "variant": "вариант",
    "iter":    "итерация",
    "best_f":  "лучшее_f",
    "mean_f":  "среднее_f",
    "best_dx": "dx_лучшей",
}

_SUMMARY_RU = {
    "algo":          "алгоритм",
    "variant":       "вариант",
    "repeats":       "повторов",
    "mean_best_f":   "среднее_f",
    "std_best_f":    "std_f",
    "median_best_f": "медиана_f",
    "min_best_f":    "мин_f",
    "max_best_f":    "макс_f",
    "mean_dx":       "среднее_dx",
    "std_dx":        "std_dx",
    "median_dx":     "медиана_dx",
    "min_dx":        "мин_dx",
    "max_dx":        "макс_dx",
    "success_rate":  "доля_успеха",
    "mean_time_sec": "среднее_время_сек",
    "mean_evals":    "среднее_вычислений",
}


def _run_batch(selected_configs: list[dict], repeats: int,
               success_dx: float, progress_cb) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Запустить все выбранные конфигурации и вернуть (results_df, traces_df, summary_df).

    Все три DataFrame используют внутренние (английские) имена столбцов.
    При сохранении в CSV столбцы переименовываются в русские.
    progress_cb(done: int, total: int, msg: str) вызывается после каждого запуска.
    """
    os.makedirs("out", exist_ok=True)

    all_results: list[dict] = []
    all_traces:  list[pd.DataFrame] = []
    run_id = 0
    total_runs = len(selected_configs) * repeats

    for exp_idx, config in enumerate(selected_configs):
        variant = config.get("variant", config.get("algo", "?"))
        algo    = config["algo"]
        seed_base = config.get("seed_base", 1000)

        for rep in range(repeats):
            seed = seed_base + rep
            progress_cb(
                run_id,
                total_runs,
                f"[{exp_idx + 1}/{len(selected_configs)}] "
                f"{_VARIANT_DISPLAY.get(variant, variant)},  "
                f"повторение {rep + 1}/{repeats}"
            )
            try:
                row, trace_df = _run_one(config, seed)
            except Exception as exc:
                progress_cb(run_id, total_runs, f"ОШИБКА: {exc}")
                run_id += 1
                continue

            row["run_id"] = run_id
            all_results.append(row)

            trace_df = trace_df.copy()
            trace_df["run_id"]  = run_id
            trace_df["variant"] = variant
            trace_df["algo"]    = algo
            if "gen" in trace_df.columns:
                trace_df.rename(columns={"gen": "iter"}, inplace=True)
            all_traces.append(trace_df)
            run_id += 1

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        return results_df, pd.DataFrame(), pd.DataFrame()

    # Упорядочить столбцы (внутренние имена)
    cols_order = ["run_id", "algo", "variant", "params_json", "seed",
                  "best_x", "best_y", "best_f", "dx", "df",
                  "evals", "iters_or_gens", "time_sec"]
    results_df = results_df[cols_order]

    # Сохранить results.csv с русскими заголовками
    results_df.rename(columns=_RESULTS_RU).to_csv(
        os.path.join("out", "results.csv"), index=False, encoding="utf-8")

    traces_df = pd.DataFrame()
    if all_traces:
        traces_df = pd.concat(all_traces, ignore_index=True)
        trace_cols = [c for c in
                      ["run_id", "algo", "variant", "iter",
                       "best_f", "mean_f", "best_dx"]
                      if c in traces_df.columns]
        traces_df = traces_df[trace_cols]
        # Сохранить traces.csv с русскими заголовками
        traces_df.rename(columns=_TRACES_RU).to_csv(
            os.path.join("out", "traces.csv"), index=False, encoding="utf-8")

    # Сводка (внутренние имена в памяти, русские при сохранении)
    summary_rows = []
    for variant_name, grp in results_df.groupby("variant", sort=False):
        algo_name = grp["algo"].iloc[0]
        n = len(grp)
        success_mask = grp["dx"] < success_dx
        summary_rows.append({
            "algo":          algo_name,
            "variant":       variant_name,
            "repeats":       n,
            "mean_best_f":   grp["best_f"].mean(),
            "std_best_f":    grp["best_f"].std(),
            "median_best_f": grp["best_f"].median(),
            "min_best_f":    grp["best_f"].min(),
            "max_best_f":    grp["best_f"].max(),
            "mean_dx":       grp["dx"].mean(),
            "std_dx":        grp["dx"].std(),
            "median_dx":     grp["dx"].median(),
            "min_dx":        grp["dx"].min(),
            "max_dx":        grp["dx"].max(),
            "success_rate":  float(success_mask.sum()) / n,
            "mean_time_sec": grp["time_sec"].mean(),
            "mean_evals":    grp["evals"].mean(),
        })
    summary_df = pd.DataFrame(summary_rows)
    # Сохранить summary.csv с русскими заголовками
    summary_df.rename(columns=_SUMMARY_RU).to_csv(
        os.path.join("out", "summary.csv"), index=False, encoding="utf-8")

    meta = {
        "created_at":           datetime.datetime.now().isoformat(
                                    timespec="seconds"),
        "repeats":              repeats,
        "SUCCESS_DX_THRESHOLD": success_dx,
        "number_of_variants":   len(selected_configs),
        "total_runs":           run_id,
        "bounds":               [list(b) for b in BOUNDS],
        "true_min":             list(TRUE_MIN),
        "true_f":               TRUE_F,
    }
    with open(os.path.join("out", "experiment_meta.json"),
              "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    return results_df, traces_df, summary_df


# ===========================================================================
# CLI
# ===========================================================================

def main():
    """CLI-запуск: все конфигурации с фиксированными сидами."""
    # Фиксированные параметры для воспроизводимых исследовательских данных
    REPEATS_CLI = 30
    SEED_BASE   = 1001   # сиды: 1001, 1002, …, 1030

    # Добавляем seed_base ко всем конфигурациям
    configs = [{**cfg, "seed_base": SEED_BASE} for cfg in EXPERIMENTS]

    total_runs = len(configs) * REPEATS_CLI
    print(f"Запуск экспериментов: {len(configs)} вариантов × {REPEATS_CLI} повторений")
    print(f"Всего запусков: {total_runs}")
    print()

    done_count = [0]

    def progress_cb(done: int, total: int, msg: str):
        done_count[0] += 1
        print(f"  [{done_count[0]:3d}/{total_runs}] {msg}")

    results_df, _traces_df, summary_df = _run_batch(
        configs, REPEATS_CLI, SUCCESS_DX_THRESHOLD, progress_cb)

    if not results_df.empty:
        print()
        print("=== Сводка ===")
        for _, row in summary_df.iterrows():
            display = _VARIANT_DISPLAY.get(row["variant"], row["variant"])
            print(f"  {display}:  среднее f = {row['mean_best_f']:.4f},  "
                  f"доля успеха = {row['success_rate'] * 100:.1f}%")
        print()
        print("Файлы сохранены в папке out/")
    else:
        print("Нет результатов!")


if __name__ == "__main__":
    main()

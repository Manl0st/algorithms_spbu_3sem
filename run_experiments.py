"""
run_experiments.py
==================
Пакетный запуск экспериментов для сравнения алгоритмов GA и PSO
на функции Eggholder.

Как использовать
----------------
Запустите::

    python run_experiments.py

Откроется GUI-окно для настройки и запуска экспериментов.

Результаты будут сохранены в папке out/:
  - results.csv          — одна строка на один запуск
  - traces.csv           — динамика сходимости (все итерации всех запусков)
  - summary.csv          — агрегированная статистика по конфигурациям
  - experiment_meta.json — метаданные запуска (порог успеха, число повторений и т.д.)
"""

import datetime
import json
import os
import queue as _queue
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

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
# Пакетный запуск (используется GUI)
# ===========================================================================

def _run_batch(selected_configs: list[dict], repeats: int,
               success_dx: float, progress_cb) -> tuple[pd.DataFrame,
                                                         pd.DataFrame,
                                                         pd.DataFrame]:
    """
    Запустить все выбранные конфигурации и вернуть (results_df, traces_df, summary_df).

    progress_cb(msg: str) вызывается для обновления статуса.
    """
    os.makedirs("out", exist_ok=True)

    all_results: list[dict] = []
    all_traces:  list[pd.DataFrame] = []
    run_id = 0

    for exp_idx, config in enumerate(selected_configs):
        variant = config.get("variant", config.get("algo", "?"))
        algo    = config["algo"]
        seed_base = config.get("seed_base", 1000)

        for rep in range(repeats):
            seed = seed_base + rep
            progress_cb(
                f"[{exp_idx + 1}/{len(selected_configs)}] "
                f"{variant},  повторение {rep + 1}/{repeats}"
            )
            try:
                row, trace_df = _run_one(config, seed)
            except Exception as exc:
                progress_cb(f"ОШИБКА: {exc}")
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

    # Упорядочить столбцы
    cols_order = ["run_id", "algo", "variant", "params_json", "seed",
                  "best_x", "best_y", "best_f", "dx", "df",
                  "evals", "iters_or_gens", "time_sec"]
    results_df = results_df[cols_order]
    results_df.to_csv(os.path.join("out", "results.csv"), index=False)

    traces_df = pd.DataFrame()
    if all_traces:
        traces_df = pd.concat(all_traces, ignore_index=True)
        trace_cols = [c for c in
                      ["run_id", "algo", "variant", "iter",
                       "best_f", "mean_f", "best_dx"]
                      if c in traces_df.columns]
        traces_df = traces_df[trace_cols]
        traces_df.to_csv(os.path.join("out", "traces.csv"), index=False)

    # Сводка
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
            "success_rate":  float(success_mask.sum()) / n,
            "mean_time_sec": grp["time_sec"].mean(),
            "mean_evals":    grp["evals"].mean(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join("out", "summary.csv"), index=False)

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
# GUI
# ===========================================================================

def main():
    """Точка входа при запуске `python run_experiments.py` — открывает GUI."""
    q: _queue.Queue = _queue.Queue()

    root = tk.Tk()
    root.title("Пакетные эксперименты — Eggholder")
    root.resizable(False, False)

    # ---- Настройки ----
    sf = ttk.LabelFrame(root, text="Настройки", padding=8)
    sf.grid(row=0, column=0, padx=10, pady=8, sticky="ew")

    ttk.Label(sf, text="Повторений:").grid(row=0, column=0, sticky="w",
                                           padx=4, pady=2)
    rep_var = tk.StringVar(value=str(REPEATS))
    ttk.Entry(sf, textvariable=rep_var, width=10).grid(row=0, column=1,
                                                        sticky="w", padx=4)

    ttk.Label(sf, text="Порог успеха dx:").grid(row=1, column=0, sticky="w",
                                                 padx=4, pady=2)
    dx_var = tk.StringVar(value=str(SUCCESS_DX_THRESHOLD))
    ttk.Entry(sf, textvariable=dx_var, width=10).grid(row=1, column=1,
                                                       sticky="w", padx=4)

    # ---- Варианты ----
    vf = ttk.LabelFrame(root, text="Варианты экспериментов", padding=8)
    vf.grid(row=1, column=0, padx=10, pady=4, sticky="ew")

    checks: list[tuple[tk.BooleanVar, dict]] = []
    for cfg in EXPERIMENTS:
        var = tk.BooleanVar(value=True)
        name = cfg.get("variant", cfg.get("algo", "?"))
        ttk.Checkbutton(vf, text=f"{cfg['algo']} / {name}",
                        variable=var).pack(anchor="w")
        checks.append((var, cfg))

    # ---- Управление ----
    cf = ttk.Frame(root, padding=8)
    cf.grid(row=2, column=0, sticky="ew")

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
                    length=400, mode="determinate").grid(
        row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=2)

    prog_lbl = tk.StringVar(value="")
    ttk.Label(cf, textvariable=prog_lbl, wraplength=400).grid(
        row=3, column=0, columnspan=2, sticky="w", padx=4)

    # ---- Сводная таблица ----
    tf = ttk.LabelFrame(root, text="Сводка результатов (top 5)", padding=8)
    tf.grid(row=3, column=0, padx=10, pady=8, sticky="nsew")

    tree_cols = ("variant", "mean_best_f", "success_rate",
                 "mean_dx", "mean_evals")
    tree = ttk.Treeview(tf, columns=tree_cols, show="headings", height=5)
    tree.heading("variant",      text="Вариант")
    tree.heading("mean_best_f",  text="Среднее f")
    tree.heading("success_rate", text="Успех %")
    tree.heading("mean_dx",      text="Среднее dx")
    tree.heading("mean_evals",   text="Вычислений")
    tree.column("variant",      width=160)
    tree.column("mean_best_f",  width=100)
    tree.column("success_rate", width=80)
    tree.column("mean_dx",      width=100)
    tree.column("mean_evals",   width=100)
    tree.grid(row=0, column=0, sticky="nsew")
    sb = ttk.Scrollbar(tf, orient="vertical", command=tree.yview)
    sb.grid(row=0, column=1, sticky="ns")
    tree.configure(yscrollcommand=sb.set)

    def _fill_tree(summary_df: pd.DataFrame):
        for row in tree.get_children():
            tree.delete(row)
        for _, r in summary_df.head(5).iterrows():
            tree.insert("", "end", values=(
                r["variant"],
                f"{r['mean_best_f']:.4f}",
                f"{r['success_rate'] * 100:.1f}%",
                f"{r['mean_dx']:.4f}",
                f"{r['mean_evals']:.0f}",
            ))

    def _worker():
        try:
            repeats   = int(rep_var.get())
            success_dx = float(dx_var.get())
        except ValueError as exc:
            q.put(("error", f"Ошибка в параметрах: {exc}"))
            return

        selected = [cfg for var, cfg in checks if var.get()]
        if not selected:
            q.put(("error", "Не выбрано ни одного варианта."))
            return

        total_runs = len(selected) * repeats
        done_runs  = [0]

        def progress_cb(msg: str):
            done_runs[0] += 1
            pct = int(done_runs[0] / max(total_runs, 1) * 100)
            q.put(("progress", msg, pct))

        try:
            results_df, traces_df, summary_df = _run_batch(
                selected, repeats, success_dx, progress_cb)
            q.put(("done", summary_df))
        except Exception as exc:
            q.put(("error", str(exc)))

    def _start():
        run_btn.config(state="disabled")
        pb_var.set(0)
        prog_lbl.set("")
        status_var.set("Выполняется…")
        for row in tree.get_children():
            tree.delete(row)
        threading.Thread(target=_worker, daemon=True).start()
        root.after(100, _poll)

    def _poll():
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == "progress":
                    _, text, pct = msg
                    pb_var.set(pct)
                    prog_lbl.set(text)
                elif msg[0] == "done":
                    summary_df = msg[1]
                    status_var.set("Готово")
                    pb_var.set(100)
                    prog_lbl.set(
                        "Сохранено: out/results.csv, out/traces.csv, "
                        "out/summary.csv, out/experiment_meta.json")
                    if not summary_df.empty:
                        _fill_tree(summary_df)
                    run_btn.config(state="normal")
                    return
                elif msg[0] == "error":
                    status_var.set("Ошибка")
                    messagebox.showerror("Ошибка", msg[1])
                    run_btn.config(state="normal")
                    return
        except _queue.Empty:
            pass
        root.after(100, _poll)

    run_btn.config(command=_start)
    root.mainloop()


if __name__ == "__main__":
    main()


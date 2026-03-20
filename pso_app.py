"""
pso_app.py
==========
Роевой алгоритм (Particle Swarm Optimisation, PSO) для минимизации функции Eggholder.

Поддерживает два режима:
  - basic        : классический PSO с инерционным весом w
  - constriction : PSO с коэффициентом сжатия χ (Clerc & Kennedy, 2002)

Запуск (открывает GUI-окно)::

    python pso_app.py

Для использования из run_experiments.py доступна функция::

    result, trace_df = run_pso(params: dict)
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
# Вспомогательные утилиты
# ---------------------------------------------------------------------------

def _euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    """Евклидово расстояние между точками (x1,y1) и (x2,y2)."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ===========================================================================
# Вычисление коэффициента сжатия (Clerc-Kennedy)
# ===========================================================================

def _clerc_chi(c1: float, c2: float) -> float:
    """
    Вычислить коэффициент сжатия χ по формуле Клерка–Кеннеди.

    Условие применимости: phi = c1 + c2 > 4.

    χ = 2 / |2 - phi - sqrt(phi^2 - 4*phi)|

    Параметры
    ----------
    c1, c2 : float
        Когнитивный и социальный коэффициенты ускорения.

    Возвращает
    ----------
    float
        Коэффициент сжатия χ.
    """
    phi = c1 + c2
    if phi <= 4.0:
        raise ValueError(
            f"Для конструкции Клерка-Кеннеди необходимо c1+c2 > 4. "
            f"Получено phi = {phi:.4f}"
        )
    chi = 2.0 / abs(2.0 - phi - math.sqrt(phi ** 2 - 4.0 * phi))
    return chi


# ===========================================================================
# Основной алгоритм PSO
# ===========================================================================

def run_pso(params: dict, callback=None) -> tuple[dict, pd.DataFrame]:
    """
    Запустить PSO с заданными параметрами.

    Параметры (ключи словаря params)
    ---------------------------------
    mode        : str   — "basic" или "constriction"
    swarm_size  : int   — число частиц
    iters       : int   — число итераций
    seed        : int   — начальный seed ГСЧ
    w           : float — инерционный вес (только basic; default 0.7)
    c1          : float — когнитивный коэффициент (default 1.5 / 2.05)
    c2          : float — социальный коэффициент  (default 1.5 / 2.05)
    vmax        : float — максимальная скорость по модулю (None → без ограничений)
    callback : callable, optional
        Вызывается на каждой итерации: callback(it, total_iters, gbest_val).
        Используется GUI для отображения прогресса.

    Возвращает
    ----------
    result : dict
        Итоговые метрики: gbest_x, gbest_y, best_f, dx, df, evals, time_sec и т.д.
    trace_df : pd.DataFrame
        Статистика по итерациям.
    """

    # -- параметры с дефолтами -----------------------------------------------
    mode       = params.get("mode",       "basic")
    n_part     = int(params.get("swarm_size", 30))
    iters      = int(params.get("iters",       300))
    seed       = int(params.get("seed",        42))
    vmax_raw   = params.get("vmax",       None)
    vmax       = float(vmax_raw) if vmax_raw is not None else None
    # Досрочный останов, когда best_dx < порога (0 = отключён)
    stop_eps_dx = float(params.get("stop_eps_dx", 0.0))

    if mode == "basic":
        w  = float(params.get("w",  0.7))
        c1 = float(params.get("c1", 1.5))
        c2 = float(params.get("c2", 1.5))
        chi = None  # не используется
    else:  # constriction
        c1  = float(params.get("c1", 2.05))
        c2  = float(params.get("c2", 2.05))
        chi = _clerc_chi(c1, c2)
        w   = None  # не используется

    # -- инициализация --------------------------------------------------------
    rng = np.random.default_rng(seed)
    obj = CounterObjective()
    t_start = time.perf_counter()

    lo, hi = BOUNDS[0][0], BOUNDS[0][1]
    dim = 2  # размерность задачи (x, y)

    # Позиции: равномерно в [lo, hi]
    positions  = rng.uniform(lo, hi, size=(n_part, dim))

    # Скорости: небольшие случайные (±10% диапазона)
    v_init_scale = (hi - lo) * 0.1
    velocities   = rng.uniform(-v_init_scale, v_init_scale, size=(n_part, dim))

    # Оценить начальные позиции — эти значения используются как pbest_val
    # и как cur_vals первой итерации (без повторного вычисления)
    cur_vals  = np.array([obj(p[0], p[1]) for p in positions])
    pbest_pos = positions.copy()                 # лучшая позиция частицы
    pbest_val = cur_vals.copy()                  # значение в pbest

    # Глобальный лидер
    gbest_idx = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = float(pbest_val[gbest_idx])

    # -- история --------------------------------------------------------------
    trace_rows = []

    # -- главный цикл ---------------------------------------------------------
    for it in range(iters):

        # 1. Статистика по уже вычисленным значениям cur_vals
        mean_f    = float(np.mean(cur_vals))
        best_iter = float(np.min(cur_vals))  # лучшее в текущей итерации

        # Обновить gbest, если нашли лучше
        if best_iter < gbest_val:
            best_now_idx = int(np.argmin(cur_vals))
            gbest_pos    = positions[best_now_idx].copy()
            gbest_val    = best_iter

        best_dx = _euclidean(gbest_pos[0], gbest_pos[1], TRUE_MIN[0], TRUE_MIN[1])

        trace_rows.append({
            "iter":      it,
            "best_f":    gbest_val,
            "mean_f":    mean_f,
            "best_dx":   best_dx,
            "gbest_x":   gbest_pos[0],
            "gbest_y":   gbest_pos[1],
        })

        if callback is not None:
            callback(it, iters, gbest_val)

        # Досрочный останов по близости к истинному минимуму (если включён)
        if stop_eps_dx > 0.0 and best_dx < stop_eps_dx:
            break

        # 2. Обновить pbest для всех частиц
        improved = cur_vals < pbest_val
        pbest_pos[improved] = positions[improved].copy()
        pbest_val[improved] = cur_vals[improved]

        # 3. Обновить скорости и позиции
        r1 = rng.random(size=(n_part, dim))  # когнитивная случайность
        r2 = rng.random(size=(n_part, dim))  # социальная случайность

        if mode == "basic":
            # v(t+1) = w*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)
            velocities = (
                w * velocities
                + c1 * r1 * (pbest_pos - positions)
                + c2 * r2 * (gbest_pos - positions)
            )
        else:
            # v(t+1) = chi * [v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)]
            velocities = chi * (
                velocities
                + c1 * r1 * (pbest_pos - positions)
                + c2 * r2 * (gbest_pos - positions)
            )

        # Ограничить скорость (если задан vmax)
        if vmax is not None:
            speed = np.linalg.norm(velocities, axis=1, keepdims=True)
            too_fast = speed > vmax
            velocities[too_fast[:, 0]] = (
                velocities[too_fast[:, 0]] /
                speed[too_fast[:, 0]] * vmax
            )

        # Обновить позиции
        positions += velocities

        # Клиппинг в границы
        positions = np.clip(positions, lo, hi)

        # 4. Вычислить значения для обновлённых позиций (кешируются для следующей итерации)
        cur_vals = np.array([obj(p[0], p[1]) for p in positions])

    # -- финальный результат --------------------------------------------------
    # cur_vals содержит значения для последних позиций — используем их для gbest
    final_best_idx = int(np.argmin(cur_vals))
    if cur_vals[final_best_idx] < gbest_val:
        gbest_pos = positions[final_best_idx].copy()
        gbest_val = float(cur_vals[final_best_idx])

    best_dx = _euclidean(gbest_pos[0], gbest_pos[1], TRUE_MIN[0], TRUE_MIN[1])
    best_df = gbest_val - TRUE_F
    elapsed = time.perf_counter() - t_start

    result = {
        # параметры
        "mode":       mode,
        "swarm_size": n_part,
        "iters":      iters,
        "seed":       seed,
        "c1":         c1,
        "c2":         c2,
        "vmax":       vmax,
        # результаты
        "gbest_x":  float(gbest_pos[0]),
        "gbest_y":  float(gbest_pos[1]),
        "best_f":   gbest_val,
        "dx":       best_dx,
        "df":       best_df,
        "evals":    obj.evals,
        "time_sec": elapsed,
    }

    # Добавить специфичные для режима параметры
    if mode == "basic":
        result["w"] = w
    else:
        result["chi"] = chi

    trace_df = pd.DataFrame(trace_rows)
    return result, trace_df


# ===========================================================================
# GUI
# ===========================================================================

def main():
    """Точка входа при запуске `python pso_app.py` — открывает GUI-окно."""
    q: _queue.Queue = _queue.Queue()

    root = tk.Tk()
    root.title("PSO — Eggholder")
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

    mode_var,  _       = _row(pf, 0, "Режим PSO:",                    "basic",
                              ["basic", "constriction"])
    swarm_var, _       = _row(pf, 1, "Размер роя:",                    "30")
    iters_var, _       = _row(pf, 2, "Итераций:",                      "300")
    seed_var,  _       = _row(pf, 3, "Seed:",                          "42")
    w_var,     w_entry = _row(pf, 4, "Инерционный вес w:",             "0.7")
    c1_var,    _       = _row(pf, 5, "Когнитивный коэф. c1:",          "1.5")
    c2_var,    _       = _row(pf, 6, "Социальный коэф. c2:",           "1.5")
    vmax_var,  _       = _row(pf, 7, "vmax (пусто = без огранич.):",   "")
    epsdx_var, _       = _row(pf, 8, "stop_eps_dx (0=откл.):",         "0.0")

    def _on_mode(*_):
        if mode_var.get() == "constriction":
            w_entry.config(state="disabled")
            c1_var.set("2.05")
            c2_var.set("2.05")
        else:
            w_entry.config(state="normal")
            c1_var.set("1.5")
            c2_var.set("1.5")

    mode_var.trace_add("write", _on_mode)

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

    res_txt = tk.Text(rf, height=8, width=54, state="disabled",
                      font=("Courier", 9))
    res_txt.grid(row=0, column=0, sticky="nsew")

    def _show(r):
        res_txt.config(state="normal")
        res_txt.delete("1.0", "end")
        res_txt.insert("end",
                       f"Лучшая точка : x = {r['gbest_x']:.6f},  "
                       f"y = {r['gbest_y']:.6f}\n")
        res_txt.insert("end", f"f(x*, y*)     = {r['best_f']:.6f}\n")
        res_txt.insert("end", f"dx до истины  = {r['dx']:.6f}\n")
        res_txt.insert("end", f"df = f − f*   = {r['df']:.6f}\n")
        res_txt.insert("end", f"Вычислений f  : {r['evals']}\n")
        res_txt.insert("end", f"Итераций      : {r['iters']}\n")
        if r["mode"] == "constriction":
            res_txt.insert("end", f"Коэффициент χ : {r['chi']:.6f}\n")
        res_txt.insert("end", f"Время         : {r['time_sec']:.3f} с\n")
        res_txt.config(state="disabled")

    def _worker():
        try:
            params: dict = {
                "mode":       mode_var.get(),
                "swarm_size": int(swarm_var.get()),
                "iters":      int(iters_var.get()),
                "seed":       int(seed_var.get()),
                "c1":         float(c1_var.get()),
                "c2":         float(c2_var.get()),
                "stop_eps_dx": float(epsdx_var.get()),
            }
            if mode_var.get() == "basic":
                params["w"] = float(w_var.get())
            vmax_str = vmax_var.get().strip()
            if vmax_str:
                params["vmax"] = float(vmax_str)
        except ValueError as exc:
            q.put(("error", f"Ошибка в параметрах: {exc}"))
            return

        def cb(it, total, gbest_val):
            pct = int(it / max(total, 1) * 100)
            q.put(("progress", it, total, gbest_val, pct))

        try:
            result, trace_df = run_pso(params, callback=cb)
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
                    _, it, total, gbest_val, pct = msg
                    pb_var.set(pct)
                    pb_lbl.set(
                        f"Итерация {it}/{total},  best_f = {gbest_val:.4f}")
                elif msg[0] == "done":
                    result, trace_df = msg[1], msg[2]
                    status_var.set("Готово")
                    pb_var.set(100)
                    _show(result)
                    os.makedirs("out", exist_ok=True)
                    with open(os.path.join("out", "pso_last_result.json"),
                              "w", encoding="utf-8") as fh:
                        json.dump(result, fh, ensure_ascii=False, indent=2)
                    trace_df.to_csv(
                        os.path.join("out", "pso_last_trace.csv"), index=False)
                    pb_lbl.set(
                        "Сохранено: out/pso_last_result.json, "
                        "out/pso_last_trace.csv")
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


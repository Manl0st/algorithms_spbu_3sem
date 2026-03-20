"""
pso_app.py
==========
Роевой алгоритм (Particle Swarm Optimisation, PSO) для минимизации функции Eggholder.

Поддерживает два режима:
  - basic        : классический PSO с инерционным весом w
  - constriction : PSO с коэффициентом сжатия χ (Clerc & Kennedy, 2002)

Как пользоваться
----------------
Запустите GUI::

    python pso_app.py

В окне настройте параметры и нажмите «Запустить».
Результаты сохраняются в папке out/:
  - pso_last_result.json — параметры и итоговые метрики
  - pso_last_trace.csv   — трасса сходимости по итерациям (с русскими заголовками)

Для использования из run_experiments.py доступна функция::

    result, trace_df = run_pso(params: dict)
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
    stop_eps_dx : float — останов по близости к минимуму (0 = отключён)
    viz_every   : int   — передавать позиции в callback каждые N итераций (0 = никогда)
    callback : callable, optional
        Вызывается на каждой итерации:
          callback(it, total_iters, gbest_val, positions_xy=None, gbest_xy=None)
        positions_xy — np.array (swarm_size, 2) текущих позиций (только при viz_every)
        gbest_xy     — np.array [gbest_x, gbest_y]
        Используется GUI для отображения прогресса и визуализации.

    Возвращает
    ----------
    result : dict
        Итоговые метрики: gbest_x, gbest_y, best_f, dx, df, evals, time_sec и т.д.
    trace_df : pd.DataFrame
        Статистика по итерациям (внутренние имена столбцов).
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
    viz_every   = int(params.get("viz_every",  0))  # 0 = не передавать позиции

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

    dim = 2  # размерность задачи (x, y)

    # FIX: раздельные нижние и верхние границы для каждой переменной
    lows  = np.array([b[0] for b in BOUNDS])   # [-512, -512]
    highs = np.array([b[1] for b in BOUNDS])   # [ 512,  512]

    # Позиции: равномерно в [lows, highs] по каждой переменной
    positions  = rng.uniform(lows, highs, size=(n_part, dim))

    # Скорости: небольшие случайные (±10% диапазона каждой переменной)
    v_init_scales = (highs - lows) * 0.1
    velocities    = rng.uniform(-v_init_scales, v_init_scales, size=(n_part, dim))

    # Оценить начальные позиции
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

        # Подготовить данные для визуализации (только при viz_every)
        positions_xy = None
        gbest_xy_arr = None
        if callback is not None and viz_every > 0 and it % viz_every == 0:
            positions_xy = positions.copy()
            gbest_xy_arr = gbest_pos.copy()

        if callback is not None:
            callback(it, iters, gbest_val, positions_xy, gbest_xy_arr)

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

        # Ограничить скорость (если задан vmax): покомпонентный клиппинг
        if vmax is not None:
            velocities = np.clip(velocities, -vmax, vmax)

        # Обновить позиции
        positions += velocities

        # FIX: клиппинг с раздельными границами по каждой переменной
        positions = np.clip(positions, lows, highs)

        # 4. Вычислить значения для обновлённых позиций
        cur_vals = np.array([obj(p[0], p[1]) for p in positions])

    # -- финальный результат --------------------------------------------------
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
        "iters_done": len(trace_rows),  # фактически выполненных итераций
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

# Словари для перевода между человекочитаемыми именами и внутренними значениями
_MODE_LABELS = {
    "Без коэффициента сжатия":    "basic",
    "С коэффициентом сжатия (χ)": "constriction",
}
_MODE_LABELS_INV = {v: k for k, v in _MODE_LABELS.items()}

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
    """Точка входа при запуске `python pso_app.py` — открывает GUI-окно."""
    q: _queue.Queue = _queue.Queue()

    root = tk.Tk()
    root.title("PSO — Eggholder")
    root.resizable(False, False)

    # =========================================================================
    # Переменные параметров (только ключевые)
    # =========================================================================
    mode_var  = tk.StringVar(value="Без коэффициента сжатия")
    swarm_var = tk.StringVar(value="30")
    iters_var = tk.StringVar(value="300")
    w_var     = tk.StringVar(value="0.7")
    c1_var    = tk.StringVar(value="1.5")
    c2_var    = tk.StringVar(value="1.5")

    # Кэш значений w/c1/c2 для каждого режима
    _mode_cache = {
        "basic":        {"w": "0.7",   "c1": "1.5",  "c2": "1.5"},
        "constriction": {"c1": "2.05", "c2": "2.05"},
    }
    _current_mode = ["basic"]

    # =========================================================================
    # Основной контейнер (2 колонки: параметры + визуализация)
    # =========================================================================
    main_frame = ttk.Frame(root)
    main_frame.grid(row=0, column=0, sticky="nsew")

    # ----- Левая колонка: параметры -----
    pf = ttk.LabelFrame(main_frame, text="Параметры PSO", padding=8)
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

    _field(pf, 0, "Модификация:", mode_var, list(_MODE_LABELS.keys()),
           width=22)
    _field(pf, 1, "Число частиц:", swarm_var)
    _field(pf, 2, "Число итераций:", iters_var)
    w_entry = _field(pf, 3, "w:", w_var)
    _field(pf, 4, "c1:", c1_var)
    _field(pf, 5, "c2:", c2_var)

    def _on_mode(*_):
        new_mode = _MODE_LABELS.get(mode_var.get(), "basic")
        old_mode = _current_mode[0]
        if new_mode == old_mode:
            return
        # Сохранить текущие значения в кэш старого режима
        _mode_cache[old_mode]["c1"] = c1_var.get()
        _mode_cache[old_mode]["c2"] = c2_var.get()
        if old_mode == "basic":
            _mode_cache[old_mode]["w"] = w_var.get()
        # Загрузить значения нового режима
        _current_mode[0] = new_mode
        cached = _mode_cache[new_mode]
        c1_var.set(cached["c1"])
        c2_var.set(cached["c2"])
        if new_mode == "constriction":
            w_entry.config(state="disabled")
        else:
            w_entry.config(state="normal")
            w_var.set(cached.get("w", "0.7"))

    mode_var.trace_add("write", _on_mode)

    # ----- Правая колонка: визуализация -----
    vf = ttk.LabelFrame(main_frame, text="Визуализация роя", padding=4)
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
              text="● Синие: частицы   ● Оранжевая: лучший   ★ Красная: минимум",
              font=("", 8), foreground="#555555").grid(
        row=1, column=0, sticky="w", padx=2, pady=(2, 0))

    # Живая статистика
    info_var = tk.StringVar(value="")
    ttk.Label(vf, textvariable=info_var, font=("Courier", 9),
              foreground="#225599").grid(row=2, column=0, sticky="w",
                                         padx=4, pady=(2, 4))

    # Буфер следа лучшей позиции (последние N точек)
    _best_trail: list = []
    _TRAIL_LEN = 30

    def _redraw_viz(it: int, total: int, gbest_val: float,
                    positions_xy: np.ndarray, gbest_xy: np.ndarray):
        """Перерисовать холст с текущим роем, следом и статистикой."""
        viz_canvas.delete("swarm")

        if gbest_xy is not None:
            tx, ty = _world_to_canvas(gbest_xy[0], gbest_xy[1])
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
                                   fill=trail_color, outline="", tags="swarm")

        if positions_xy is not None:
            for xi, yi in positions_xy:
                cx, cy = _world_to_canvas(xi, yi)
                viz_canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                                       fill="#4488cc", outline="", tags="swarm")

        if gbest_xy is not None:
            bx_c, by_c = _world_to_canvas(gbest_xy[0], gbest_xy[1])
            viz_canvas.create_oval(bx_c - 6, by_c - 6, bx_c + 6, by_c + 6,
                                   fill="#ff8800", outline="#ffffff", width=1,
                                   tags="swarm")

        viz_canvas.tag_raise("true_min")

        if gbest_xy is not None:
            info_var.set(
                f"Итерация: {it + 1} / {total}\n"
                f"Лучшее f: {gbest_val:.4f}\n"
                f"Лучшая точка: ({gbest_xy[0]:.2f}, {gbest_xy[1]:.2f})"
            )

    def _clear_viz():
        _best_trail.clear()
        viz_canvas.delete("swarm")
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
                       f"Лучшая точка : x = {r['gbest_x']:.6f},  "
                       f"y = {r['gbest_y']:.6f}\n")
        res_txt.insert("end", f"f(x, y)       = {r['best_f']:.6f}\n")
        res_txt.insert("end", f"dx до истины  = {r['dx']:.6f}\n")
        res_txt.insert("end", f"df = f − f*   = {r['df']:.6f}\n")
        res_txt.insert("end", f"Вычислений f  : {r['evals']}\n")
        res_txt.insert("end", f"Итераций      : {r['iters_done']}\n")
        if r["mode"] == "constriction":
            res_txt.insert("end", f"Коэффициент χ : {r['chi']:.6f}\n")
        res_txt.insert("end", f"Время         : {r['time_sec']:.3f} с\n")
        res_txt.config(state="disabled")

    # Трасса PSO с русскими заголовками столбцов
    _TRACE_RU_COLUMNS = {
        "iter":    "итерация",
        "best_f":  "лучшее_f",
        "mean_f":  "среднее_f",
        "best_dx": "dx_лучшей",
        "gbest_x": "лучший_x",
        "gbest_y": "лучший_y",
    }

    # =========================================================================
    # Рабочий поток
    # =========================================================================
    def _worker():
        try:
            mode_internal = _MODE_LABELS.get(mode_var.get(), "basic")
            params: dict = {
                "mode":       mode_internal,
                "swarm_size": int(swarm_var.get()),
                "iters":      int(iters_var.get()),
                "c1":         float(c1_var.get()),
                "c2":         float(c2_var.get()),
                "viz_every":  10,
            }
            if mode_internal == "basic":
                params["w"] = float(w_var.get())
        except ValueError as exc:
            q.put(("error", f"Ошибка в параметрах: {exc}"))
            return

        def cb(it, total, gbest_val, positions_xy=None, gbest_xy=None):
            pct = int(it / max(total, 1) * 100)
            q.put(("progress", it, total, gbest_val, pct, positions_xy, gbest_xy))

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
        _clear_viz()
        threading.Thread(target=_worker, daemon=True).start()
        root.after(50, _poll)

    def _poll():
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == "progress":
                    _, it, total, gbest_val, pct, positions_xy, gbest_xy = msg
                    pb_var.set(pct)
                    pb_lbl.set(f"Итерация {it}/{total},  f = {gbest_val:.4f}")
                    if positions_xy is not None:
                        _redraw_viz(it, total, gbest_val, positions_xy, gbest_xy)
                elif msg[0] == "done":
                    result, trace_df = msg[1], msg[2]
                    status_var.set("Готово")
                    pb_var.set(100)
                    _show(result)
                    os.makedirs("out", exist_ok=True)
                    with open(os.path.join("out", "pso_last_result.json"),
                              "w", encoding="utf-8") as fh:
                        json.dump(result, fh, ensure_ascii=False, indent=2)
                    trace_ru = trace_df.rename(columns=_TRACE_RU_COLUMNS)
                    trace_ru.to_csv(
                        os.path.join("out", "pso_last_trace.csv"), index=False,
                        encoding="utf-8")
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

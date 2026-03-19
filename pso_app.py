"""
pso_app.py
==========
Роевой алгоритм (Particle Swarm Optimisation, PSO) для минимизации функции Eggholder.

Поддерживает два режима:
  - basic        : классический PSO с инерционным весом w
  - constriction : PSO с коэффициентом сжатия χ (Clerc & Kennedy, 2002)

Запуск в интерактивном режиме::

    python pso_app.py

Для использования из run_experiments.py доступна функция::

    result, trace_df = run_pso(params: dict)
"""

import json
import math
import os
import time

import numpy as np
import pandas as pd

from objective import BOUNDS, TRUE_F, TRUE_MIN, CounterObjective

# ---------------------------------------------------------------------------
# Вспомогательные утилиты
# ---------------------------------------------------------------------------

def _ask(prompt: str, default):
    """Запросить ввод с подсказкой; пустой ввод → default."""
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    return type(default)(raw)


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

def run_pso(params: dict) -> tuple[dict, pd.DataFrame]:
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
# Интерактивный запуск
# ===========================================================================

def _interactive_params() -> dict:
    """Запросить параметры PSO у пользователя через input() с дефолтами."""

    print("\n=== PSO — Eggholder ===")
    print("Нажмите Enter для принятия значения по умолчанию.\n")

    mode = _ask("Режим PSO (basic / constriction)", "basic")
    if mode not in ("basic", "constriction"):
        print(f"Неизвестный режим '{mode}', используется 'basic'.")
        mode = "basic"

    swarm_size = int(_ask("Размер роя (swarm_size)", 30))
    iters      = int(_ask("Число итераций (iters)", 300))
    seed       = int(_ask("Начальный seed", 42))

    if mode == "basic":
        w  = float(_ask("Инерционный вес w", 0.7))
        c1 = float(_ask("Когнитивный коэффициент c1", 1.5))
        c2 = float(_ask("Социальный коэффициент c2", 1.5))
        params = {"mode": mode, "swarm_size": swarm_size, "iters": iters,
                  "seed": seed, "w": w, "c1": c1, "c2": c2}
    else:
        c1 = float(_ask("Когнитивный коэффициент c1 (нужен c1+c2 > 4)", 2.05))
        c2 = float(_ask("Социальный коэффициент c2", 2.05))
        params = {"mode": mode, "swarm_size": swarm_size, "iters": iters,
                  "seed": seed, "c1": c1, "c2": c2}

    vmax_str = input(f"Макс. скорость vmax (Enter = без ограничений): ").strip()
    if vmax_str:
        params["vmax"] = float(vmax_str)

    return params


def main():
    """Точка входа при запуске `python pso_app.py`."""

    params = _interactive_params()
    result, trace_df = run_pso(params)

    # -- вывод результатов ---------------------------------------------------
    print("\n--- Результаты PSO ---")
    print(f"  Лучшая точка  : x = {result['gbest_x']:.6f}, y = {result['gbest_y']:.6f}")
    print(f"  f(x*, y*)     : {result['best_f']:.6f}")
    print(f"  dx до истины  : {result['dx']:.6f}")
    print(f"  df = f - f*   : {result['df']:.6f}")
    print(f"  Вычислений f  : {result['evals']}")
    print(f"  Итераций      : {result['iters']}")
    print(f"  Время         : {result['time_sec']:.3f} с")

    # -- сохранение результатов ----------------------------------------------
    os.makedirs("out", exist_ok=True)

    json_path = os.path.join("out", "pso_last_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nРезультат сохранён: {json_path}")

    csv_path = os.path.join("out", "pso_last_trace.csv")
    trace_df.to_csv(csv_path, index=False)
    print(f"Трасса сохранена : {csv_path}")


if __name__ == "__main__":
    main()

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
import time

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)
import pyqtgraph as pg

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

# Трасса PSO с русскими заголовками столбцов
_TRACE_RU_COLUMNS = {
    "iter":    "итерация",
    "best_f":  "лучшее_f",
    "mean_f":  "среднее_f",
    "best_dx": "dx_лучшей",
    "gbest_x": "лучший_x",
    "gbest_y": "лучший_y",
}


class _PSOWorker(QThread):
    """Рабочий поток для запуска PSO без блокировки UI."""

    progress = pyqtSignal(int, int, float, object, object)  # it, total, gbest_val, positions_xy, gbest_xy
    done     = pyqtSignal(dict, object)   # result, trace_df
    error    = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self._params = params

    def run(self):
        def cb(it, total, gbest_val, positions_xy=None, gbest_xy=None):
            self.progress.emit(it, total, gbest_val, positions_xy, gbest_xy)

        try:
            result, trace_df = run_pso(self._params, callback=cb)
            self.done.emit(result, trace_df)
        except Exception as exc:
            self.error.emit(str(exc))


class _PSOWindow(QWidget):
    """Главное окно PSO — Eggholder."""

    _TRAIL_LEN = 30

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PSO — Eggholder")
        self._best_trail: list[tuple[float, float]] = []
        self._worker: _PSOWorker | None = None
        # Кэш значений w/c1/c2 для каждого режима
        self._mode_cache = {
            "basic":        {"w": "0.7",   "c1": "1.5",  "c2": "1.5"},
            "constriction": {"c1": "2.05", "c2": "2.05"},
        }
        self._current_mode = "basic"
        self._build_ui()

    # ------------------------------------------------------------------
    # Построение UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)

        # ---- Верхняя строка: параметры + визуализация ----
        top_row = QHBoxLayout()
        root_layout.addLayout(top_row)

        # Левая колонка: параметры
        params_box = QGroupBox("Параметры PSO")
        params_layout = QFormLayout(params_box)
        params_layout.setContentsMargins(8, 12, 8, 8)
        params_layout.setSpacing(6)
        top_row.addWidget(params_box, stretch=0)

        self._mode_cb   = QComboBox(); self._mode_cb.addItems(list(_MODE_LABELS.keys()))
        self._mode_cb.setMinimumWidth(200)
        self._swarm_ed  = QLineEdit("30");   self._swarm_ed.setMaximumWidth(80)
        self._iters_ed  = QLineEdit("300");  self._iters_ed.setMaximumWidth(80)
        self._w_ed      = QLineEdit("0.7");  self._w_ed.setMaximumWidth(80)
        self._c1_ed     = QLineEdit("1.5");  self._c1_ed.setMaximumWidth(80)
        self._c2_ed     = QLineEdit("1.5");  self._c2_ed.setMaximumWidth(80)

        params_layout.addRow("Модификация:",    self._mode_cb)
        params_layout.addRow("Число частиц:",   self._swarm_ed)
        params_layout.addRow("Число итераций:", self._iters_ed)
        params_layout.addRow("w:",              self._w_ed)
        params_layout.addRow("c1:",             self._c1_ed)
        params_layout.addRow("c2:",             self._c2_ed)

        self._mode_cb.currentTextChanged.connect(self._on_mode_changed)

        # Правая колонка: визуализация
        viz_box = QGroupBox("Визуализация роя")
        viz_layout = QVBoxLayout(viz_box)
        viz_layout.setContentsMargins(6, 12, 6, 6)
        viz_layout.setSpacing(4)
        top_row.addWidget(viz_box, stretch=1)

        pg.setConfigOptions(antialias=True, background="#1a1a2e", foreground="#cccccc")
        self._plot = pg.PlotWidget()
        self._plot.setMinimumSize(440, 440)
        self._plot.setAspectLocked(True)
        bx_lo, bx_hi = BOUNDS[0]
        by_lo, by_hi = BOUNDS[1]
        self._plot.setXRange(bx_lo, bx_hi, padding=0)
        self._plot.setYRange(by_lo, by_hi, padding=0)
        self._plot.getAxis("bottom").setLabel("x")
        self._plot.getAxis("left").setLabel("y")
        viz_layout.addWidget(self._plot)

        # Легенда
        legend_lbl = QLabel("● Синие: частицы   ● Оранжевая: лучший   ★ Красная: минимум")
        legend_lbl.setStyleSheet("color: #888888; font-size: 9px;")
        viz_layout.addWidget(legend_lbl)

        # Живая статистика
        self._info_lbl = QLabel("")
        mono = QFont("Monospace", 9)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self._info_lbl.setFont(mono)
        self._info_lbl.setStyleSheet("color: #4488cc;")
        viz_layout.addWidget(self._info_lbl)

        # Постоянные элементы графика
        self._swarm_scatter = pg.ScatterPlotItem(size=6,  pen=None, brush=pg.mkBrush("#4488cc"))
        self._best_scatter  = pg.ScatterPlotItem(size=12, pen=pg.mkPen("#ffffff", width=1),
                                                 brush=pg.mkBrush("#ff8800"))
        self._trail_curve   = pg.PlotCurveItem(pen=pg.mkPen("#00cc44", width=1.5))
        self._true_min_sct  = pg.ScatterPlotItem(
            pos=[[TRUE_MIN[0], TRUE_MIN[1]]], size=16,
            symbol="star", pen=pg.mkPen("#ff4444", width=1),
            brush=pg.mkBrush("#ff4444"))

        self._plot.addItem(self._trail_curve)
        self._plot.addItem(self._swarm_scatter)
        self._plot.addItem(self._best_scatter)
        self._plot.addItem(self._true_min_sct)

        # ---- Управление ----
        ctrl_layout = QHBoxLayout()
        root_layout.addLayout(ctrl_layout)

        self._run_btn = QPushButton("▶  Запустить")
        self._run_btn.clicked.connect(self._on_start)
        ctrl_layout.addWidget(self._run_btn)

        self._status_lbl = QLabel("Готово")
        bold = QFont(); bold.setBold(True); bold.setPointSize(10)
        self._status_lbl.setFont(bold)
        ctrl_layout.addWidget(self._status_lbl)
        ctrl_layout.addStretch()

        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        root_layout.addWidget(self._progress_bar)

        self._pb_lbl = QLabel("")
        self._pb_lbl.setStyleSheet("font-size: 8pt;")
        root_layout.addWidget(self._pb_lbl)

        # ---- Результаты ----
        res_box = QGroupBox("Результаты")
        res_layout = QVBoxLayout(res_box)
        root_layout.addWidget(res_box)

        self._res_txt = QTextEdit()
        self._res_txt.setReadOnly(True)
        mono2 = QFont("Monospace", 9)
        mono2.setStyleHint(QFont.StyleHint.Monospace)
        self._res_txt.setFont(mono2)
        self._res_txt.setFixedHeight(150)
        res_layout.addWidget(self._res_txt)

    # ------------------------------------------------------------------
    # Логика UI
    # ------------------------------------------------------------------

    def _on_mode_changed(self, text: str):
        new_mode = _MODE_LABELS.get(text, "basic")
        old_mode = self._current_mode
        if new_mode == old_mode:
            return
        # Сохранить текущие значения в кэш старого режима
        self._mode_cache[old_mode]["c1"] = self._c1_ed.text()
        self._mode_cache[old_mode]["c2"] = self._c2_ed.text()
        if old_mode == "basic":
            self._mode_cache[old_mode]["w"] = self._w_ed.text()
        # Загрузить значения нового режима
        self._current_mode = new_mode
        cached = self._mode_cache[new_mode]
        self._c1_ed.setText(cached["c1"])
        self._c2_ed.setText(cached["c2"])
        if new_mode == "constriction":
            self._w_ed.setEnabled(False)
        else:
            self._w_ed.setEnabled(True)
            self._w_ed.setText(cached.get("w", "0.7"))

    def _clear_viz(self):
        self._best_trail.clear()
        self._swarm_scatter.setData([], [])
        self._best_scatter.setData([], [])
        self._trail_curve.setData([], [])
        self._info_lbl.setText("")

    def _redraw_viz(self, it: int, total: int, gbest_val: float,
                    positions_xy, gbest_xy):
        if gbest_xy is not None:
            self._best_trail.append((float(gbest_xy[0]), float(gbest_xy[1])))
            if len(self._best_trail) > self._TRAIL_LEN:
                del self._best_trail[:-self._TRAIL_LEN]

        # Trail
        if len(self._best_trail) >= 2:
            tx = [p[0] for p in self._best_trail]
            ty = [p[1] for p in self._best_trail]
            self._trail_curve.setData(tx, ty)

        # Swarm
        if positions_xy is not None and len(positions_xy) > 0:
            self._swarm_scatter.setData(
                x=positions_xy[:, 0].tolist(),
                y=positions_xy[:, 1].tolist(),
            )

        # Best particle
        if gbest_xy is not None:
            self._best_scatter.setData(
                x=[float(gbest_xy[0])],
                y=[float(gbest_xy[1])],
            )

        if gbest_xy is not None:
            self._info_lbl.setText(
                f"Итерация: {it + 1} / {total}\n"
                f"Лучшее f: {gbest_val:.4f}\n"
                f"Лучшая точка: ({gbest_xy[0]:.2f}, {gbest_xy[1]:.2f})"
            )

    # ------------------------------------------------------------------
    # Запуск / результаты
    # ------------------------------------------------------------------

    def _on_start(self):
        mode_internal = _MODE_LABELS.get(self._mode_cb.currentText(), "basic")
        try:
            params: dict = {
                "mode":       mode_internal,
                "swarm_size": int(self._swarm_ed.text()),
                "iters":      int(self._iters_ed.text()),
                "c1":         float(self._c1_ed.text()),
                "c2":         float(self._c2_ed.text()),
                "viz_every":  10,
            }
            if mode_internal == "basic":
                params["w"] = float(self._w_ed.text())
        except ValueError as exc:
            QMessageBox.critical(self, "Ошибка", f"Ошибка в параметрах: {exc}")
            return

        self._run_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._pb_lbl.setText("")
        self._status_lbl.setText("Выполняется…")
        self._clear_viz()

        self._worker = _PSOWorker(params)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, it: int, total: int, gbest_val: float,
                     positions_xy, gbest_xy):
        pct = int(it / max(total, 1) * 100)
        self._progress_bar.setValue(pct)
        self._pb_lbl.setText(f"Итерация {it}/{total},  f = {gbest_val:.4f}")
        if positions_xy is not None:
            self._redraw_viz(it, total, gbest_val, positions_xy, gbest_xy)

    def _on_done(self, result: dict, trace_df):
        self._status_lbl.setText("Готово")
        self._progress_bar.setValue(100)
        self._show_result(result)
        os.makedirs("out", exist_ok=True)
        with open(os.path.join("out", "pso_last_result.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        trace_ru = trace_df.rename(columns=_TRACE_RU_COLUMNS)
        trace_ru.to_csv(
            os.path.join("out", "pso_last_trace.csv"), index=False,
            encoding="utf-8")
        self._pb_lbl.setText(
            "Сохранено: out/pso_last_result.json, out/pso_last_trace.csv")
        self._run_btn.setEnabled(True)

    def _on_error(self, msg: str):
        self._status_lbl.setText("Ошибка")
        QMessageBox.critical(self, "Ошибка", msg)
        self._run_btn.setEnabled(True)

    def _show_result(self, r: dict):
        lines = [
            f"Лучшая точка : x = {r['gbest_x']:.6f},  y = {r['gbest_y']:.6f}",
            f"f(x, y)       = {r['best_f']:.6f}",
            f"dx до истины  = {r['dx']:.6f}",
            f"df = f − f*   = {r['df']:.6f}",
            f"Вычислений f  : {r['evals']}",
            f"Итераций      : {r['iters_done']}",
        ]
        if r["mode"] == "constriction":
            lines.append(f"Коэффициент χ : {r['chi']:.6f}")
        lines.append(f"Время         : {r['time_sec']:.3f} с")
        self._res_txt.setPlainText("\n".join(lines))


def main():
    """Точка входа при запуске `python pso_app.py` — открывает GUI-окно."""
    app = QApplication.instance() or QApplication([])
    window = _PSOWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

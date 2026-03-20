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


class _GAWorker(QThread):
    """Рабочий поток для запуска ГА без блокировки UI."""

    progress = pyqtSignal(int, int, float, object, object)  # gen, gens, best_f, positions_xy, best_xy
    done     = pyqtSignal(dict, object)   # result, trace_df
    error    = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self._params = params

    def run(self):
        def cb(gen, gens, best_f, positions_xy=None, best_xy=None):
            self.progress.emit(gen, gens, best_f, positions_xy, best_xy)

        try:
            result, trace_df = run_ga(self._params, callback=cb)
            self.done.emit(result, trace_df)
        except Exception as exc:
            self.error.emit(str(exc))


class _GAWindow(QWidget):
    """Главное окно ГА — Eggholder."""

    _TRAIL_LEN = 30

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ГА — Eggholder")
        self._best_trail: list[tuple[float, float]] = []
        self._worker: _GAWorker | None = None
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
        params_box = QGroupBox("Параметры ГА")
        params_layout = QFormLayout(params_box)
        params_layout.setContentsMargins(8, 12, 8, 8)
        params_layout.setSpacing(6)
        top_row.addWidget(params_box, stretch=0)

        self._enc_cb  = QComboBox(); self._enc_cb.addItems(list(_ENC_LABELS.keys()))
        self._cx_cb   = QComboBox(); self._cx_cb.addItems(list(_CX_BIN_LABELS.keys()))
        self._tk_ed   = QLineEdit("3");   self._tk_ed.setMaximumWidth(80)
        self._eli_ed  = QLineEdit("1");   self._eli_ed.setMaximumWidth(80)
        self._pop_ed  = QLineEdit("50");  self._pop_ed.setMaximumWidth(80)
        self._gen_ed  = QLineEdit("300"); self._gen_ed.setMaximumWidth(80)
        self._cxr_ed  = QLineEdit("0.9"); self._cxr_ed.setMaximumWidth(80)
        self._mut_ed  = QLineEdit("0.02"); self._mut_ed.setMaximumWidth(80)

        params_layout.addRow("Тип кодирования:",   self._enc_cb)
        params_layout.addRow("Тип кроссинговера:",  self._cx_cb)
        params_layout.addRow("Размер турнира:",      self._tk_ed)
        params_layout.addRow("Элитизм:",             self._eli_ed)
        params_layout.addRow("Размер популяции:",    self._pop_ed)
        params_layout.addRow("Число поколений:",     self._gen_ed)
        params_layout.addRow("Вер. кроссинговера:",  self._cxr_ed)
        params_layout.addRow("Вер. мутации:",        self._mut_ed)

        self._enc_cb.currentTextChanged.connect(self._on_enc_changed)

        # Правая колонка: визуализация
        viz_box = QGroupBox("Визуализация популяции")
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
        legend_lbl = QLabel("● Синие: особи   ● Оранжевая: лучший   ★ Красная: минимум")
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
        self._pop_scatter  = pg.ScatterPlotItem(size=6,  pen=None, brush=pg.mkBrush("#4488cc"))
        self._best_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen("#ffffff", width=1),
                                                brush=pg.mkBrush("#ff8800"))
        self._trail_curve  = pg.PlotCurveItem(pen=pg.mkPen("#00cc44", width=1.5))
        self._true_min_sct = pg.ScatterPlotItem(
            pos=[[TRUE_MIN[0], TRUE_MIN[1]]], size=16,
            symbol="star", pen=pg.mkPen("#ff4444", width=1),
            brush=pg.mkBrush("#ff4444"))

        self._plot.addItem(self._trail_curve)
        self._plot.addItem(self._pop_scatter)
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
        self._res_txt.setFixedHeight(140)
        res_layout.addWidget(self._res_txt)

    # ------------------------------------------------------------------
    # Логика UI
    # ------------------------------------------------------------------

    def _on_enc_changed(self, text: str):
        if text == "Вещественное":
            self._cx_cb.clear()
            self._cx_cb.addItems(list(_CX_REAL_LABELS.keys()))
            self._cx_cb.setEnabled(False)
        else:
            self._cx_cb.clear()
            self._cx_cb.addItems(list(_CX_BIN_LABELS.keys()))
            self._cx_cb.setEnabled(True)

    def _clear_viz(self):
        self._best_trail.clear()
        self._pop_scatter.setData([], [])
        self._best_scatter.setData([], [])
        self._trail_curve.setData([], [])
        self._info_lbl.setText("")

    def _redraw_viz(self, gen: int, total: int, best_f: float,
                    positions_xy, best_xy):
        if best_xy is not None:
            self._best_trail.append((float(best_xy[0]), float(best_xy[1])))
            if len(self._best_trail) > self._TRAIL_LEN:
                del self._best_trail[:-self._TRAIL_LEN]

        # Trail
        if len(self._best_trail) >= 2:
            tx = [p[0] for p in self._best_trail]
            ty = [p[1] for p in self._best_trail]
            self._trail_curve.setData(tx, ty)

        # Population
        if positions_xy is not None and len(positions_xy) > 0:
            self._pop_scatter.setData(
                x=positions_xy[:, 0].tolist(),
                y=positions_xy[:, 1].tolist(),
            )

        # Best individual
        if best_xy is not None:
            self._best_scatter.setData(
                x=[float(best_xy[0])],
                y=[float(best_xy[1])],
            )

        if best_xy is not None:
            self._info_lbl.setText(
                f"Поколение: {gen + 1} / {total}\n"
                f"Лучшее f: {best_f:.4f}\n"
                f"Лучшая точка: ({best_xy[0]:.2f}, {best_xy[1]:.2f})"
            )

    # ------------------------------------------------------------------
    # Запуск / результаты
    # ------------------------------------------------------------------

    def _on_start(self):
        enc_text = self._enc_cb.currentText()
        enc_internal = _ENC_LABELS.get(enc_text, "binary")
        cx_label = self._cx_cb.currentText()
        if enc_internal == "binary":
            cx_internal = _CX_BIN_LABELS.get(cx_label, "one_point")
        else:
            cx_internal = _CX_REAL_LABELS.get(cx_label, "arithmetic")

        try:
            params = {
                "encoding":       enc_internal,
                "pop_size":       int(self._pop_ed.text()),
                "generations":    int(self._gen_ed.text()),
                "crossover_type": cx_internal,
                "crossover_rate": float(self._cxr_ed.text()),
                "mutation_rate":  float(self._mut_ed.text()),
                "tournament_k":   int(self._tk_ed.text()),
                "elitism":        int(self._eli_ed.text()),
                "viz_every":      10,
            }
        except ValueError as exc:
            QMessageBox.critical(self, "Ошибка", f"Ошибка в параметрах: {exc}")
            return

        self._run_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._pb_lbl.setText("")
        self._status_lbl.setText("Выполняется…")
        self._clear_viz()

        self._worker = _GAWorker(params)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, gen: int, gens: int, best_f: float,
                     positions_xy, best_xy):
        pct = int((gen + 1) / max(gens, 1) * 100)
        self._progress_bar.setValue(pct)
        self._pb_lbl.setText(f"Поколение {gen}/{gens},  f = {best_f:.4f}")
        if positions_xy is not None:
            self._redraw_viz(gen, gens, best_f, positions_xy, best_xy)

    def _on_done(self, result: dict, trace_df):
        self._status_lbl.setText("Готово")
        self._progress_bar.setValue(100)
        self._show_result(result)
        os.makedirs("out", exist_ok=True)
        with open(os.path.join("out", "ga_last_result.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        trace_ru = trace_df.rename(columns=_TRACE_RU_COLUMNS)
        trace_ru.to_csv(
            os.path.join("out", "ga_last_trace.csv"), index=False,
            encoding="utf-8")
        self._pb_lbl.setText(
            "Сохранено: out/ga_last_result.json, out/ga_last_trace.csv")
        self._run_btn.setEnabled(True)

    def _on_error(self, msg: str):
        self._status_lbl.setText("Ошибка")
        QMessageBox.critical(self, "Ошибка", msg)
        self._run_btn.setEnabled(True)

    def _show_result(self, r: dict):
        lines = [
            f"Лучшая точка : x = {r['best_x']:.6f},  y = {r['best_y']:.6f}",
            f"f(x, y)       = {r['best_f']:.6f}",
            f"dx до истины  = {r['dx']:.6f}",
            f"df = f − f*   = {r['df']:.6f}",
            f"Вычислений f  : {r['evals']}",
            f"Поколений     : {r['gens_done']}",
            f"Время         : {r['time_sec']:.3f} с",
        ]
        self._res_txt.setPlainText("\n".join(lines))


def main():
    """Точка входа при запуске `python ga_app.py` — открывает GUI-окно."""
    app = QApplication.instance() or QApplication([])
    window = _GAWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

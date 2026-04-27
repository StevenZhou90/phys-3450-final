from __future__ import annotations

from typing import Callable

import numpy as np


def state_overlap_kernel(
    xs: np.ndarray,
    ys: np.ndarray,
    state_fn: Callable[[np.ndarray | float], object],
) -> np.ndarray:
    x_states = [state_fn(x) for x in xs]
    y_states = [state_fn(y) for y in ys]

    kernel = np.empty((len(xs), len(ys)), dtype=float)
    for i, state_x in enumerate(x_states):
        for j, state_y in enumerate(y_states):
            overlap = state_x.data.conj() @ state_y.data
            kernel[i, j] = float(np.abs(overlap) ** 2)
    return kernel


def one_qubit_analytic_kernel(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)[:, None]
    ys = np.asarray(ys, dtype=float)[None, :]
    return np.cos((xs - ys) / 2.0) ** 2


def frobenius_relative_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    numerator = np.linalg.norm(reference - estimate)
    denominator = np.linalg.norm(reference)
    return float(numerator / denominator)

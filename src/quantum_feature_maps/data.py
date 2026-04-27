from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class DatasetBundle:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    x_all: np.ndarray
    y_all: np.ndarray


def make_one_qubit_grid(num_points: int = 60) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, num_points)


def make_two_qubit_classification_dataset(
    n_samples: int = 240,
    noise: float = 0.16,
    random_state: int = 7,
) -> DatasetBundle:
    x_raw, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_raw,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    x_train = scaler.fit_transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)
    x_all = scaler.transform(x_raw)

    return DatasetBundle(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        x_all=x_all,
        y_all=y,
    )

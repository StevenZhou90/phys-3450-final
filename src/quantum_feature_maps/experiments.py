from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

from .data import DatasetBundle, make_one_qubit_grid, make_two_qubit_classification_dataset
from .feature_maps import OneQubitRotationMap, TwoQubitEntanglingMap
from .kernels import frobenius_relative_error, one_qubit_analytic_kernel, state_overlap_kernel
from .plotting import save_decision_boundary_plot, save_kernel_heatmap, save_line_comparison


@dataclass(frozen=True)
class ExperimentMetrics:
    one_qubit_relative_kernel_error: float
    raw_logistic_accuracy: float
    polynomial_logistic_accuracy: float
    quantum_kernel_svm_accuracy: float
    classical_rbf_svm_accuracy: float


def _decision_surface(
    model,
    grid_points: np.ndarray,
    kernel_builder=None,
) -> np.ndarray:
    if kernel_builder is None:
        predictions = model.predict(grid_points)
    else:
        kernel = kernel_builder(grid_points)
        predictions = model.predict(kernel)
    return predictions


def _make_plot_grid(x_train: np.ndarray, num_points: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max = x_train[:, 0].min() - 0.15, x_train[:, 0].max() + 0.15
    y_min, y_max = x_train[:, 1].min() - 0.15, x_train[:, 1].max() + 0.15
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, num_points),
        np.linspace(y_min, y_max, num_points),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, grid


def _fit_classical_models(dataset: DatasetBundle):
    raw_model = LogisticRegression(max_iter=2000)
    raw_model.fit(dataset.x_train, dataset.y_train)

    poly_model = Pipeline(
        [
            ("features", PolynomialFeatures(degree=3, include_bias=False)),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )
    poly_model.fit(dataset.x_train, dataset.y_train)

    rbf_model = SVC(kernel="rbf", gamma="scale")
    rbf_model.fit(dataset.x_train, dataset.y_train)
    return raw_model, poly_model, rbf_model


def _fit_quantum_kernel_model(dataset: DatasetBundle, feature_map: TwoQubitEntanglingMap) -> tuple[SVC, np.ndarray]:
    train_kernel = state_overlap_kernel(dataset.x_train, dataset.x_train, feature_map.statevector)
    model = SVC(kernel="precomputed")
    model.fit(train_kernel, dataset.y_train)
    test_kernel = state_overlap_kernel(dataset.x_test, dataset.x_train, feature_map.statevector)
    return model, test_kernel


def run_full_experiment(output_dir: str | Path = "results") -> ExperimentMetrics:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    one_qubit_map = OneQubitRotationMap()
    one_qubit_grid = make_one_qubit_grid()
    one_qubit_grid_2d = one_qubit_grid[:, None]

    analytic_kernel = one_qubit_analytic_kernel(one_qubit_grid, one_qubit_grid)
    simulated_kernel = state_overlap_kernel(one_qubit_grid_2d, one_qubit_grid_2d, lambda value: one_qubit_map.statevector(float(value[0])))
    relative_error = frobenius_relative_error(analytic_kernel, simulated_kernel)

    save_kernel_heatmap(
        simulated_kernel,
        output_path / "one_qubit_kernel_heatmap.png",
        title="Single-Qubit Kernel Matrix",
        x_label="Input index",
        y_label="Input index",
    )

    differences = np.linspace(-np.pi, np.pi, 300)
    analytic_line = np.cos(differences / 2.0) ** 2
    simulated_line = np.array(
        [
            state_overlap_kernel(
                np.array([[0.0]]),
                np.array([[difference]]),
                lambda value: one_qubit_map.statevector(float(value[0])),
            )[0, 0]
            for difference in differences
        ]
    )
    save_line_comparison(
        differences,
        analytic_line,
        simulated_line,
        output_path / "analytic_vs_simulated_kernel.png",
    )

    dataset = make_two_qubit_classification_dataset()
    two_qubit_map = TwoQubitEntanglingMap(entangling_scale=1.5)
    two_qubit_kernel = state_overlap_kernel(dataset.x_all, dataset.x_all, two_qubit_map.statevector)
    save_kernel_heatmap(
        two_qubit_kernel,
        output_path / "two_qubit_kernel_heatmap.png",
        title="Two-Qubit Kernel Matrix on Moon Dataset",
        x_label="Sample index",
        y_label="Sample index",
    )

    raw_model, poly_model, rbf_model = _fit_classical_models(dataset)
    quantum_model, test_kernel = _fit_quantum_kernel_model(dataset, two_qubit_map)

    metrics = ExperimentMetrics(
        one_qubit_relative_kernel_error=relative_error,
        raw_logistic_accuracy=accuracy_score(dataset.y_test, raw_model.predict(dataset.x_test)),
        polynomial_logistic_accuracy=accuracy_score(dataset.y_test, poly_model.predict(dataset.x_test)),
        quantum_kernel_svm_accuracy=accuracy_score(dataset.y_test, quantum_model.predict(test_kernel)),
        classical_rbf_svm_accuracy=accuracy_score(dataset.y_test, rbf_model.predict(dataset.x_test)),
    )

    xx, yy, grid = _make_plot_grid(dataset.x_train)
    raw_surface = _decision_surface(raw_model, grid).reshape(xx.shape)
    quantum_surface = _decision_surface(
        quantum_model,
        grid,
        kernel_builder=lambda points: state_overlap_kernel(points, dataset.x_train, two_qubit_map.statevector),
    ).reshape(xx.shape)
    poly_surface = _decision_surface(poly_model, grid).reshape(xx.shape)

    save_decision_boundary_plot(
        xx,
        yy,
        [raw_surface, poly_surface, quantum_surface],
        dataset.x_train,
        dataset.y_train,
        ["Raw Logistic Regression", "Polynomial Features", "Quantum Kernel SVM"],
        output_path / "classification_comparison.png",
    )

    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(metrics), handle, indent=2)

    return metrics

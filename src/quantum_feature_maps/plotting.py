from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def save_kernel_heatmap(
    matrix: np.ndarray,
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, cmap="mako", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_line_comparison(
    xs: np.ndarray,
    analytic: np.ndarray,
    simulated: np.ndarray,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, analytic, label="Analytic kernel", linewidth=2.5)
    ax.plot(xs, simulated, label="Simulated kernel", linewidth=2.0, linestyle="--")
    ax.set_xlabel("Input difference")
    ax.set_ylabel("Kernel value")
    ax.set_title("Single-Qubit Kernel: Analytic vs Simulated")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_decision_boundary_plot(
    xx: np.ndarray,
    yy: np.ndarray,
    surfaces: list[np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    titles: list[str],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(surfaces), figsize=(6 * len(surfaces), 5), sharex=True, sharey=True)
    if len(surfaces) == 1:
        axes = [axes]

    for ax, surface, title in zip(axes, surfaces, titles, strict=True):
        ax.contourf(xx, yy, surface, levels=30, cmap="RdYlBu", alpha=0.75)
        ax.scatter(
            x_train[:, 0],
            x_train[:, 1],
            c=y_train,
            cmap="RdYlBu",
            edgecolor="black",
            s=40,
        )
        ax.set_title(title)
        ax.set_xlabel("Scaled feature 1")
        ax.set_ylabel("Scaled feature 2")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

from __future__ import annotations

import os
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(workspace_root / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(workspace_root / ".cache"))
(workspace_root / ".mplconfig").mkdir(exist_ok=True)
(workspace_root / ".cache").mkdir(exist_ok=True)

from quantum_feature_maps import run_full_experiment


def main() -> None:
    metrics = run_full_experiment()
    print("Experiment complete.")
    for key, value in metrics.__dict__.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()

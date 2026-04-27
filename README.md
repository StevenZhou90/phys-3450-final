# Quantum Feature Maps as Simple Embeddings for Machine Learning

This repository contains a reproducible Qiskit-based implementation of the project
"Quantum Feature Maps as Simple Embeddings for Machine Learning."

The code is organized to support the main ingredients of the report rubric:

- explicit quantum feature maps on one and two qubits
- analytic and simulated kernel comparisons
- figures with labeled axes for a toy example and a classification task
- saved metrics that show the pipeline runs successfully

## Project layout

```text
src/quantum_feature_maps/
  data.py
  experiments.py
  feature_maps.py
  kernels.py
  plotting.py
scripts/
  run_experiment.py
```

## Reproducibility

Create an environment and install the package:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Run the full experiment suite:

```bash
.venv/bin/python scripts/run_experiment.py
```

This creates a `results/` directory containing:

- `metrics.json`: numerical summary of the main benchmarks
- `one_qubit_kernel_heatmap.png`: simulated single-qubit kernel matrix
- `analytic_vs_simulated_kernel.png`: analytic vs simulated single-qubit kernel
- `two_qubit_kernel_heatmap.png`: two-qubit kernel matrix on the classification task
- `classification_comparison.png`: decision-boundary comparison

## Main metric

The simplest success metric is the test accuracy comparison stored in
`results/metrics.json`. The experiment compares:

- logistic regression on raw inputs
- logistic regression on a classical polynomial feature lift
- an SVM using a quantum kernel computed from Qiskit statevectors
- an SVM with a classical RBF kernel baseline

This makes it easy to check whether the quantum embedding changes the similarity
structure in a useful way.

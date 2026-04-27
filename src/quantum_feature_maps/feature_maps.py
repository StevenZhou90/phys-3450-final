from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass(frozen=True)
class OneQubitRotationMap:
    """Simple one-qubit map |0> -> RY(x)|0>."""

    def circuit(self, x: float) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.ry(float(x), 0)
        return qc

    def statevector(self, x: float) -> Statevector:
        return Statevector.from_instruction(self.circuit(x))


@dataclass(frozen=True)
class TwoQubitEntanglingMap:
    """Two-qubit map with local rotations and one entangling phase."""

    entangling_scale: float = 1.0

    def circuit(self, x: np.ndarray) -> QuantumCircuit:
        x = np.asarray(x, dtype=float)
        if x.shape != (2,):
            raise ValueError("TwoQubitEntanglingMap expects a length-2 input vector.")

        qc = QuantumCircuit(2)
        qc.ry(x[0], 0)
        qc.ry(x[1], 1)
        qc.cx(0, 1)
        qc.rz(self.entangling_scale * (x[0] + x[1]), 1)
        qc.cx(0, 1)
        return qc

    def statevector(self, x: np.ndarray) -> Statevector:
        return Statevector.from_instruction(self.circuit(x))

"""
Photonic Quantum Computer Simulator

A comprehensive simulator for photonic quantum computing with support for:
- Quantum state representation and manipulation
- Single and multi-qubit quantum gates
- Entanglement operations
- Quantum circuits
- Quantum algorithms
"""

__version__ = "0.1.0"
__author__ = "ShPavelikus"

from .quantum_state import PhotonicState
from .quantum_gates import (
    QuantumGate,
    HadamardGate,
    PauliXGate,
    PauliYGate,
    PauliZGate,
    PhaseGate,
    SGate,
    TGate,
    RotationGate,
    CNOTGate,
    CZGate,
    SWAPGate,
)
from .measurement import Measurement
from .entanglement import create_bell_state, create_ghz_state, entanglement_entropy, concurrence
from .circuits import QuantumCircuit

__all__ = [
    "PhotonicState",
    "QuantumGate",
    "HadamardGate",
    "PauliXGate",
    "PauliYGate",
    "PauliZGate",
    "PhaseGate",
    "SGate",
    "TGate",
    "RotationGate",
    "CNOTGate",
    "CZGate",
    "SWAPGate",
    "Measurement",
    "create_bell_state",
    "create_ghz_state",
    "entanglement_entropy",
    "concurrence",
    "QuantumCircuit",
]

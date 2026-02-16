"""
Quantum Gates for Photonic Quantum Computing

This module implements single-qubit and multi-qubit quantum gates
with their matrix representations and application to quantum states.
"""

import numpy as np
from typing import Optional
from .quantum_state import PhotonicState


class QuantumGate:
    """
    Base class for quantum gates.
    
    Attributes:
        matrix (np.ndarray): Unitary matrix representation of the gate
        num_qubits (int): Number of qubits the gate acts on
    """
    
    def __init__(self, matrix: np.ndarray):
        """
        Initialize a quantum gate.
        
        Args:
            matrix: Unitary matrix representing the gate
        """
        self.matrix = np.array(matrix, dtype=complex)
        size = len(self.matrix)
        self.num_qubits = int(np.log2(size))
        
        if 2 ** self.num_qubits != size:
            raise ValueError(f"Gate matrix size must be a power of 2, got {size}")
    
    def is_unitary(self, tolerance: float = 1e-10) -> bool:
        """
        Check if the gate is unitary (U†U = I).
        
        Args:
            tolerance: Numerical tolerance
            
        Returns:
            True if gate is unitary
        """
        product = self.matrix @ self.matrix.conj().T
        identity = np.eye(len(self.matrix))
        return np.allclose(product, identity, atol=tolerance)
    
    def apply(self, state: PhotonicState, target_qubits: Optional[list] = None) -> PhotonicState:
        """
        Apply the gate to a quantum state.
        
        Args:
            state: Input quantum state
            target_qubits: List of target qubit indices (for single/multi-qubit gates on specific qubits)
            
        Returns:
            New quantum state after applying the gate
        """
        if target_qubits is None:
            # Apply to all qubits
            if self.num_qubits != state.num_qubits:
                raise ValueError(f"Gate acts on {self.num_qubits} qubits but state has {state.num_qubits}")
            new_vector = self.matrix @ state.state_vector
            return PhotonicState(new_vector, normalize=False)
        else:
            # Apply to specific qubits
            if len(target_qubits) != self.num_qubits:
                raise ValueError(f"Gate acts on {self.num_qubits} qubits but {len(target_qubits)} targets provided")
            
            # Build full matrix with gate on target qubits
            full_matrix = self._embed_gate(state.num_qubits, target_qubits)
            new_vector = full_matrix @ state.state_vector
            return PhotonicState(new_vector, normalize=False)
    
    def _embed_gate(self, total_qubits: int, target_qubits: list) -> np.ndarray:
        """
        Embed gate matrix into larger Hilbert space.
        
        Args:
            total_qubits: Total number of qubits in the system
            target_qubits: Indices of target qubits
            
        Returns:
            Full matrix acting on all qubits
        """
        if max(target_qubits) >= total_qubits:
            raise ValueError("Target qubit index out of range")
        
        # For single-qubit gates
        if self.num_qubits == 1:
            target = target_qubits[0]
            # Build tensor product: I ⊗ ... ⊗ I ⊗ U ⊗ I ⊗ ... ⊗ I
            result = np.eye(1, dtype=complex)
            for i in range(total_qubits):
                if i == target:
                    result = np.kron(result, self.matrix)
                else:
                    result = np.kron(result, np.eye(2))
            return result
        
        # For multi-qubit gates (simplified for adjacent qubits)
        elif self.num_qubits == 2:
            q1, q2 = sorted(target_qubits)
            if q2 - q1 != 1:
                # For non-adjacent qubits, use SWAP gates (simplified implementation)
                # This is a simplified version - full implementation would handle all cases
                pass
            
            # Build matrix for adjacent qubits
            result = np.eye(1, dtype=complex)
            for i in range(0, total_qubits, 2):
                if i == q1:
                    result = np.kron(result, self.matrix)
                elif i < total_qubits:
                    result = np.kron(result, np.eye(2))
            
            return result
        
        return self.matrix
    
    def __matmul__(self, other: 'QuantumGate') -> 'QuantumGate':
        """
        Compose two gates: (A @ B) means apply B then A.
        
        Args:
            other: Another quantum gate
            
        Returns:
            Composed gate
        """
        if self.num_qubits != other.num_qubits:
            raise ValueError("Gates must act on same number of qubits")
        
        return QuantumGate(self.matrix @ other.matrix)
    
    def dagger(self) -> 'QuantumGate':
        """
        Return the Hermitian conjugate (adjoint) of the gate.
        
        Returns:
            Adjoint gate U†
        """
        return QuantumGate(self.matrix.conj().T)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.num_qubits}-qubit)"


# Single-qubit gates

class HadamardGate(QuantumGate):
    """
    Hadamard gate: Creates superposition.
    Implemented with 50/50 beam splitter in photonics.
    
    Matrix: (1/√2) [[1,  1],
                     [1, -1]]
    """
    
    def __init__(self):
        matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        super().__init__(matrix)


class PauliXGate(QuantumGate):
    """
    Pauli-X gate (NOT gate): Bit flip.
    
    Matrix: [[0, 1],
             [1, 0]]
    """
    
    def __init__(self):
        matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        super().__init__(matrix)


class PauliYGate(QuantumGate):
    """
    Pauli-Y gate: Bit and phase flip.
    
    Matrix: [[0, -i],
             [i,  0]]
    """
    
    def __init__(self):
        matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
        super().__init__(matrix)


class PauliZGate(QuantumGate):
    """
    Pauli-Z gate: Phase flip.
    
    Matrix: [[1,  0],
             [0, -1]]
    """
    
    def __init__(self):
        matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        super().__init__(matrix)


class PhaseGate(QuantumGate):
    """
    Phase gate: Applies phase rotation.
    
    Matrix: [[1, 0      ],
             [0, e^(iφ)]]
    """
    
    def __init__(self, phase: float):
        """
        Initialize phase gate.
        
        Args:
            phase: Phase angle φ in radians
        """
        self.phase = phase
        matrix = np.array([[1, 0], [0, np.exp(1j * phase)]], dtype=complex)
        super().__init__(matrix)


class SGate(QuantumGate):
    """
    S gate (Phase gate with φ = π/2).
    
    Matrix: [[1, 0],
             [0, i]]
    """
    
    def __init__(self):
        matrix = np.array([[1, 0], [0, 1j]], dtype=complex)
        super().__init__(matrix)


class TGate(QuantumGate):
    """
    T gate (Phase gate with φ = π/4).
    
    Matrix: [[1, 0           ],
             [0, e^(iπ/4)]]
    """
    
    def __init__(self):
        matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        super().__init__(matrix)


class RotationGate(QuantumGate):
    """
    General rotation gate around axis on Bloch sphere.
    
    R(θ, φ, λ) is the general single-qubit rotation.
    """
    
    def __init__(self, theta: float, phi: float, lam: float):
        """
        Initialize rotation gate.
        
        Args:
            theta: Rotation angle θ
            phi: Phase angle φ
            lam: Phase angle λ
        """
        self.theta = theta
        self.phi = phi
        self.lam = lam
        
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        
        matrix = np.array([
            [cos, -np.exp(1j * lam) * sin],
            [np.exp(1j * phi) * sin, np.exp(1j * (phi + lam)) * cos]
        ], dtype=complex)
        super().__init__(matrix)


# Two-qubit gates

class CNOTGate(QuantumGate):
    """
    CNOT (Controlled-NOT) gate.
    Implemented through Hong-Ou-Mandel interference in photonics.
    
    Matrix: [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]]
    """
    
    def __init__(self):
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        super().__init__(matrix)


class CZGate(QuantumGate):
    """
    Controlled-Z gate.
    
    Matrix: [[1, 0, 0,  0],
             [0, 1, 0,  0],
             [0, 0, 1,  0],
             [0, 0, 0, -1]]
    """
    
    def __init__(self):
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
        super().__init__(matrix)


class SWAPGate(QuantumGate):
    """
    SWAP gate: Exchanges two qubits.
    
    Matrix: [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
    """
    
    def __init__(self):
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        super().__init__(matrix)


def tensor_product(*gates: QuantumGate) -> QuantumGate:
    """
    Compute tensor product of multiple gates.
    
    Args:
        gates: Variable number of quantum gates
        
    Returns:
        Tensor product gate
    """
    if not gates:
        raise ValueError("At least one gate required")
    
    result = gates[0].matrix
    for gate in gates[1:]:
        result = np.kron(result, gate.matrix)
    
    return QuantumGate(result)

"""
Quantum State Representation for Photonic Quantum Computing

This module provides the PhotonicState class for representing and manipulating
quantum states of photons in a quantum computer.
"""

import numpy as np
from typing import List, Tuple, Optional


class PhotonicState:
    """
    Represents a quantum state of photonic qubits.
    
    The state is represented as a complex-valued vector in the computational basis.
    For n qubits, the state vector has dimension 2^n.
    
    Attributes:
        state_vector (np.ndarray): Complex vector representing the quantum state
        num_qubits (int): Number of qubits in the system
    """
    
    def __init__(self, state_vector: np.ndarray, normalize: bool = True):
        """
        Initialize a photonic quantum state.
        
        Args:
            state_vector: Complex vector representing the quantum state
            normalize: Whether to normalize the state vector (default: True)
        """
        self.state_vector = np.array(state_vector, dtype=complex)
        
        # Determine number of qubits
        size = len(self.state_vector)
        self.num_qubits = int(np.log2(size))
        
        if 2 ** self.num_qubits != size:
            raise ValueError(f"State vector size must be a power of 2, got {size}")
        
        if normalize:
            self.normalize()
    
    def normalize(self):
        """Normalize the state vector to unit length."""
        norm = np.linalg.norm(self.state_vector)
        if norm > 1e-10:
            self.state_vector /= norm
    
    def copy(self) -> 'PhotonicState':
        """Create a copy of the state."""
        return PhotonicState(self.state_vector.copy(), normalize=False)
    
    @classmethod
    def zero_state(cls, num_qubits: int) -> 'PhotonicState':
        """
        Create a |0...0⟩ state with n qubits.
        
        Args:
            num_qubits: Number of qubits
            
        Returns:
            PhotonicState in |0...0⟩ state
        """
        size = 2 ** num_qubits
        state_vector = np.zeros(size, dtype=complex)
        state_vector[0] = 1.0
        return cls(state_vector, normalize=False)
    
    @classmethod
    def one_state(cls, num_qubits: int) -> 'PhotonicState':
        """
        Create a |1...1⟩ state with n qubits.
        
        Args:
            num_qubits: Number of qubits
            
        Returns:
            PhotonicState in |1...1⟩ state
        """
        size = 2 ** num_qubits
        state_vector = np.zeros(size, dtype=complex)
        state_vector[-1] = 1.0
        return cls(state_vector, normalize=False)
    
    @classmethod
    def basis_state(cls, num_qubits: int, basis_index: int) -> 'PhotonicState':
        """
        Create a computational basis state |i⟩.
        
        Args:
            num_qubits: Number of qubits
            basis_index: Index of the basis state (0 to 2^n - 1)
            
        Returns:
            PhotonicState in basis state |i⟩
        """
        size = 2 ** num_qubits
        if basis_index < 0 or basis_index >= size:
            raise ValueError(f"Basis index must be between 0 and {size-1}")
        
        state_vector = np.zeros(size, dtype=complex)
        state_vector[basis_index] = 1.0
        return cls(state_vector, normalize=False)
    
    @classmethod
    def superposition(cls, num_qubits: int) -> 'PhotonicState':
        """
        Create an equal superposition state (|0⟩ + |1⟩)/√2 ⊗ ... ⊗ (|0⟩ + |1⟩)/√2.
        
        Args:
            num_qubits: Number of qubits
            
        Returns:
            PhotonicState in equal superposition
        """
        size = 2 ** num_qubits
        state_vector = np.ones(size, dtype=complex) / np.sqrt(size)
        return cls(state_vector, normalize=False)
    
    def density_matrix(self) -> np.ndarray:
        """
        Compute the density matrix ρ = |ψ⟩⟨ψ| for this pure state.
        
        Returns:
            Density matrix as a 2D numpy array
        """
        return np.outer(self.state_vector, self.state_vector.conj())
    
    def is_normalized(self, tolerance: float = 1e-10) -> bool:
        """
        Check if the state is normalized.
        
        Args:
            tolerance: Numerical tolerance for the norm check
            
        Returns:
            True if ||ψ|| ≈ 1
        """
        norm = np.linalg.norm(self.state_vector)
        return abs(norm - 1.0) < tolerance
    
    def probability(self, basis_index: int) -> float:
        """
        Compute the probability of measuring a specific basis state.
        
        Args:
            basis_index: Index of the basis state
            
        Returns:
            Probability P(|i⟩) = |⟨i|ψ⟩|²
        """
        if basis_index < 0 or basis_index >= len(self.state_vector):
            raise ValueError(f"Basis index must be between 0 and {len(self.state_vector)-1}")
        
        return abs(self.state_vector[basis_index]) ** 2
    
    def probabilities(self) -> np.ndarray:
        """
        Compute probabilities for all basis states.
        
        Returns:
            Array of probabilities for each basis state
        """
        return np.abs(self.state_vector) ** 2
    
    def inner_product(self, other: 'PhotonicState') -> complex:
        """
        Compute inner product ⟨φ|ψ⟩ with another state.
        
        Args:
            other: Another PhotonicState
            
        Returns:
            Complex inner product
        """
        if self.num_qubits != other.num_qubits:
            raise ValueError("States must have the same number of qubits")
        
        return np.vdot(other.state_vector, self.state_vector)
    
    def fidelity(self, other: 'PhotonicState') -> float:
        """
        Compute fidelity |⟨φ|ψ⟩|² with another state.
        
        Args:
            other: Another PhotonicState
            
        Returns:
            Fidelity value between 0 and 1
        """
        return abs(self.inner_product(other)) ** 2
    
    def expectation_value(self, operator: np.ndarray) -> complex:
        """
        Compute expectation value ⟨ψ|O|ψ⟩ of an operator.
        
        Args:
            operator: Operator matrix
            
        Returns:
            Expectation value
        """
        return np.vdot(self.state_vector, operator @ self.state_vector)
    
    def partial_trace(self, traced_qubits: List[int]) -> np.ndarray:
        """
        Compute partial trace over specified qubits.
        
        Args:
            traced_qubits: List of qubit indices to trace out
            
        Returns:
            Reduced density matrix
        """
        # Get full density matrix
        rho = self.density_matrix()
        
        # Determine kept qubits
        kept_qubits = [i for i in range(self.num_qubits) if i not in traced_qubits]
        
        if not kept_qubits:
            # All qubits traced out, return scalar
            return np.array([[np.trace(rho)]])
        
        # Dimensions
        dim_kept = 2 ** len(kept_qubits)
        dim_traced = 2 ** len(traced_qubits)
        
        # Initialize reduced density matrix
        rho_reduced = np.zeros((dim_kept, dim_kept), dtype=complex)
        
        # Sum over traced qubits
        for i in range(dim_kept):
            for j in range(dim_kept):
                # Map reduced indices to full indices
                for k in range(dim_traced):
                    # Build full indices
                    idx_i = 0
                    idx_j = 0
                    
                    # Insert kept qubit values
                    for pos, qubit in enumerate(kept_qubits):
                        bit_i = (i >> pos) & 1
                        bit_j = (j >> pos) & 1
                        idx_i |= (bit_i << qubit)
                        idx_j |= (bit_j << qubit)
                    
                    # Insert traced qubit values (same for both)
                    for pos, qubit in enumerate(traced_qubits):
                        bit = (k >> pos) & 1
                        idx_i |= (bit << qubit)
                        idx_j |= (bit << qubit)
                    
                    rho_reduced[i, j] += rho[idx_i, idx_j]
        
        return rho_reduced
    
    def __str__(self) -> str:
        """String representation of the state."""
        result = []
        for i, amplitude in enumerate(self.state_vector):
            if abs(amplitude) > 1e-10:
                basis = format(i, f'0{self.num_qubits}b')
                if amplitude.imag == 0:
                    result.append(f"{amplitude.real:+.4f}|{basis}⟩")
                else:
                    result.append(f"({amplitude.real:+.4f}{amplitude.imag:+.4f}j)|{basis}⟩")
        return " + ".join(result) if result else "0"
    
    def __repr__(self) -> str:
        """Representation of the state."""
        return f"PhotonicState(num_qubits={self.num_qubits}, state={str(self)})"

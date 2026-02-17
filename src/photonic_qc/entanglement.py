"""
Entanglement Operations for Photonic Quantum Computing

This module provides functions for creating and analyzing entangled states.
"""

import numpy as np
from typing import List
from .quantum_state import PhotonicState
from .quantum_gates import HadamardGate, CNOTGate


def create_bell_state(bell_type: str = "phi_plus") -> PhotonicState:
    """
    Create a Bell state (maximally entangled two-qubit state).
    
    Bell states:
    - |Φ+⟩ = (|00⟩ + |11⟩)/√2
    - |Φ-⟩ = (|00⟩ - |11⟩)/√2
    - |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    - |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    
    Args:
        bell_type: Type of Bell state ("phi_plus", "phi_minus", "psi_plus", "psi_minus")
        
    Returns:
        PhotonicState representing the Bell state
    """
    bell_states = {
        "phi_plus": np.array([1, 0, 0, 1]) / np.sqrt(2),
        "phi_minus": np.array([1, 0, 0, -1]) / np.sqrt(2),
        "psi_plus": np.array([0, 1, 1, 0]) / np.sqrt(2),
        "psi_minus": np.array([0, 1, -1, 0]) / np.sqrt(2),
    }
    
    if bell_type not in bell_states:
        raise ValueError(f"Unknown Bell state type: {bell_type}")
    
    return PhotonicState(bell_states[bell_type], normalize=False)


def create_ghz_state(num_qubits: int) -> PhotonicState:
    """
    Create a GHZ (Greenberger-Horne-Zeilinger) state.
    
    |GHZ⟩ = (|0...0⟩ + |1...1⟩)/√2
    
    Args:
        num_qubits: Number of qubits
        
    Returns:
        PhotonicState representing the GHZ state
    """
    if num_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits")
    
    size = 2 ** num_qubits
    state_vector = np.zeros(size, dtype=complex)
    state_vector[0] = 1.0 / np.sqrt(2)
    state_vector[-1] = 1.0 / np.sqrt(2)
    
    return PhotonicState(state_vector, normalize=False)


def entanglement_entropy(state: PhotonicState, subsystem_qubits: List[int]) -> float:
    """
    Compute the entanglement entropy (von Neumann entropy) of a subsystem.
    
    S(ρ_A) = -Tr(ρ_A log₂(ρ_A))
    
    Args:
        state: Quantum state
        subsystem_qubits: List of qubit indices for subsystem A
        
    Returns:
        Entanglement entropy in bits
    """
    if not subsystem_qubits:
        return 0.0
    
    # Get all qubits not in subsystem
    all_qubits = set(range(state.num_qubits))
    traced_qubits = list(all_qubits - set(subsystem_qubits))
    
    if not traced_qubits:
        # No qubits to trace out, state is pure
        return 0.0
    
    # Compute reduced density matrix
    rho_reduced = state.partial_trace(traced_qubits)
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(rho_reduced)
    
    # Compute von Neumann entropy
    entropy = 0.0
    for eigenval in eigenvalues:
        if eigenval > 1e-10:
            entropy -= eigenval * np.log2(eigenval)
    
    return entropy


def schmidt_decomposition(state: PhotonicState, partition: int) -> tuple:
    """
    Compute Schmidt decomposition for bipartite system.
    
    Args:
        state: Quantum state of n qubits
        partition: Number of qubits in subsystem A (rest go to subsystem B)
        
    Returns:
        Tuple of (schmidt_coefficients, basis_A, basis_B)
    """
    if partition <= 0 or partition >= state.num_qubits:
        raise ValueError("Invalid partition")
    
    # Reshape state vector into matrix
    dim_a = 2 ** partition
    dim_b = 2 ** (state.num_qubits - partition)
    
    psi_matrix = state.state_vector.reshape(dim_a, dim_b)
    
    # Perform SVD
    U, s, Vh = np.linalg.svd(psi_matrix, full_matrices=False)
    
    return s, U, Vh.conj().T


def is_entangled(state: PhotonicState, tolerance: float = 1e-10) -> bool:
    """
    Check if a two-qubit state is entangled using Schmidt decomposition.
    
    Args:
        state: Two-qubit quantum state
        tolerance: Numerical tolerance
        
    Returns:
        True if state is entangled
    """
    if state.num_qubits != 2:
        raise ValueError("Entanglement check implemented for two-qubit states only")
    
    schmidt_coeffs, _, _ = schmidt_decomposition(state, 1)
    
    # State is entangled if more than one Schmidt coefficient is non-zero
    non_zero_coeffs = np.sum(schmidt_coeffs > tolerance)
    
    return non_zero_coeffs > 1


def concurrence(state: PhotonicState) -> float:
    """
    Compute concurrence for a two-qubit state (measure of entanglement).
    
    Args:
        state: Two-qubit quantum state
        
    Returns:
        Concurrence value (0 for separable, 1 for maximally entangled)
    """
    if state.num_qubits != 2:
        raise ValueError("Concurrence defined for two-qubit states only")
    
    # Get density matrix
    rho = state.density_matrix()
    
    # Spin-flip operator
    sigma_y = np.array([[0, -1j], [1j, 0]])
    spin_flip = np.kron(sigma_y, sigma_y)
    
    # Compute R = ρ * (σ_y ⊗ σ_y) * ρ* * (σ_y ⊗ σ_y)
    rho_tilde = spin_flip @ rho.conj() @ spin_flip
    R = rho @ rho_tilde
    
    # Get eigenvalues and sort in descending order
    eigenvalues = np.linalg.eigvalsh(R)
    eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Concurrence
    C = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
    
    return C

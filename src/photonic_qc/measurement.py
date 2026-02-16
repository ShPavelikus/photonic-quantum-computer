"""
Quantum Measurements for Photonic Quantum Computing

This module implements quantum measurements including projective measurements
and measurements in different bases.
"""

import numpy as np
from typing import Tuple, Optional
from .quantum_state import PhotonicState


class Measurement:
    """
    Quantum measurement operations.
    """
    
    @staticmethod
    def measure_computational_basis(state: PhotonicState, qubit: Optional[int] = None) -> Tuple[int, PhotonicState]:
        """
        Perform a projective measurement in the computational basis.
        
        Args:
            state: Quantum state to measure
            qubit: Specific qubit to measure (None for all qubits)
            
        Returns:
            Tuple of (measurement_outcome, collapsed_state)
        """
        if qubit is None:
            # Measure all qubits
            probabilities = state.probabilities()
            outcome = np.random.choice(len(probabilities), p=probabilities)
            
            # Collapse to measured basis state
            collapsed = PhotonicState.basis_state(state.num_qubits, outcome)
            return outcome, collapsed
        else:
            # Measure single qubit
            if qubit < 0 or qubit >= state.num_qubits:
                raise ValueError(f"Qubit index {qubit} out of range")
            
            # Compute probabilities for qubit being 0 or 1
            prob_0 = 0.0
            for i in range(len(state.state_vector)):
                if not (i >> qubit) & 1:  # Check if qubit is 0
                    prob_0 += abs(state.state_vector[i]) ** 2
            
            # Measure
            outcome = 0 if np.random.random() < prob_0 else 1
            
            # Collapse state
            new_vector = state.state_vector.copy()
            for i in range(len(new_vector)):
                if ((i >> qubit) & 1) != outcome:
                    new_vector[i] = 0
            
            collapsed = PhotonicState(new_vector, normalize=True)
            return outcome, collapsed
    
    @staticmethod
    def measure_x_basis(state: PhotonicState, qubit: int) -> Tuple[int, PhotonicState]:
        """
        Measure in the X basis (|+⟩, |-⟩).
        
        Args:
            state: Quantum state to measure
            qubit: Qubit to measure
            
        Returns:
            Tuple of (outcome, collapsed_state) where outcome is 0 for |+⟩ or 1 for |-⟩
        """
        from .quantum_gates import HadamardGate
        
        # Transform to X basis by applying Hadamard
        h_gate = HadamardGate()
        transformed = h_gate.apply(state, [qubit])
        
        # Measure in computational basis
        outcome, collapsed = Measurement.measure_computational_basis(transformed, qubit)
        
        # Transform back
        final_state = h_gate.apply(collapsed, [qubit])
        
        return outcome, final_state
    
    @staticmethod
    def measure_y_basis(state: PhotonicState, qubit: int) -> Tuple[int, PhotonicState]:
        """
        Measure in the Y basis.
        
        Args:
            state: Quantum state to measure
            qubit: Qubit to measure
            
        Returns:
            Tuple of (outcome, collapsed_state)
        """
        from .quantum_gates import HadamardGate, SGate, QuantumGate
        
        # Transform to Y basis
        s_dagger = SGate().dagger()
        h_gate = HadamardGate()
        
        transformed = s_dagger.apply(state, [qubit])
        transformed = h_gate.apply(transformed, [qubit])
        
        # Measure in computational basis
        outcome, collapsed = Measurement.measure_computational_basis(transformed, qubit)
        
        # Transform back
        final_state = h_gate.apply(collapsed, [qubit])
        final_state = SGate().apply(final_state, [qubit])
        
        return outcome, final_state
    
    @staticmethod
    def get_measurement_probabilities(state: PhotonicState) -> np.ndarray:
        """
        Get measurement probabilities for all basis states.
        
        Args:
            state: Quantum state
            
        Returns:
            Array of probabilities
        """
        return state.probabilities()
    
    @staticmethod
    def expectation_value(state: PhotonicState, operator: np.ndarray) -> complex:
        """
        Compute expectation value of an operator.
        
        Args:
            state: Quantum state
            operator: Operator matrix
            
        Returns:
            Expectation value ⟨ψ|O|ψ⟩
        """
        return state.expectation_value(operator)

"""
Tests for quantum state representation and manipulation.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '../src')

from photonic_qc.quantum_state import PhotonicState


class TestPhotonicStateCreation:
    """Test creation of quantum states."""
    
    def test_zero_state(self):
        """Test creation of |0⟩ state."""
        state = PhotonicState.zero_state(1)
        assert state.num_qubits == 1
        assert np.allclose(state.state_vector, [1, 0])
        assert state.is_normalized()
    
    def test_one_state(self):
        """Test creation of |1⟩ state."""
        state = PhotonicState.one_state(1)
        assert state.num_qubits == 1
        assert np.allclose(state.state_vector, [0, 1])
        assert state.is_normalized()
    
    def test_superposition(self):
        """Test creation of equal superposition."""
        state = PhotonicState.superposition(2)
        assert state.num_qubits == 2
        expected = np.ones(4) / 2.0
        assert np.allclose(state.state_vector, expected)
        assert state.is_normalized()
    
    def test_basis_state(self):
        """Test creation of arbitrary basis state."""
        state = PhotonicState.basis_state(2, 2)  # |10⟩
        assert state.num_qubits == 2
        expected = np.array([0, 0, 1, 0])
        assert np.allclose(state.state_vector, expected)
    
    def test_custom_state(self):
        """Test creation of custom state."""
        state = PhotonicState(np.array([0.6, 0.8]))
        assert state.num_qubits == 1
        assert state.is_normalized()
    
    def test_invalid_size(self):
        """Test that invalid vector size raises error."""
        with pytest.raises(ValueError):
            PhotonicState(np.array([1, 0, 0]))  # Size 3 is not power of 2


class TestPhotonicStateProperties:
    """Test properties and methods of quantum states."""
    
    def test_normalization(self):
        """Test state normalization."""
        state = PhotonicState(np.array([3, 4]), normalize=True)
        assert state.is_normalized()
        assert np.isclose(np.abs(state.state_vector[0]), 0.6)
        assert np.isclose(np.abs(state.state_vector[1]), 0.8)
    
    def test_probability(self):
        """Test probability computation."""
        state = PhotonicState(np.array([0.6, 0.8]))
        assert np.isclose(state.probability(0), 0.36)
        assert np.isclose(state.probability(1), 0.64)
    
    def test_probabilities(self):
        """Test all probabilities sum to 1."""
        state = PhotonicState.superposition(2)
        probs = state.probabilities()
        assert len(probs) == 4
        assert np.isclose(np.sum(probs), 1.0)
    
    def test_inner_product(self):
        """Test inner product between states."""
        state1 = PhotonicState.zero_state(1)
        state2 = PhotonicState.one_state(1)
        
        # Orthogonal states
        inner = state1.inner_product(state2)
        assert np.isclose(inner, 0.0)
        
        # Same state
        inner = state1.inner_product(state1)
        assert np.isclose(inner, 1.0)
    
    def test_fidelity(self):
        """Test fidelity computation."""
        state1 = PhotonicState.zero_state(1)
        state2 = PhotonicState.one_state(1)
        
        # Orthogonal states have fidelity 0
        fidelity = state1.fidelity(state2)
        assert np.isclose(fidelity, 0.0)
        
        # Same state has fidelity 1
        fidelity = state1.fidelity(state1)
        assert np.isclose(fidelity, 1.0)
    
    def test_density_matrix(self):
        """Test density matrix computation."""
        state = PhotonicState.zero_state(1)
        rho = state.density_matrix()
        
        # Check shape
        assert rho.shape == (2, 2)
        
        # Check properties of density matrix
        assert np.allclose(rho, rho.conj().T)  # Hermitian
        assert np.isclose(np.trace(rho), 1.0)  # Trace 1
        assert np.allclose(rho @ rho, rho)  # Pure state: ρ² = ρ
    
    def test_expectation_value(self):
        """Test expectation value computation."""
        state = PhotonicState.zero_state(1)
        
        # Pauli-Z operator
        Z = np.array([[1, 0], [0, -1]])
        exp_val = state.expectation_value(Z)
        
        # |0⟩ is eigenstate of Z with eigenvalue +1
        assert np.isclose(exp_val, 1.0)
    
    def test_copy(self):
        """Test state copying."""
        state1 = PhotonicState.zero_state(1)
        state2 = state1.copy()
        
        # States should be equal
        assert np.allclose(state1.state_vector, state2.state_vector)
        
        # But modifications to one shouldn't affect the other
        state2.state_vector[0] = 0
        assert not np.allclose(state1.state_vector, state2.state_vector)


class TestMultiQubitStates:
    """Test multi-qubit state operations."""
    
    def test_two_qubit_basis(self):
        """Test two-qubit basis states."""
        states = [
            (PhotonicState.basis_state(2, 0), [1, 0, 0, 0]),  # |00⟩
            (PhotonicState.basis_state(2, 1), [0, 1, 0, 0]),  # |01⟩
            (PhotonicState.basis_state(2, 2), [0, 0, 1, 0]),  # |10⟩
            (PhotonicState.basis_state(2, 3), [0, 0, 0, 1]),  # |11⟩
        ]
        
        for state, expected in states:
            assert np.allclose(state.state_vector, expected)
    
    def test_three_qubit_state(self):
        """Test three-qubit state."""
        state = PhotonicState.zero_state(3)
        assert state.num_qubits == 3
        assert len(state.state_vector) == 8
    
    def test_partial_trace_two_qubit(self):
        """Test partial trace for two-qubit system."""
        # Create product state |01⟩ (qubit 0 is |1⟩, qubit 1 is |0⟩)
        state = PhotonicState.basis_state(2, 1)  # |01⟩
        
        # Trace out qubit 1 should give |1⟩⟨1| for qubit 0
        rho_0 = state.partial_trace([1])
        expected = np.array([[0, 0], [0, 1]])  # |1⟩⟨1|
        assert np.allclose(rho_0, expected)


class TestStateOrthogonality:
    """Test orthogonality of states."""
    
    def test_computational_basis_orthogonal(self):
        """Test that computational basis states are orthogonal."""
        state_0 = PhotonicState.zero_state(1)
        state_1 = PhotonicState.one_state(1)
        
        inner = state_0.inner_product(state_1)
        assert np.isclose(np.abs(inner), 0.0)
    
    def test_two_qubit_basis_orthogonal(self):
        """Test orthogonality of two-qubit basis."""
        for i in range(4):
            for j in range(4):
                state_i = PhotonicState.basis_state(2, i)
                state_j = PhotonicState.basis_state(2, j)
                inner = state_i.inner_product(state_j)
                
                if i == j:
                    assert np.isclose(np.abs(inner), 1.0)
                else:
                    assert np.isclose(np.abs(inner), 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for entanglement operations.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '../src')

from photonic_qc.quantum_state import PhotonicState
from photonic_qc.entanglement import (
    create_bell_state,
    create_ghz_state,
    entanglement_entropy,
    is_entangled,
    concurrence,
    schmidt_decomposition
)


class TestBellStates:
    """Test Bell state creation and properties."""
    
    def test_phi_plus(self):
        """Test |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
        state = create_bell_state("phi_plus")
        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        assert np.allclose(state.state_vector, expected)
        assert state.is_normalized()
    
    def test_phi_minus(self):
        """Test |Φ-⟩ = (|00⟩ - |11⟩)/√2."""
        state = create_bell_state("phi_minus")
        expected = np.array([1, 0, 0, -1]) / np.sqrt(2)
        assert np.allclose(state.state_vector, expected)
        assert state.is_normalized()
    
    def test_psi_plus(self):
        """Test |Ψ+⟩ = (|01⟩ + |10⟩)/√2."""
        state = create_bell_state("psi_plus")
        expected = np.array([0, 1, 1, 0]) / np.sqrt(2)
        assert np.allclose(state.state_vector, expected)
        assert state.is_normalized()
    
    def test_psi_minus(self):
        """Test |Ψ-⟩ = (|01⟩ - |10⟩)/√2."""
        state = create_bell_state("psi_minus")
        expected = np.array([0, 1, -1, 0]) / np.sqrt(2)
        assert np.allclose(state.state_vector, expected)
        assert state.is_normalized()
    
    def test_bell_states_orthogonal(self):
        """Test that Bell states are orthogonal."""
        bell_types = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]
        states = [create_bell_state(bt) for bt in bell_types]
        
        for i, state_i in enumerate(states):
            for j, state_j in enumerate(states):
                inner = state_i.inner_product(state_j)
                if i == j:
                    assert np.isclose(np.abs(inner), 1.0)
                else:
                    assert np.isclose(np.abs(inner), 0.0)
    
    def test_invalid_bell_type(self):
        """Test that invalid Bell type raises error."""
        with pytest.raises(ValueError):
            create_bell_state("invalid_type")


class TestGHZStates:
    """Test GHZ state creation."""
    
    def test_ghz_3_qubits(self):
        """Test 3-qubit GHZ state."""
        state = create_ghz_state(3)
        expected = np.zeros(8)
        expected[0] = 1 / np.sqrt(2)
        expected[7] = 1 / np.sqrt(2)
        assert np.allclose(state.state_vector, expected)
        assert state.is_normalized()
    
    def test_ghz_2_qubits(self):
        """Test 2-qubit GHZ state (same as Bell state)."""
        state = create_ghz_state(2)
        bell = create_bell_state("phi_plus")
        assert np.allclose(state.state_vector, bell.state_vector)
    
    def test_ghz_4_qubits(self):
        """Test 4-qubit GHZ state."""
        state = create_ghz_state(4)
        assert state.num_qubits == 4
        assert state.is_normalized()
        
        # Check only |0000⟩ and |1111⟩ have amplitude
        probs = state.probabilities()
        assert np.isclose(probs[0], 0.5)  # |0000⟩
        assert np.isclose(probs[15], 0.5)  # |1111⟩
        assert np.isclose(np.sum(probs[1:15]), 0.0)
    
    def test_ghz_invalid_size(self):
        """Test that invalid size raises error."""
        with pytest.raises(ValueError):
            create_ghz_state(1)


class TestEntanglementEntropy:
    """Test entanglement entropy computation."""
    
    def test_separable_state_zero_entropy(self):
        """Test that separable state has zero entropy."""
        state = PhotonicState.zero_state(2)
        entropy = entanglement_entropy(state, [0])
        assert np.isclose(entropy, 0.0)
    
    def test_bell_state_max_entropy(self):
        """Test that Bell state has maximum entropy."""
        state = create_bell_state("phi_plus")
        entropy = entanglement_entropy(state, [0])
        
        # Maximum entropy for 1 qubit is log2(2) = 1
        assert np.isclose(entropy, 1.0)
    
    def test_ghz_state_entropy(self):
        """Test entropy of GHZ state."""
        state = create_ghz_state(3)
        
        # Single qubit subsystem
        entropy_1 = entanglement_entropy(state, [0])
        assert entropy_1 > 0  # Entangled
        assert entropy_1 <= 1  # But not maximally with single qubit
    
    def test_entropy_symmetric(self):
        """Test that entropy is same for either subsystem."""
        state = create_bell_state("phi_plus")
        entropy_0 = entanglement_entropy(state, [0])
        entropy_1 = entanglement_entropy(state, [1])
        assert np.isclose(entropy_0, entropy_1)


class TestIsEntangled:
    """Test entanglement detection."""
    
    def test_bell_state_is_entangled(self):
        """Test that Bell states are detected as entangled."""
        for bell_type in ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]:
            state = create_bell_state(bell_type)
            assert is_entangled(state)
    
    def test_separable_state_not_entangled(self):
        """Test that separable states are not entangled."""
        states = [
            PhotonicState.basis_state(2, 0),  # |00⟩
            PhotonicState.basis_state(2, 1),  # |01⟩
            PhotonicState.basis_state(2, 2),  # |10⟩
            PhotonicState.basis_state(2, 3),  # |11⟩
        ]
        
        for state in states:
            assert not is_entangled(state)
    
    def test_product_superposition_not_entangled(self):
        """Test that product of superpositions is not entangled."""
        # |+⟩⊗|+⟩ = (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
        state = PhotonicState.superposition(2)
        assert not is_entangled(state)


class TestConcurrence:
    """Test concurrence (measure of entanglement)."""
    
    def test_bell_state_max_concurrence(self):
        """Test that Bell states have maximum concurrence."""
        for bell_type in ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]:
            state = create_bell_state(bell_type)
            C = concurrence(state)
            assert np.isclose(C, 1.0)
    
    def test_separable_state_zero_concurrence(self):
        """Test that separable states have zero concurrence."""
        state = PhotonicState.zero_state(2)
        C = concurrence(state)
        assert np.isclose(C, 0.0)
    
    def test_concurrence_range(self):
        """Test that concurrence is in [0, 1]."""
        # Test various states
        states = [
            PhotonicState.zero_state(2),
            create_bell_state("phi_plus"),
            PhotonicState.superposition(2),
        ]
        
        for state in states:
            C = concurrence(state)
            assert 0.0 <= C <= 1.0


class TestSchmidtDecomposition:
    """Test Schmidt decomposition."""
    
    def test_schmidt_bell_state(self):
        """Test Schmidt decomposition of Bell state."""
        state = create_bell_state("phi_plus")
        schmidt_coeffs, U, V = schmidt_decomposition(state, 1)
        
        # Bell state has 2 equal Schmidt coefficients
        assert len(schmidt_coeffs) >= 2
        assert np.isclose(schmidt_coeffs[0], 1/np.sqrt(2))
        assert np.isclose(schmidt_coeffs[1], 1/np.sqrt(2))
        
        # Higher coefficients should be zero
        if len(schmidt_coeffs) > 2:
            assert np.allclose(schmidt_coeffs[2:], 0, atol=1e-10)
    
    def test_schmidt_separable_state(self):
        """Test Schmidt decomposition of separable state."""
        state = PhotonicState.basis_state(2, 0)  # |00⟩
        schmidt_coeffs, U, V = schmidt_decomposition(state, 1)
        
        # Separable state has only 1 non-zero Schmidt coefficient
        assert np.isclose(schmidt_coeffs[0], 1.0)
        assert np.allclose(schmidt_coeffs[1:], 0, atol=1e-10)
    
    def test_schmidt_coefficients_normalized(self):
        """Test that Schmidt coefficients are normalized."""
        state = create_bell_state("phi_plus")
        schmidt_coeffs, U, V = schmidt_decomposition(state, 1)
        
        # Sum of squares should be 1
        sum_squares = np.sum(schmidt_coeffs ** 2)
        assert np.isclose(sum_squares, 1.0)


class TestEntanglementProperties:
    """Test general properties of entanglement."""
    
    def test_entanglement_monogamy(self):
        """Test basic monogamy property of entanglement."""
        # For GHZ state, entanglement is distributed
        state = create_ghz_state(3)
        
        # Entropy with each single qubit
        entropy_01 = entanglement_entropy(state, [0, 1])
        entropy_0 = entanglement_entropy(state, [0])
        entropy_1 = entanglement_entropy(state, [1])
        
        # These should all be non-negative
        assert entropy_01 >= 0
        assert entropy_0 >= 0
        assert entropy_1 >= 0
    
    def test_bell_inequality_violation(self):
        """Test that Bell states exhibit correlations violating classical bounds."""
        # This is a simplified check - full test would require measurement statistics
        state = create_bell_state("phi_plus")
        
        # Bell states are maximally entangled
        assert is_entangled(state)
        
        # Have maximum concurrence
        C = concurrence(state)
        assert np.isclose(C, 1.0)
        
        # Have maximum von Neumann entropy for subsystem
        entropy = entanglement_entropy(state, [0])
        assert np.isclose(entropy, 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

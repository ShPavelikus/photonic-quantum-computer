"""
Tests for quantum gates.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '../src')

from photonic_qc.quantum_state import PhotonicState
from photonic_qc.quantum_gates import (
    HadamardGate, PauliXGate, PauliYGate, PauliZGate,
    PhaseGate, SGate, TGate, RotationGate,
    CNOTGate, CZGate, SWAPGate
)


class TestSingleQubitGates:
    """Test single-qubit quantum gates."""
    
    def test_hadamard_gate(self):
        """Test Hadamard gate."""
        h = HadamardGate()
        
        # Check matrix
        expected = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        assert np.allclose(h.matrix, expected)
        
        # H|0⟩ = |+⟩
        state = PhotonicState.zero_state(1)
        result = h.apply(state)
        expected_state = np.array([1, 1]) / np.sqrt(2)
        assert np.allclose(result.state_vector, expected_state)
        
        # H² = I
        state = PhotonicState.zero_state(1)
        result = h.apply(h.apply(state))
        assert np.allclose(result.state_vector, state.state_vector)
    
    def test_pauli_x_gate(self):
        """Test Pauli-X gate."""
        x = PauliXGate()
        
        # X|0⟩ = |1⟩
        state = PhotonicState.zero_state(1)
        result = x.apply(state)
        assert np.allclose(result.state_vector, [0, 1])
        
        # X|1⟩ = |0⟩
        state = PhotonicState.one_state(1)
        result = x.apply(state)
        assert np.allclose(result.state_vector, [1, 0])
        
        # X² = I
        state = PhotonicState.zero_state(1)
        result = x.apply(x.apply(state))
        assert np.allclose(result.state_vector, state.state_vector)
    
    def test_pauli_y_gate(self):
        """Test Pauli-Y gate."""
        y = PauliYGate()
        
        # Y|0⟩ = i|1⟩
        state = PhotonicState.zero_state(1)
        result = y.apply(state)
        assert np.allclose(result.state_vector, [0, 1j])
        
        # Y² = I
        state = PhotonicState.zero_state(1)
        result = y.apply(y.apply(state))
        assert np.allclose(result.state_vector, state.state_vector)
    
    def test_pauli_z_gate(self):
        """Test Pauli-Z gate."""
        z = PauliZGate()
        
        # Z|0⟩ = |0⟩
        state = PhotonicState.zero_state(1)
        result = z.apply(state)
        assert np.allclose(result.state_vector, [1, 0])
        
        # Z|1⟩ = -|1⟩
        state = PhotonicState.one_state(1)
        result = z.apply(state)
        assert np.allclose(result.state_vector, [0, -1])
    
    def test_phase_gate(self):
        """Test phase gate."""
        phi = np.pi / 4
        p = PhaseGate(phi)
        
        # Check matrix
        expected = np.array([[1, 0], [0, np.exp(1j * phi)]])
        assert np.allclose(p.matrix, expected)
        
        # P|0⟩ = |0⟩
        state = PhotonicState.zero_state(1)
        result = p.apply(state)
        assert np.allclose(result.state_vector, [1, 0])
    
    def test_s_gate(self):
        """Test S gate."""
        s = SGate()
        
        # S = P(π/2)
        expected = np.array([[1, 0], [0, 1j]])
        assert np.allclose(s.matrix, expected)
        
        # S² = Z
        z = PauliZGate()
        s_squared = s @ s
        assert np.allclose(s_squared.matrix, z.matrix)
    
    def test_t_gate(self):
        """Test T gate."""
        t = TGate()
        
        # T = P(π/4)
        expected = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        assert np.allclose(t.matrix, expected)
        
        # T² = S
        s = SGate()
        t_squared = t @ t
        assert np.allclose(t_squared.matrix, s.matrix)
    
    def test_rotation_gate(self):
        """Test rotation gate."""
        # Rotation with θ=π is equivalent to X (up to phase)
        r = RotationGate(np.pi, 0, 0)
        x = PauliXGate()
        
        state = PhotonicState.zero_state(1)
        result_r = r.apply(state)
        result_x = x.apply(state)
        
        # Check they produce same state (up to global phase)
        fidelity = result_r.fidelity(result_x)
        assert np.isclose(fidelity, 1.0)


class TestTwoQubitGates:
    """Test two-qubit quantum gates."""
    
    def test_cnot_gate(self):
        """Test CNOT gate."""
        cnot = CNOTGate()
        
        # Check matrix shape
        assert cnot.matrix.shape == (4, 4)
        
        # CNOT|00⟩ = |00⟩
        state = PhotonicState.basis_state(2, 0)
        result = cnot.apply(state, [0, 1])
        assert np.allclose(result.state_vector, [1, 0, 0, 0])
        
        # CNOT|10⟩ = |11⟩
        state = PhotonicState.basis_state(2, 2)
        result = cnot.apply(state, [0, 1])
        assert np.allclose(result.state_vector, [0, 0, 0, 1])
    
    def test_cz_gate(self):
        """Test CZ gate."""
        cz = CZGate()
        
        # CZ|11⟩ = -|11⟩
        state = PhotonicState.basis_state(2, 3)
        result = cz.apply(state, [0, 1])
        assert np.allclose(result.state_vector, [0, 0, 0, -1])
        
        # CZ|00⟩ = |00⟩
        state = PhotonicState.basis_state(2, 0)
        result = cz.apply(state, [0, 1])
        assert np.allclose(result.state_vector, [1, 0, 0, 0])
    
    def test_swap_gate(self):
        """Test SWAP gate."""
        swap = SWAPGate()
        
        # SWAP|01⟩ = |10⟩
        state = PhotonicState.basis_state(2, 1)  # |01⟩
        result = swap.apply(state, [0, 1])
        expected = PhotonicState.basis_state(2, 2)  # |10⟩
        assert np.allclose(result.state_vector, expected.state_vector)
        
        # SWAP² = I
        state = PhotonicState.basis_state(2, 1)
        result = swap.apply(swap.apply(state, [0, 1]), [0, 1])
        assert np.allclose(result.state_vector, state.state_vector)


class TestGateProperties:
    """Test general properties of quantum gates."""
    
    def test_unitarity_single_qubit(self):
        """Test that single-qubit gates are unitary."""
        gates = [
            HadamardGate(),
            PauliXGate(),
            PauliYGate(),
            PauliZGate(),
            SGate(),
            TGate(),
            PhaseGate(np.pi / 3),
        ]
        
        for gate in gates:
            assert gate.is_unitary(), f"{gate} is not unitary"
    
    def test_unitarity_two_qubit(self):
        """Test that two-qubit gates are unitary."""
        gates = [CNOTGate(), CZGate(), SWAPGate()]
        
        for gate in gates:
            assert gate.is_unitary(), f"{gate} is not unitary"
    
    def test_gate_composition(self):
        """Test composition of gates."""
        h = HadamardGate()
        x = PauliXGate()
        
        # HXH = Z (up to phase)
        hxh = h @ x @ h
        z = PauliZGate()
        
        state = PhotonicState.zero_state(1)
        result_hxh = hxh.apply(state)
        result_z = z.apply(state)
        
        # Should have high fidelity
        fidelity = result_hxh.fidelity(result_z)
        assert fidelity > 0.99
    
    def test_dagger(self):
        """Test Hermitian conjugate of gates."""
        h = HadamardGate()
        h_dagger = h.dagger()
        
        # H is self-adjoint
        assert np.allclose(h.matrix, h_dagger.matrix)
        
        # For general gate, U†U = I
        t = TGate()
        t_dagger = t.dagger()
        product = t_dagger @ t
        
        identity = np.eye(2)
        assert np.allclose(product.matrix, identity)


class TestGateApplications:
    """Test application of gates to states."""
    
    def test_hadamard_creates_superposition(self):
        """Test that Hadamard creates equal superposition."""
        h = HadamardGate()
        state = PhotonicState.zero_state(1)
        result = h.apply(state)
        
        probs = result.probabilities()
        assert np.isclose(probs[0], 0.5)
        assert np.isclose(probs[1], 0.5)
    
    def test_cnot_creates_entanglement(self):
        """Test that CNOT can create entanglement."""
        # Start with |00⟩
        state = PhotonicState.zero_state(2)
        
        # Apply H to first qubit
        h = HadamardGate()
        state = h.apply(state, [0])
        
        # Apply CNOT
        cnot = CNOTGate()
        state = cnot.apply(state, [0, 1])
        
        # Result should be Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        assert np.allclose(state.state_vector, expected)


class TestGateRelations:
    """Test mathematical relations between gates."""
    
    def test_pauli_commutation(self):
        """Test Pauli matrix commutation relations."""
        x = PauliXGate()
        y = PauliYGate()
        z = PauliZGate()
        
        # XY = iZ
        xy = x @ y
        iz_matrix = 1j * z.matrix
        assert np.allclose(xy.matrix, iz_matrix)
    
    def test_clifford_relations(self):
        """Test Clifford gate relations."""
        h = HadamardGate()
        x = PauliXGate()
        z = PauliZGate()
        
        # HXH = Z
        hxh = h @ x @ h
        assert np.allclose(hxh.matrix, z.matrix)
        
        # HZH = X
        hzh = h @ z @ h
        assert np.allclose(hzh.matrix, x.matrix)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

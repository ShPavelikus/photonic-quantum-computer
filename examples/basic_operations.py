"""
Basic Operations Example

This example demonstrates fundamental operations with photonic quantum states:
- Creating quantum states
- Applying single-qubit gates
- Measuring states
- Computing probabilities
"""

import numpy as np
import sys
sys.path.insert(0, '../src')

from photonic_qc import PhotonicState, HadamardGate, PauliXGate, PauliZGate, Measurement


def main():
    print("=" * 60)
    print("Photonic Quantum Computer - Basic Operations")
    print("=" * 60)
    
    # 1. Creating basic states
    print("\n1. Creating Quantum States")
    print("-" * 40)
    
    # Create |0⟩ state
    state_0 = PhotonicState.zero_state(1)
    print(f"|0⟩ state: {state_0}")
    
    # Create |1⟩ state
    state_1 = PhotonicState.one_state(1)
    print(f"|1⟩ state: {state_1}")
    
    # Create superposition state
    superposition = PhotonicState.superposition(1)
    print(f"|+⟩ state (equal superposition): {superposition}")
    
    # Create custom state
    custom_state = PhotonicState(np.array([np.sqrt(0.3), np.sqrt(0.7)]))
    print(f"Custom state (0.3|0⟩ + 0.7|1⟩): {custom_state}")
    
    # 2. Applying single-qubit gates
    print("\n2. Applying Quantum Gates")
    print("-" * 40)
    
    # Apply Hadamard gate to |0⟩
    h_gate = HadamardGate()
    state_after_h = h_gate.apply(state_0)
    print(f"H|0⟩ = {state_after_h}")
    
    # Apply Pauli-X (NOT) gate to |0⟩
    x_gate = PauliXGate()
    state_after_x = x_gate.apply(state_0)
    print(f"X|0⟩ = {state_after_x}")
    
    # Apply Pauli-Z gate
    z_gate = PauliZGate()
    state_after_z = z_gate.apply(state_1)
    print(f"Z|1⟩ = {state_after_z}")
    
    # Compose gates: X then H
    state_xh = h_gate.apply(x_gate.apply(state_0))
    print(f"H·X|0⟩ = {state_xh}")
    
    # 3. Measurement probabilities
    print("\n3. Measurement Probabilities")
    print("-" * 40)
    
    # Create superposition state
    plus_state = h_gate.apply(state_0)
    probs = plus_state.probabilities()
    print(f"State: {plus_state}")
    print(f"P(|0⟩) = {probs[0]:.4f}")
    print(f"P(|1⟩) = {probs[1]:.4f}")
    
    # 4. Performing measurements
    print("\n4. Performing Measurements")
    print("-" * 40)
    
    # Measure superposition state multiple times
    plus_state = h_gate.apply(state_0)
    print(f"Initial state: {plus_state}")
    print("Measuring 10 times:")
    
    results = {'0': 0, '1': 0}
    for _ in range(10):
        measurement = Measurement()
        outcome, collapsed = measurement.measure_computational_basis(plus_state.copy())
        results[str(outcome)] += 1
    
    print(f"Outcomes: |0⟩ measured {results['0']} times, |1⟩ measured {results['1']} times")
    
    # 5. Multi-qubit states
    print("\n5. Multi-Qubit States")
    print("-" * 40)
    
    # Create 2-qubit state |00⟩
    state_00 = PhotonicState.zero_state(2)
    print(f"|00⟩ state: {state_00}")
    
    # Create 2-qubit state |11⟩
    state_11 = PhotonicState.one_state(2)
    print(f"|11⟩ state: {state_11}")
    
    # Apply Hadamard to first qubit
    state_h0 = h_gate.apply(state_00, [0])
    print(f"(H ⊗ I)|00⟩ = {state_h0}")
    
    # 6. State properties
    print("\n6. State Properties")
    print("-" * 40)
    
    state = PhotonicState(np.array([0.6, 0.8]))
    print(f"State: {state}")
    print(f"Is normalized: {state.is_normalized()}")
    print(f"Probability of |0⟩: {state.probability(0):.4f}")
    print(f"Probability of |1⟩: {state.probability(1):.4f}")
    
    # Density matrix
    rho = state.density_matrix()
    print(f"Density matrix:\n{rho}")
    
    # Fidelity between states
    state_a = PhotonicState.zero_state(1)
    state_b = h_gate.apply(state_a)
    fidelity = state_a.fidelity(state_b)
    print(f"\nFidelity between |0⟩ and |+⟩: {fidelity:.4f}")
    
    print("\n" + "=" * 60)
    print("Basic operations demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Bell States Example

This example demonstrates:
- Creation of Bell states (maximally entangled two-qubit states)
- Analysis of entanglement properties
- Measurements in different bases
"""

import numpy as np

from photonic_qc import (
    PhotonicState, HadamardGate, CNOTGate, 
    create_bell_state, entanglement_entropy, concurrence
)


def main():
    print("=" * 60)
    print("Photonic Quantum Computer - Bell States")
    print("=" * 60)
    
    # 1. Create all four Bell states directly
    print("\n1. Four Bell States")
    print("-" * 40)
    
    bell_types = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]
    bell_names = ["|Φ+⟩", "|Φ-⟩", "|Ψ+⟩", "|Ψ-⟩"]
    
    for bell_type, bell_name in zip(bell_types, bell_names):
        state = create_bell_state(bell_type)
        print(f"{bell_name} = {state}")
    
    # 2. Create Bell state using circuit
    print("\n2. Creating |Φ+⟩ using Quantum Circuit")
    print("-" * 40)
    
    # Start with |00⟩
    initial_state = PhotonicState.zero_state(2)
    print(f"Initial state: {initial_state}")
    
    # Apply Hadamard to first qubit
    h_gate = HadamardGate()
    after_h = h_gate.apply(initial_state, [0])
    print(f"After H on qubit 0: {after_h}")
    
    # Apply CNOT
    cnot_gate = CNOTGate()
    bell_state = cnot_gate.apply(after_h, [0, 1])
    print(f"After CNOT(0,1): {bell_state}")
    print(f"This is the Bell state |Φ+⟩")
    
    # 3. Analyze entanglement
    print("\n3. Entanglement Analysis")
    print("-" * 40)
    
    phi_plus = create_bell_state("phi_plus")
    
    # Compute entanglement entropy
    entropy_0 = entanglement_entropy(phi_plus, [0])
    print(f"Entanglement entropy (tracing out qubit 0): {entropy_0:.4f} bits")
    
    entropy_1 = entanglement_entropy(phi_plus, [1])
    print(f"Entanglement entropy (tracing out qubit 1): {entropy_1:.4f} bits")
    
    print(f"Maximum entanglement entropy for 1 qubit: {np.log2(2):.4f} bits")
    
    # Compute concurrence
    C = concurrence(phi_plus)
    print(f"Concurrence: {C:.4f} (1 = maximally entangled)")
    
    # 4. Compare with separable state
    print("\n4. Comparison with Separable State")
    print("-" * 40)
    
    # Create separable state |00⟩
    separable = PhotonicState.zero_state(2)
    print(f"Separable state: {separable}")
    
    entropy_sep = entanglement_entropy(separable, [0])
    print(f"Entanglement entropy: {entropy_sep:.4f} bits (0 = not entangled)")
    
    C_sep = concurrence(separable)
    print(f"Concurrence: {C_sep:.4f} (0 = not entangled)")
    
    # 5. Measurement correlations
    print("\n5. Measurement Correlations in Bell State")
    print("-" * 40)
    
    phi_plus = create_bell_state("phi_plus")
    print(f"State: {phi_plus}")
    print("Measurement probabilities:")
    
    probs = phi_plus.probabilities()
    print(f"P(|00⟩) = {probs[0]:.4f}")
    print(f"P(|01⟩) = {probs[1]:.4f}")
    print(f"P(|10⟩) = {probs[2]:.4f}")
    print(f"P(|11⟩) = {probs[3]:.4f}")
    
    print("\nNote: Only |00⟩ and |11⟩ have non-zero probability!")
    print("This shows perfect correlation: when one qubit is 0, other is 0;")
    print("when one is 1, other is 1.")
    
    # 6. Density matrix analysis
    print("\n6. Density Matrix of Bell State")
    print("-" * 40)
    
    rho = phi_plus.density_matrix()
    print("Density matrix ρ =")
    print(np.array2string(rho, precision=3, suppress_small=True))
    
    # Partial trace (reduced density matrix)
    rho_reduced = phi_plus.partial_trace([1])
    print("\nReduced density matrix (tracing out qubit 1):")
    print(np.array2string(rho_reduced, precision=3, suppress_small=True))
    print("Note: This is a maximally mixed state (identity matrix / 2)")
    print("indicating maximum entanglement!")
    
    # 7. All Bell states properties
    print("\n7. Properties of All Bell States")
    print("-" * 40)
    
    for bell_type, bell_name in zip(bell_types, bell_names):
        state = create_bell_state(bell_type)
        entropy = entanglement_entropy(state, [0])
        C = concurrence(state)
        print(f"{bell_name}: Entropy = {entropy:.4f}, Concurrence = {C:.4f}")
    
    print("\nAll Bell states are maximally entangled!")
    
    print("\n" + "=" * 60)
    print("Bell states demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

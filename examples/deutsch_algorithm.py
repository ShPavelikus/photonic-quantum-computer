"""
Deutsch Algorithm Example

This example demonstrates Deutsch's algorithm, which determines
whether a boolean function f: {0,1} -> {0,1} is constant or balanced
using only one query (compared to two queries classically required).
"""

import numpy as np

from photonic_qc import QuantumCircuit, PhotonicState


def create_constant_0_oracle(circuit):
    """Oracle for f(x) = 0 (constant function)"""
    # Do nothing - output qubit stays unchanged
    pass


def create_constant_1_oracle(circuit):
    """Oracle for f(x) = 1 (constant function)"""
    # Flip output qubit regardless of input
    circuit.x(1)


def create_balanced_identity_oracle(circuit):
    """Oracle for f(x) = x (balanced function)"""
    # CNOT: flip output if input is 1
    circuit.cnot(0, 1)


def create_balanced_not_oracle(circuit):
    """Oracle for f(x) = NOT x (balanced function)"""
    # Flip output if input is 0
    circuit.x(0)
    circuit.cnot(0, 1)
    circuit.x(0)


def deutsch_algorithm_step_by_step(oracle, oracle_name):
    """
    Run Deutsch's algorithm step by step with explanations.
    
    Args:
        oracle: Function that applies the oracle to a circuit
        oracle_name: Name of the oracle for display
        
    Returns:
        Result: "constant" or "balanced"
    """
    print(f"\nRunning Deutsch's Algorithm with {oracle_name}")
    print("-" * 60)
    
    # Create circuit with 2 qubits
    circuit = QuantumCircuit(2)
    
    # Step 1: Initialize
    print("Step 1: Initialize qubits")
    print("  Qubit 0 (input):  |0⟩")
    print("  Qubit 1 (output): |0⟩")
    
    # Step 2: Prepare output qubit in |1⟩
    circuit.x(1)
    state = circuit.get_statevector()
    print("\nStep 2: Flip output qubit to |1⟩")
    print(f"  State: {state}")
    
    # Step 3: Apply Hadamard to both qubits
    circuit.h(0)
    circuit.h(1)
    state = circuit.get_statevector()
    print("\nStep 3: Apply Hadamard to both qubits")
    print(f"  Input:  (|0⟩ + |1⟩)/√2")
    print(f"  Output: (|0⟩ - |1⟩)/√2")
    
    # Step 4: Apply oracle
    print(f"\nStep 4: Apply oracle ({oracle_name})")
    oracle(circuit)
    state = circuit.get_statevector()
    print(f"  Oracle applied")
    
    # Step 5: Apply Hadamard to input qubit
    circuit.h(0)
    state = circuit.get_statevector()
    print("\nStep 5: Apply Hadamard to input qubit")
    print(f"  Final state before measurement")
    
    # Step 6: Measure input qubit
    circuit.measure(0)
    results = circuit.run(shots=10)
    
    print("\nStep 6: Measure input qubit")
    print(f"  Measurement results (10 shots): {results}")
    
    # Determine result
    # If we measure 0, function is constant
    # If we measure 1, function is balanced
    most_common = max(results, key=results.get)
    result = "constant" if most_common[0] == '0' else "balanced"
    
    print(f"\nResult: Function is {result}!")
    return result


def main():
    print("=" * 60)
    print("Photonic Quantum Computer - Deutsch's Algorithm")
    print("=" * 60)
    
    print("""
Deutsch's Algorithm Overview:
-----------------------------
Goal: Determine if f: {0,1} -> {0,1} is constant or balanced

Classical approach: Must evaluate f(0) and f(1) (2 queries)
Quantum approach:   Only 1 query needed!

Function types:
- Constant: f(0) = f(1) [both 0 or both 1]
- Balanced: f(0) ≠ f(1) [one 0, one 1]
""")
    
    # Test all four possible boolean functions
    oracles = [
        (create_constant_0_oracle, "f(x) = 0 (constant)"),
        (create_constant_1_oracle, "f(x) = 1 (constant)"),
        (create_balanced_identity_oracle, "f(x) = x (balanced)"),
        (create_balanced_not_oracle, "f(x) = NOT x (balanced)"),
    ]
    
    print("\n" + "=" * 60)
    print("Testing All Four Boolean Functions")
    print("=" * 60)
    
    for oracle, name in oracles:
        result = deutsch_algorithm_step_by_step(oracle, name)
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("Classical vs Quantum Comparison")
    print("=" * 60)
    print("""
Classical Algorithm:
  - Must query f(0) and f(1)
  - Total queries: 2
  - Time complexity: O(2)

Deutsch's Algorithm (Quantum):
  - Evaluates f on superposition
  - Total queries: 1
  - Time complexity: O(1)
  - Quantum advantage: 2x speedup
  
This demonstrates quantum parallelism:
  The quantum algorithm evaluates f on both inputs simultaneously
  due to superposition!
""")
    
    # Generalization
    print("=" * 60)
    print("Deutsch-Jozsa Algorithm (Generalization)")
    print("=" * 60)
    print("""
For n-bit input function f: {0,1}^n -> {0,1}:

Classical:
  - Worst case: 2^(n-1) + 1 queries
  - Exponential in n

Deutsch-Jozsa (Quantum):
  - Always: 1 query
  - Exponential speedup!
  
This was one of the first examples demonstrating that
quantum computers can be exponentially faster than
classical computers for certain problems.
""")
    
    print("=" * 60)
    print("Deutsch's algorithm demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

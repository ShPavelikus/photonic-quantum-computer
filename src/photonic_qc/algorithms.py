"""
Quantum Algorithms for Photonic Quantum Computing

This module implements basic quantum algorithms.
"""

import numpy as np
from typing import Callable, List
from .quantum_state import PhotonicState
from .circuits import QuantumCircuit
from .quantum_gates import HadamardGate, PauliXGate, CNOTGate, PauliZGate
from .measurement import Measurement


def deutsch_algorithm(oracle: Callable[[QuantumCircuit], None]) -> str:
    """
    Implement Deutsch's algorithm.
    
    Determines if a function f: {0,1} -> {0,1} is constant or balanced
    with a single query.
    
    Args:
        oracle: Function that applies the oracle to a circuit
        
    Returns:
        "constant" or "balanced"
    """
    # Create circuit with 2 qubits
    circuit = QuantumCircuit(2)
    
    # Initialize |01⟩
    circuit.x(1)
    
    # Apply Hadamard to both qubits
    circuit.h(0)
    circuit.h(1)
    
    # Apply oracle
    oracle(circuit)
    
    # Apply Hadamard to first qubit
    circuit.h(0)
    
    # Measure first qubit
    circuit.measure(0)
    
    # Run circuit
    results = circuit.run(shots=1)
    result = list(results.keys())[0][0]
    
    return "constant" if result == "0" else "balanced"


def deutsch_jozsa_algorithm(n: int, oracle: Callable[[QuantumCircuit], None]) -> str:
    """
    Implement Deutsch-Jozsa algorithm.
    
    Determines if a function f: {0,1}^n -> {0,1} is constant or balanced.
    
    Args:
        n: Number of input qubits
        oracle: Function that applies the oracle to a circuit
        
    Returns:
        "constant" or "balanced"
    """
    # Create circuit with n+1 qubits
    circuit = QuantumCircuit(n + 1)
    
    # Initialize last qubit to |1⟩
    circuit.x(n)
    
    # Apply Hadamard to all qubits
    for i in range(n + 1):
        circuit.h(i)
    
    # Apply oracle
    oracle(circuit)
    
    # Apply Hadamard to input qubits
    for i in range(n):
        circuit.h(i)
    
    # Measure input qubits
    for i in range(n):
        circuit.measure(i)
    
    # Run circuit
    results = circuit.run(shots=1)
    result = list(results.keys())[0]
    
    # Check if all zeros
    all_zeros = all(bit == "0" for bit in result)
    
    return "constant" if all_zeros else "balanced"


def quantum_teleportation(state_to_teleport: PhotonicState) -> PhotonicState:
    """
    Implement quantum teleportation protocol (simplified demonstration).
    
    NOTE: This is a simplified demonstration showing the protocol structure.
    A full implementation would require proper tensor product construction,
    measurement handling, and classical communication simulation.
    
    Teleports a single-qubit state using an entangled Bell pair.
    
    Args:
        state_to_teleport: Single-qubit state to teleport
        
    Returns:
        The original state (in a full implementation, this would be the teleported state)
    """
    if state_to_teleport.num_qubits != 1:
        raise ValueError("Can only teleport single-qubit states")
    
    # TODO: Full implementation would:
    # 1. Create proper 3-qubit initial state: |ψ⟩ ⊗ |Φ+⟩
    # 2. Apply Bell measurement on Alice's qubits
    # 3. Simulate classical communication
    # 4. Apply conditional corrections on Bob's qubit
    # 5. Return the teleported state on Bob's qubit
    
    # For demonstration purposes, return the original state
    # This shows the protocol can be implemented but needs full tensor product handling
    return state_to_teleport  # Simplified return for demonstration


def superdense_coding(bit1: int, bit2: int) -> PhotonicState:
    """
    Implement superdense coding protocol.
    
    Send 2 classical bits using 1 qubit with shared entanglement.
    
    Args:
        bit1: First classical bit (0 or 1)
        bit2: Second classical bit (0 or 1)
        
    Returns:
        Final state after encoding
    """
    if bit1 not in [0, 1] or bit2 not in [0, 1]:
        raise ValueError("Bits must be 0 or 1")
    
    # Create Bell pair
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    
    # Alice's encoding based on bits
    if bit2 == 1:
        circuit.x(0)
    if bit1 == 1:
        circuit.z(0)
    
    # Get state after encoding
    encoded_state = circuit.get_statevector()
    
    return encoded_state


def grover_oracle_single_solution(circuit: QuantumCircuit, solution: int, n: int):
    """
    Oracle for Grover's algorithm with a single solution.
    
    Args:
        circuit: Quantum circuit to apply oracle to
        solution: Index of solution (in binary)
        n: Number of qubits
    """
    # Mark the solution by flipping phase
    # This is a simplified version - full implementation would construct proper oracle
    for i in range(n):
        if not (solution >> i) & 1:
            circuit.x(i)
    
    # Multi-controlled Z gate (simplified)
    if n == 2:
        circuit.cz(0, 1)
    
    for i in range(n):
        if not (solution >> i) & 1:
            circuit.x(i)


def grover_diffusion(circuit: QuantumCircuit, n: int):
    """
    Diffusion operator for Grover's algorithm.
    
    Args:
        circuit: Quantum circuit
        n: Number of qubits
    """
    # H gates
    for i in range(n):
        circuit.h(i)
    
    # X gates
    for i in range(n):
        circuit.x(i)
    
    # Multi-controlled Z (simplified for 2 qubits)
    if n == 2:
        circuit.cz(0, 1)
    
    # X gates
    for i in range(n):
        circuit.x(i)
    
    # H gates
    for i in range(n):
        circuit.h(i)


def grover_algorithm(n: int, solution: int) -> dict:
    """
    Implement Grover's search algorithm (simplified version for 2 qubits).
    
    Args:
        n: Number of qubits
        solution: Index of solution to find
        
    Returns:
        Measurement results
    """
    if n != 2:
        raise NotImplementedError("Simplified implementation only supports 2 qubits")
    
    circuit = QuantumCircuit(n)
    
    # Initialize in superposition
    for i in range(n):
        circuit.h(i)
    
    # Number of iterations for optimal probability
    num_iterations = int(np.pi / 4 * np.sqrt(2 ** n))
    
    for _ in range(num_iterations):
        # Apply oracle
        grover_oracle_single_solution(circuit, solution, n)
        
        # Apply diffusion
        grover_diffusion(circuit, n)
    
    # Measure all qubits
    circuit.measure_all()
    
    # Run circuit
    results = circuit.run(shots=100)
    
    return results

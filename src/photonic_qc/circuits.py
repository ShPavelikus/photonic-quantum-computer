"""
Quantum Circuits for Photonic Quantum Computing

This module provides a framework for building and simulating quantum circuits.
"""

import numpy as np
from typing import List, Tuple, Optional
from .quantum_state import PhotonicState
from .quantum_gates import QuantumGate
from .measurement import Measurement


class QuantumCircuit:
    """
    Quantum circuit for composing and executing quantum operations.
    
    Attributes:
        num_qubits (int): Number of qubits in the circuit
        operations (List): List of (gate, target_qubits) tuples
        measurements (List): List of measurement operations
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize a quantum circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
        """
        self.num_qubits = num_qubits
        self.operations: List[Tuple[QuantumGate, List[int]]] = []
        self.measurements: List[int] = []
        self.state: Optional[PhotonicState] = None
    
    def add_gate(self, gate: QuantumGate, target_qubits: List[int]):
        """
        Add a gate to the circuit.
        
        Args:
            gate: Quantum gate to add
            target_qubits: List of target qubit indices
        """
        if not target_qubits:
            raise ValueError("Must specify target qubits")
        
        if max(target_qubits) >= self.num_qubits or min(target_qubits) < 0:
            raise ValueError("Target qubit index out of range")
        
        if len(target_qubits) != gate.num_qubits:
            raise ValueError(f"Gate requires {gate.num_qubits} qubits but {len(target_qubits)} provided")
        
        self.operations.append((gate, target_qubits))
    
    def h(self, qubit: int):
        """Add Hadamard gate."""
        from .quantum_gates import HadamardGate
        self.add_gate(HadamardGate(), [qubit])
    
    def x(self, qubit: int):
        """Add Pauli-X gate."""
        from .quantum_gates import PauliXGate
        self.add_gate(PauliXGate(), [qubit])
    
    def y(self, qubit: int):
        """Add Pauli-Y gate."""
        from .quantum_gates import PauliYGate
        self.add_gate(PauliYGate(), [qubit])
    
    def z(self, qubit: int):
        """Add Pauli-Z gate."""
        from .quantum_gates import PauliZGate
        self.add_gate(PauliZGate(), [qubit])
    
    def s(self, qubit: int):
        """Add S gate."""
        from .quantum_gates import SGate
        self.add_gate(SGate(), [qubit])
    
    def t(self, qubit: int):
        """Add T gate."""
        from .quantum_gates import TGate
        self.add_gate(TGate(), [qubit])
    
    def phase(self, qubit: int, angle: float):
        """Add phase gate."""
        from .quantum_gates import PhaseGate
        self.add_gate(PhaseGate(angle), [qubit])
    
    def cnot(self, control: int, target: int):
        """Add CNOT gate."""
        from .quantum_gates import CNOTGate
        self.add_gate(CNOTGate(), [control, target])
    
    def cz(self, control: int, target: int):
        """Add CZ gate."""
        from .quantum_gates import CZGate
        self.add_gate(CZGate(), [control, target])
    
    def swap(self, qubit1: int, qubit2: int):
        """Add SWAP gate."""
        from .quantum_gates import SWAPGate
        self.add_gate(SWAPGate(), [qubit1, qubit2])
    
    def measure(self, qubit: int):
        """
        Add measurement operation.
        
        Args:
            qubit: Qubit to measure
        """
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError("Qubit index out of range")
        self.measurements.append(qubit)
    
    def measure_all(self):
        """Add measurement of all qubits."""
        self.measurements = list(range(self.num_qubits))
    
    def run(self, initial_state: Optional[PhotonicState] = None, shots: int = 1) -> dict:
        """
        Execute the circuit.
        
        Args:
            initial_state: Initial quantum state (default: |0...0⟩)
            shots: Number of times to run the circuit
            
        Returns:
            Dictionary with measurement results
        """
        if initial_state is None:
            initial_state = PhotonicState.zero_state(self.num_qubits)
        
        if initial_state.num_qubits != self.num_qubits:
            raise ValueError("Initial state has wrong number of qubits")
        
        results = {}
        
        for _ in range(shots):
            # Start with initial state
            state = initial_state.copy()
            
            # Apply all gates
            for gate, targets in self.operations:
                state = gate.apply(state, targets)
            
            # Perform measurements
            if self.measurements:
                measurement_result = []
                for qubit in sorted(self.measurements):
                    outcome, state = Measurement.measure_computational_basis(state, qubit)
                    measurement_result.append(str(outcome))
                
                result_str = "".join(measurement_result)
                results[result_str] = results.get(result_str, 0) + 1
            else:
                # No measurements, return final state
                self.state = state
        
        return results
    
    def get_statevector(self, initial_state: Optional[PhotonicState] = None) -> PhotonicState:
        """
        Get the final state vector without measurement.
        
        Args:
            initial_state: Initial quantum state (default: |0...0⟩)
            
        Returns:
            Final quantum state
        """
        if initial_state is None:
            initial_state = PhotonicState.zero_state(self.num_qubits)
        
        state = initial_state.copy()
        
        # Apply all gates
        for gate, targets in self.operations:
            state = gate.apply(state, targets)
        
        return state
    
    def draw(self) -> str:
        """
        Generate ASCII art representation of the circuit.
        
        Returns:
            String representation of the circuit
        """
        lines = [f"q{i}: " for i in range(self.num_qubits)]
        
        # Simple visualization
        for gate, targets in self.operations:
            gate_name = gate.__class__.__name__.replace("Gate", "")
            
            for i, line in enumerate(lines):
                if i in targets:
                    lines[i] += f"─[{gate_name}]─"
                else:
                    lines[i] += "─" * (len(gate_name) + 4)
        
        # Add measurements
        if self.measurements:
            for i, line in enumerate(lines):
                if i in self.measurements:
                    lines[i] += "─[M]"
                else:
                    lines[i] += "────"
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """String representation of circuit."""
        return f"QuantumCircuit({self.num_qubits} qubits, {len(self.operations)} operations)"
    
    def __repr__(self) -> str:
        """Representation of circuit."""
        return self.__str__()

using PhotonicQuantumComputer;
using System;

Console.WriteLine("=== Photonic Quantum Computer Examples ===\n");

// Example 1: Basic Quantum State Operations
Console.WriteLine("Example 1: Basic Quantum State Operations");
var zeroState = PhotonicState.ZeroState(2);
Console.WriteLine($"Zero State: {zeroState}");

var superposition = PhotonicState.Superposition(2);
Console.WriteLine($"Superposition: {superposition}");
Console.WriteLine($"Is Normalized: {superposition.IsNormalized()}\n");

// Example 2: Quantum Gates
Console.WriteLine("Example 2: Applying Quantum Gates");
var state = PhotonicState.ZeroState(1);
var hadamard = new HadamardGate();
var result = hadamard.Apply(state, new[] { 0 });
Console.WriteLine($"After Hadamard: {result}\n");

// Example 3: Bell State Creation
Console.WriteLine("Example 3: Creating Bell States");
var bellState = Entanglement.CreateBellState("phi_plus");
Console.WriteLine($"Bell State |Φ+⟩: {bellState}");
Console.WriteLine($"Is Entangled: {Entanglement.IsEntangled(bellState)}\n");

// Example 4: Quantum Circuit
Console.WriteLine("Example 4: Building and Running a Quantum Circuit");
var circuit = new QuantumCircuit(2);
circuit.H(0);
circuit.Cnot(0, 1);
circuit.MeasureAll();

var results = circuit.Run(shots: 100);
Console.WriteLine("Measurement Results (Bell State Circuit):");
foreach (var (outcome, count) in results.OrderByDescending(kv => kv.Value))
{
    Console.WriteLine($"  {outcome}: {count} times");
}
Console.WriteLine();

// Example 5: Deutsch Algorithm
Console.WriteLine("Example 5: Deutsch Algorithm");
void constantOracle(QuantumCircuit c) { /* Do nothing - constant 0 */ }
void balancedOracle(QuantumCircuit c) { c.Cnot(0, 1); }

var constantResult = Algorithms.DeutschAlgorithm(constantOracle);
var balancedResult = Algorithms.DeutschAlgorithm(balancedOracle);
Console.WriteLine($"Constant Oracle Result: {constantResult}");
Console.WriteLine($"Balanced Oracle Result: {balancedResult}\n");

// Example 6: Grover's Algorithm (Bug #3 Fixed - Now works with N qubits!)
Console.WriteLine("Example 6: Grover's Algorithm (Searching for |10⟩ in 2-qubit space)");
var groverResults = Algorithms.GroverAlgorithm(2, solution: 2);
Console.WriteLine("Grover Search Results:");
foreach (var (outcome, count) in groverResults.OrderByDescending(kv => kv.Value).Take(3))
{
    Console.WriteLine($"  {outcome}: {count} times");
}
Console.WriteLine();

Console.WriteLine("=== All Examples Complete ===");


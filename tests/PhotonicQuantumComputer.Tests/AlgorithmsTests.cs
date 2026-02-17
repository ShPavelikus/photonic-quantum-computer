using Xunit;
using System.Numerics;
using PhotonicQuantumComputer;

namespace PhotonicQuantumComputer.Tests;

public class AlgorithmsTests
{
    [Fact]
    public void DeutschAlgorithm_ConstantFunction_ReturnsConstant()
    {
        // Constant oracle (always returns 0)
        void constantOracle(QuantumCircuit circuit)
        {
            // Do nothing - output is always 0
        }

        var result = Algorithms.DeutschAlgorithm(constantOracle);
        Assert.Equal("constant", result);
    }

    [Fact]
    public void DeutschAlgorithm_BalancedFunction_ReturnsBalanced()
    {
        // Balanced oracle (CNOT)
        void balancedOracle(QuantumCircuit circuit)
        {
            circuit.Cnot(0, 1);
        }

        var result = Algorithms.DeutschAlgorithm(balancedOracle);
        Assert.Equal("balanced", result);
    }

    [Fact]
    public void DeutschJozsaAlgorithm_ConstantFunction_ReturnsConstant()
    {
        // Constant oracle for 3 qubits
        void constantOracle(QuantumCircuit circuit)
        {
            // Do nothing - output is always 0
        }

        var result = Algorithms.DeutschJozsaAlgorithm(3, constantOracle);
        Assert.Equal("constant", result);
    }

    [Fact]
    public void QuantumTeleportation_TeleportsState()
    {
        // BUG #2 FIX TEST: Quantum teleportation should now work properly
        var originalState = PhotonicState.Superposition(1);
        var teleportedState = Algorithms.QuantumTeleportation(originalState);
        
        // The teleported state should have high fidelity with the original
        var fidelity = originalState.Fidelity(teleportedState);
        Assert.True(fidelity > 0.9, $"Fidelity {fidelity} should be close to 1");
    }

    [Fact]
    public void QuantumTeleportation_TeleportsZeroState()
    {
        // BUG #2 FIX TEST: Test with |0⟩ state
        var originalState = PhotonicState.ZeroState(1);
        var teleportedState = Algorithms.QuantumTeleportation(originalState);
        
        var fidelity = originalState.Fidelity(teleportedState);
        Assert.True(fidelity > 0.99, $"Fidelity {fidelity} should be very close to 1 for |0⟩");
    }

    [Fact]
    public void QuantumTeleportation_TeleportsOneState()
    {
        // BUG #2 FIX TEST: Test with |1⟩ state
        var originalState = PhotonicState.OneState(1);
        var teleportedState = Algorithms.QuantumTeleportation(originalState);
        
        var fidelity = originalState.Fidelity(teleportedState);
        Assert.True(fidelity > 0.99, $"Fidelity {fidelity} should be very close to 1 for |1⟩");
    }

    [Fact]
    public void SuperdenseCoding_EncodesCorrectly()
    {
        var state00 = Algorithms.SuperdenseCoding(0, 0);
        var state01 = Algorithms.SuperdenseCoding(0, 1);
        var state10 = Algorithms.SuperdenseCoding(1, 0);
        var state11 = Algorithms.SuperdenseCoding(1, 1);
        
        // All should be valid quantum states
        Assert.True(state00.IsNormalized());
        Assert.True(state01.IsNormalized());
        Assert.True(state10.IsNormalized());
        Assert.True(state11.IsNormalized());
    }

    [Fact]
    public void GroverAlgorithm_2Qubits_FindsSolution()
    {
        // BUG #3 FIX TEST: Test with 2 qubits (original implementation)
        int solution = 2; // Looking for |10⟩
        var results = Algorithms.GroverAlgorithm(2, solution);
        
        // The solution should be the most frequent result
        var mostFrequent = results.OrderByDescending(kv => kv.Value).First();
        int foundSolution = Convert.ToInt32(mostFrequent.Key, 2);
        
        Assert.Equal(solution, foundSolution);
    }

    [Fact]
    public void GroverAlgorithm_3Qubits_FindsSolution()
    {
        // BUG #3 FIX TEST: Test with 3 qubits (should now work!)
        int solution = 5; // Looking for |101⟩
        var results = Algorithms.GroverAlgorithm(3, solution);
        
        // The solution should be among the top results
        Assert.True(results.Count > 0);
        var mostFrequent = results.OrderByDescending(kv => kv.Value).First();
        int foundSolution = Convert.ToInt32(mostFrequent.Key, 2);
        
        // Grover's algorithm should find the solution with high probability
        Assert.Equal(solution, foundSolution);
    }

    [Fact]
    public void GroverAlgorithm_4Qubits_FindsSolution()
    {
        // BUG #3 FIX TEST: Test with 4 qubits (should now work!)
        int solution = 10; // Looking for |1010⟩
        var results = Algorithms.GroverAlgorithm(4, solution);
        
        Assert.True(results.Count > 0);
        var mostFrequent = results.OrderByDescending(kv => kv.Value).First();
        int foundSolution = Convert.ToInt32(mostFrequent.Key, 2);
        
        // Check that the solution is found
        Assert.Equal(solution, foundSolution);
    }
}

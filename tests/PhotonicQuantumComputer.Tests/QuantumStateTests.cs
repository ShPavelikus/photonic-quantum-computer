using Xunit;
using System.Numerics;
using PhotonicQuantumComputer;

namespace PhotonicQuantumComputer.Tests;

public class QuantumStateTests
{
    [Fact]
    public void ZeroState_CreatesCorrectState()
    {
        var state = PhotonicState.ZeroState(2);
        Assert.Equal(2, state.NumQubits);
        Assert.Equal(4, state.StateVector.Length);
        Assert.Equal(Complex.One, state.StateVector[0]);
        Assert.Equal(Complex.Zero, state.StateVector[1]);
        Assert.Equal(Complex.Zero, state.StateVector[2]);
        Assert.Equal(Complex.Zero, state.StateVector[3]);
    }

    [Fact]
    public void OneState_CreatesCorrectState()
    {
        var state = PhotonicState.OneState(2);
        Assert.Equal(2, state.NumQubits);
        Assert.Equal(Complex.One, state.StateVector[3]);
    }

    [Fact]
    public void BasisState_CreatesCorrectState()
    {
        var state = PhotonicState.BasisState(2, 2);
        Assert.Equal(Complex.One, state.StateVector[2]);
        Assert.Equal(Complex.Zero, state.StateVector[0]);
        Assert.Equal(Complex.Zero, state.StateVector[1]);
        Assert.Equal(Complex.Zero, state.StateVector[3]);
    }

    [Fact]
    public void Superposition_CreatesEqualSuperposition()
    {
        var state = PhotonicState.Superposition(2);
        double expected = 0.5;
        
        foreach (var amplitude in state.StateVector)
        {
            Assert.True(Math.Abs(amplitude.Real - expected) < 1e-10);
            Assert.True(Math.Abs(amplitude.Imaginary) < 1e-10);
        }
    }

    [Fact]
    public void IsNormalized_ReturnsTrueForNormalizedState()
    {
        var state = PhotonicState.ZeroState(2);
        Assert.True(state.IsNormalized());
    }

    [Fact]
    public void Probability_ComputesCorrectProbability()
    {
        var state = PhotonicState.BasisState(2, 1);
        Assert.Equal(1.0, state.Probability(1), 10);
        Assert.Equal(0.0, state.Probability(0), 10);
    }

    [Fact]
    public void InnerProduct_ComputesCorrectValue()
    {
        var state1 = PhotonicState.ZeroState(2);
        var state2 = PhotonicState.ZeroState(2);
        var product = state1.InnerProduct(state2);
        Assert.Equal(Complex.One, product);
    }

    [Fact]
    public void Fidelity_ReturnOne_ForIdenticalStates()
    {
        var state1 = PhotonicState.ZeroState(2);
        var state2 = PhotonicState.ZeroState(2);
        Assert.Equal(1.0, state1.Fidelity(state2), 10);
    }
}

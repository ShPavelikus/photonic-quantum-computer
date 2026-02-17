using Xunit;
using System.Numerics;
using PhotonicQuantumComputer;

namespace PhotonicQuantumComputer.Tests;

public class QuantumGatesTests
{
    [Fact]
    public void HadamardGate_CreatesSuperposition()
    {
        var state = PhotonicState.ZeroState(1);
        var h = new HadamardGate();
        var result = h.Apply(state, new[] { 0 });
        
        double expectedAmplitude = 1.0 / Math.Sqrt(2);
        Assert.True(Math.Abs(result.StateVector[0].Real - expectedAmplitude) < 1e-10);
        Assert.True(Math.Abs(result.StateVector[1].Real - expectedAmplitude) < 1e-10);
    }

    [Fact]
    public void PauliXGate_FlipsBit()
    {
        var state = PhotonicState.ZeroState(1);
        var x = new PauliXGate();
        var result = x.Apply(state, new[] { 0 });
        
        Assert.Equal(Complex.Zero, result.StateVector[0]);
        Assert.Equal(Complex.One, result.StateVector[1]);
    }

    [Fact]
    public void CnotGate_OnAdjacentQubits_Works()
    {
        var state = PhotonicState.BasisState(2, 2); // |10⟩
        var cnot = new CnotGate();
        var result = cnot.Apply(state, new[] { 0, 1 });
        
        // Should flip to |11⟩ (index 3)
        Assert.Equal(Complex.One, result.StateVector[3]);
        Assert.Equal(Complex.Zero, result.StateVector[2]);
    }

    [Fact]
    public void CnotGate_OnNonAdjacentQubits_Works()
    {
        // BUG #1 FIX TEST: This should work now with non-adjacent qubits
        var state = PhotonicState.BasisState(3, 4); // |100⟩
        var cnot = new CnotGate();
        
        // Apply CNOT with control=0, target=2 (non-adjacent!)
        var result = cnot.Apply(state, new[] { 0, 2 });
        
        // Should flip qubit 2: |100⟩ → |101⟩ (index 5)
        Assert.Equal(Complex.One, result.StateVector[5]);
        Assert.Equal(Complex.Zero, result.StateVector[4]);
    }

    [Fact]
    public void CzGate_OnAdjacentQubits_Works()
    {
        // Create |11⟩ state
        var state = PhotonicState.BasisState(2, 3);
        var cz = new CzGate();
        var result = cz.Apply(state, new[] { 0, 1 });
        
        // CZ should add a phase of -1 to |11⟩
        Assert.True(Math.Abs(result.StateVector[3].Real - (-1.0)) < 1e-10);
    }

    [Fact]
    public void SwapGate_SwapsQubits()
    {
        var state = PhotonicState.BasisState(2, 1); // |01⟩
        var swap = new SwapGate();
        var result = swap.Apply(state, new[] { 0, 1 });
        
        // Should swap to |10⟩ (index 2)
        Assert.Equal(Complex.One, result.StateVector[2]);
        Assert.Equal(Complex.Zero, result.StateVector[1]);
    }

    [Fact]
    public void SGate_AppliesPhase()
    {
        var state = PhotonicState.BasisState(1, 1); // |1⟩
        var s = new SGate();
        var result = s.Apply(state, new[] { 0 });
        
        // S gate applies phase i to |1⟩
        Assert.True(Math.Abs(result.StateVector[1].Imaginary - 1.0) < 1e-10);
        Assert.True(Math.Abs(result.StateVector[1].Real) < 1e-10);
    }

    [Fact]
    public void TGate_AppliesPhase()
    {
        var state = PhotonicState.BasisState(1, 1); // |1⟩
        var t = new TGate();
        var result = t.Apply(state, new[] { 0 });
        
        // T gate applies phase e^(iπ/4)
        double expectedReal = Math.Cos(Math.PI / 4);
        double expectedImag = Math.Sin(Math.PI / 4);
        Assert.True(Math.Abs(result.StateVector[1].Real - expectedReal) < 1e-10);
        Assert.True(Math.Abs(result.StateVector[1].Imaginary - expectedImag) < 1e-10);
    }

    [Fact]
    public void IsUnitary_ReturnsTrueForUnitaryGates()
    {
        Assert.True(new HadamardGate().IsUnitary());
        Assert.True(new PauliXGate().IsUnitary());
        Assert.True(new PauliYGate().IsUnitary());
        Assert.True(new PauliZGate().IsUnitary());
        Assert.True(new CnotGate().IsUnitary());
    }

    [Fact]
    public void TensorProduct_ComputesCorrectProduct()
    {
        var h = new HadamardGate();
        var x = new PauliXGate();
        var product = QuantumGates.TensorProduct(h, x);
        
        Assert.Equal(2, product.NumQubits);
        Assert.True(product.IsUnitary());
    }
}

using System.Numerics;

namespace PhotonicQuantumComputer;

/// <summary>
/// Quantum measurement operations.
/// </summary>
public static class Measurement
{
    private static readonly Random _random = new Random();

    /// <summary>
    /// Perform a projective measurement in the computational basis.
    /// </summary>
    /// <param name="state">Quantum state to measure</param>
    /// <param name="qubit">Specific qubit to measure (null for all qubits)</param>
    /// <returns>Tuple of (measurement_outcome, collapsed_state)</returns>
    public static (int outcome, PhotonicState collapsedState) MeasureComputationalBasis(
        PhotonicState state, 
        int? qubit = null)
    {
        if (qubit == null)
        {
            // Measure all qubits
            double[] probabilities = state.Probabilities();
            int outcome = SampleFromDistribution(probabilities);

            // Collapse to measured basis state
            var collapsed = PhotonicState.BasisState(state.NumQubits, outcome);
            return (outcome, collapsed);
        }
        else
        {
            // Measure single qubit
            int qubitIndex = qubit.Value;
            if (qubitIndex < 0 || qubitIndex >= state.NumQubits)
            {
                throw new ArgumentException($"Qubit index {qubitIndex} out of range");
            }

            // Compute probabilities for qubit being 0 or 1
            double prob0 = 0.0;
            for (int i = 0; i < state.StateVector.Length; i++)
            {
                if (((i >> qubitIndex) & 1) == 0)  // Check if qubit is 0
                {
                    var c = state.StateVector[i];
                    prob0 += c.Real * c.Real + c.Imaginary * c.Imaginary;
                }
            }

            // Measure
            int outcome = _random.NextDouble() < prob0 ? 0 : 1;

            // Collapse state
            var newVector = (Complex[])state.StateVector.Clone();
            for (int i = 0; i < newVector.Length; i++)
            {
                if (((i >> qubitIndex) & 1) != outcome)
                {
                    newVector[i] = Complex.Zero;
                }
            }

            var collapsed = new PhotonicState(newVector, normalize: true);
            return (outcome, collapsed);
        }
    }

    /// <summary>
    /// Measure in the X basis (|+⟩, |-⟩).
    /// </summary>
    /// <param name="state">Quantum state to measure</param>
    /// <param name="qubit">Qubit to measure</param>
    /// <returns>Tuple of (outcome, collapsed_state) where outcome is 0 for |+⟩ or 1 for |-⟩</returns>
    public static (int outcome, PhotonicState collapsedState) MeasureXBasis(
        PhotonicState state, 
        int qubit)
    {
        // Transform to X basis by applying Hadamard
        var hGate = new HadamardGate();
        var transformed = hGate.Apply(state, new[] { qubit });

        // Measure in computational basis
        var (outcome, collapsed) = MeasureComputationalBasis(transformed, qubit);

        // Transform back
        var finalState = hGate.Apply(collapsed, new[] { qubit });

        return (outcome, finalState);
    }

    /// <summary>
    /// Measure in the Y basis.
    /// </summary>
    /// <param name="state">Quantum state to measure</param>
    /// <param name="qubit">Qubit to measure</param>
    /// <returns>Tuple of (outcome, collapsed_state)</returns>
    public static (int outcome, PhotonicState collapsedState) MeasureYBasis(
        PhotonicState state, 
        int qubit)
    {
        // Transform to Y basis
        var sDagger = new SGate().Dagger();
        var hGate = new HadamardGate();

        var transformed = sDagger.Apply(state, new[] { qubit });
        transformed = hGate.Apply(transformed, new[] { qubit });

        // Measure in computational basis
        var (outcome, collapsed) = MeasureComputationalBasis(transformed, qubit);

        // Transform back
        var finalState = hGate.Apply(collapsed, new[] { qubit });
        finalState = new SGate().Apply(finalState, new[] { qubit });

        return (outcome, finalState);
    }

    /// <summary>
    /// Get measurement probabilities for all basis states.
    /// </summary>
    /// <param name="state">Quantum state</param>
    /// <returns>Array of probabilities</returns>
    public static double[] GetMeasurementProbabilities(PhotonicState state)
    {
        return state.Probabilities();
    }

    /// <summary>
    /// Compute expectation value of an operator.
    /// </summary>
    /// <param name="state">Quantum state</param>
    /// <param name="operatorMatrix">Operator matrix</param>
    /// <returns>Expectation value ⟨ψ|O|ψ⟩</returns>
    public static Complex ExpectationValue(PhotonicState state, Complex[,] operatorMatrix)
    {
        return state.ExpectationValue(operatorMatrix);
    }

    /// <summary>
    /// Sample from a discrete probability distribution.
    /// </summary>
    /// <param name="probabilities">Array of probabilities (must sum to 1)</param>
    /// <returns>Index of the sampled outcome</returns>
    private static int SampleFromDistribution(double[] probabilities)
    {
        double rand = _random.NextDouble();
        double cumulative = 0.0;

        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (rand < cumulative)
            {
                return i;
            }
        }

        // Fallback to last index (should not happen if probabilities sum to 1)
        return probabilities.Length - 1;
    }
}

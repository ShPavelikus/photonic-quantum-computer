using System.Numerics;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;

namespace PhotonicQuantumComputer;

/// <summary>
/// Represents a quantum state of photonic qubits.
/// The state is represented as a complex-valued vector in the computational basis.
/// For n qubits, the state vector has dimension 2^n.
/// </summary>
public class PhotonicState
{
    /// <summary>
    /// Complex vector representing the quantum state
    /// </summary>
    public Complex[] StateVector { get; private set; }

    /// <summary>
    /// Number of qubits in the system
    /// </summary>
    public int NumQubits { get; private set; }

    /// <summary>
    /// Initialize a photonic quantum state.
    /// </summary>
    /// <param name="stateVector">Complex vector representing the quantum state</param>
    /// <param name="normalize">Whether to normalize the state vector (default: true)</param>
    public PhotonicState(Complex[] stateVector, bool normalize = true)
    {
        StateVector = (Complex[])stateVector.Clone();

        // Determine number of qubits
        int size = StateVector.Length;
        NumQubits = (int)Math.Log2(size);

        if ((1 << NumQubits) != size)
        {
            throw new ArgumentException($"State vector size must be a power of 2, got {size}");
        }

        if (normalize)
        {
            Normalize();
        }
    }

    /// <summary>
    /// Normalize the state vector to unit length.
    /// </summary>
    public void Normalize()
    {
        double norm = Math.Sqrt(StateVector.Sum(c => c.Real * c.Real + c.Imaginary * c.Imaginary));
        if (norm > 1e-10)
        {
            for (int i = 0; i < StateVector.Length; i++)
            {
                StateVector[i] /= norm;
            }
        }
    }

    /// <summary>
    /// Create a copy of the state.
    /// </summary>
    /// <returns>A new PhotonicState with copied state vector</returns>
    public PhotonicState Copy()
    {
        return new PhotonicState(StateVector, normalize: false);
    }

    /// <summary>
    /// Create a |0...0⟩ state with n qubits.
    /// </summary>
    /// <param name="numQubits">Number of qubits</param>
    /// <returns>PhotonicState in |0...0⟩ state</returns>
    public static PhotonicState ZeroState(int numQubits)
    {
        int size = 1 << numQubits;
        var stateVector = new Complex[size];
        stateVector[0] = Complex.One;
        return new PhotonicState(stateVector, normalize: false);
    }

    /// <summary>
    /// Create a |1...1⟩ state with n qubits.
    /// </summary>
    /// <param name="numQubits">Number of qubits</param>
    /// <returns>PhotonicState in |1...1⟩ state</returns>
    public static PhotonicState OneState(int numQubits)
    {
        int size = 1 << numQubits;
        var stateVector = new Complex[size];
        stateVector[size - 1] = Complex.One;
        return new PhotonicState(stateVector, normalize: false);
    }

    /// <summary>
    /// Create a computational basis state |i⟩.
    /// </summary>
    /// <param name="numQubits">Number of qubits</param>
    /// <param name="basisIndex">Index of the basis state (0 to 2^n - 1)</param>
    /// <returns>PhotonicState in basis state |i⟩</returns>
    public static PhotonicState BasisState(int numQubits, int basisIndex)
    {
        int size = 1 << numQubits;
        if (basisIndex < 0 || basisIndex >= size)
        {
            throw new ArgumentException($"Basis index must be between 0 and {size - 1}");
        }

        var stateVector = new Complex[size];
        stateVector[basisIndex] = Complex.One;
        return new PhotonicState(stateVector, normalize: false);
    }

    /// <summary>
    /// Create an equal superposition state (|0⟩ + |1⟩)/√2 ⊗ ... ⊗ (|0⟩ + |1⟩)/√2.
    /// </summary>
    /// <param name="numQubits">Number of qubits</param>
    /// <returns>PhotonicState in equal superposition</returns>
    public static PhotonicState Superposition(int numQubits)
    {
        int size = 1 << numQubits;
        double amplitude = 1.0 / Math.Sqrt(size);
        var stateVector = new Complex[size];
        for (int i = 0; i < size; i++)
        {
            stateVector[i] = new Complex(amplitude, 0);
        }
        return new PhotonicState(stateVector, normalize: false);
    }

    /// <summary>
    /// Compute the density matrix ρ = |ψ⟩⟨ψ| for this pure state.
    /// </summary>
    /// <returns>Density matrix as a 2D complex array</returns>
    public Complex[,] DensityMatrix()
    {
        int size = StateVector.Length;
        var rho = new Complex[size, size];
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                rho[i, j] = StateVector[i] * Complex.Conjugate(StateVector[j]);
            }
        }
        return rho;
    }

    /// <summary>
    /// Check if the state is normalized.
    /// </summary>
    /// <param name="tolerance">Numerical tolerance for the norm check</param>
    /// <returns>True if ||ψ|| ≈ 1</returns>
    public bool IsNormalized(double tolerance = 1e-10)
    {
        double norm = Math.Sqrt(StateVector.Sum(c => c.Real * c.Real + c.Imaginary * c.Imaginary));
        return Math.Abs(norm - 1.0) < tolerance;
    }

    /// <summary>
    /// Compute the probability of measuring a specific basis state.
    /// </summary>
    /// <param name="basisIndex">Index of the basis state</param>
    /// <returns>Probability P(|i⟩) = |⟨i|ψ⟩|²</returns>
    public double Probability(int basisIndex)
    {
        if (basisIndex < 0 || basisIndex >= StateVector.Length)
        {
            throw new ArgumentException($"Basis index must be between 0 and {StateVector.Length - 1}");
        }

        var c = StateVector[basisIndex];
        return c.Real * c.Real + c.Imaginary * c.Imaginary;
    }

    /// <summary>
    /// Compute probabilities for all basis states.
    /// </summary>
    /// <returns>Array of probabilities for each basis state</returns>
    public double[] Probabilities()
    {
        return StateVector.Select(c => c.Real * c.Real + c.Imaginary * c.Imaginary).ToArray();
    }

    /// <summary>
    /// Compute inner product ⟨φ|ψ⟩ with another state.
    /// </summary>
    /// <param name="other">Another PhotonicState</param>
    /// <returns>Complex inner product</returns>
    public Complex InnerProduct(PhotonicState other)
    {
        if (NumQubits != other.NumQubits)
        {
            throw new ArgumentException("States must have the same number of qubits");
        }

        Complex sum = Complex.Zero;
        for (int i = 0; i < StateVector.Length; i++)
        {
            sum += Complex.Conjugate(other.StateVector[i]) * StateVector[i];
        }
        return sum;
    }

    /// <summary>
    /// Compute fidelity |⟨φ|ψ⟩|² with another state.
    /// </summary>
    /// <param name="other">Another PhotonicState</param>
    /// <returns>Fidelity value between 0 and 1</returns>
    public double Fidelity(PhotonicState other)
    {
        var ip = InnerProduct(other);
        return ip.Real * ip.Real + ip.Imaginary * ip.Imaginary;
    }

    /// <summary>
    /// Compute expectation value ⟨ψ|O|ψ⟩ of an operator.
    /// </summary>
    /// <param name="operatorMatrix">Operator matrix</param>
    /// <returns>Expectation value</returns>
    public Complex ExpectationValue(Complex[,] operatorMatrix)
    {
        int size = StateVector.Length;
        if (operatorMatrix.GetLength(0) != size || operatorMatrix.GetLength(1) != size)
        {
            throw new ArgumentException("Operator matrix dimensions must match state vector size");
        }

        // Compute O|ψ⟩
        var result = new Complex[size];
        for (int i = 0; i < size; i++)
        {
            result[i] = Complex.Zero;
            for (int j = 0; j < size; j++)
            {
                result[i] += operatorMatrix[i, j] * StateVector[j];
            }
        }

        // Compute ⟨ψ|O|ψ⟩
        Complex expectation = Complex.Zero;
        for (int i = 0; i < size; i++)
        {
            expectation += Complex.Conjugate(StateVector[i]) * result[i];
        }
        return expectation;
    }

    /// <summary>
    /// Compute partial trace over specified qubits.
    /// </summary>
    /// <param name="tracedQubits">List of qubit indices to trace out</param>
    /// <returns>Reduced density matrix</returns>
    public Complex[,] PartialTrace(List<int> tracedQubits)
    {
        // Get full density matrix
        var rho = DensityMatrix();

        // Determine kept qubits
        var keptQubits = Enumerable.Range(0, NumQubits)
            .Where(i => !tracedQubits.Contains(i))
            .ToList();

        if (keptQubits.Count == 0)
        {
            // All qubits traced out, return scalar
            Complex trace = Complex.Zero;
            for (int i = 0; i < rho.GetLength(0); i++)
            {
                trace += rho[i, i];
            }
            return new Complex[,] { { trace } };
        }

        // Dimensions
        int dimKept = 1 << keptQubits.Count;
        int dimTraced = 1 << tracedQubits.Count;

        // Initialize reduced density matrix
        var rhoReduced = new Complex[dimKept, dimKept];

        // Sum over traced qubits
        for (int i = 0; i < dimKept; i++)
        {
            for (int j = 0; j < dimKept; j++)
            {
                // Map reduced indices to full indices
                for (int k = 0; k < dimTraced; k++)
                {
                    // Build full indices
                    int idxI = 0;
                    int idxJ = 0;

                    // Insert kept qubit values
                    for (int pos = 0; pos < keptQubits.Count; pos++)
                    {
                        int qubit = keptQubits[pos];
                        int bitI = (i >> pos) & 1;
                        int bitJ = (j >> pos) & 1;
                        idxI |= (bitI << qubit);
                        idxJ |= (bitJ << qubit);
                    }

                    // Insert traced qubit values (same for both)
                    for (int pos = 0; pos < tracedQubits.Count; pos++)
                    {
                        int qubit = tracedQubits[pos];
                        int bit = (k >> pos) & 1;
                        idxI |= (bit << qubit);
                        idxJ |= (bit << qubit);
                    }

                    rhoReduced[i, j] += rho[idxI, idxJ];
                }
            }
        }

        return rhoReduced;
    }

    /// <summary>
    /// String representation of the state in Dirac notation.
    /// </summary>
    /// <returns>String representation</returns>
    public override string ToString()
    {
        var result = new List<string>();
        for (int i = 0; i < StateVector.Length; i++)
        {
            if (StateVector[i].Magnitude > 1e-10)
            {
                string basis = Convert.ToString(i, 2).PadLeft(NumQubits, '0');
                if (Math.Abs(StateVector[i].Imaginary) < 1e-10)
                {
                    result.Add($"{StateVector[i].Real:+0.####}|{basis}⟩");
                }
                else
                {
                    result.Add($"({StateVector[i].Real:+0.####}{StateVector[i].Imaginary:+0.####}i)|{basis}⟩");
                }
            }
        }
        return result.Count > 0 ? string.Join(" + ", result).Replace("+-", "-") : "0";
    }
}

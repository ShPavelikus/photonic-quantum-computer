using System.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;

namespace PhotonicQuantumComputer;

/// <summary>
/// Entanglement operations for photonic quantum computing.
/// Provides functions for creating and analyzing entangled states.
/// </summary>
public static class Entanglement
{
    /// <summary>
    /// Create a Bell state (maximally entangled two-qubit state).
    /// Bell states:
    /// - |Φ+⟩ = (|00⟩ + |11⟩)/√2
    /// - |Φ-⟩ = (|00⟩ - |11⟩)/√2
    /// - |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    /// - |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    /// </summary>
    /// <param name="bellType">Type of Bell state ("phi_plus", "phi_minus", "psi_plus", "psi_minus")</param>
    /// <returns>PhotonicState representing the Bell state</returns>
    public static PhotonicState CreateBellState(string bellType = "phi_plus")
    {
        double invSqrt2 = 1.0 / Math.Sqrt(2);
        Complex[] stateVector = bellType.ToLowerInvariant() switch
        {
            "phi_plus" => new Complex[] { invSqrt2, 0, 0, invSqrt2 },
            "phi_minus" => new Complex[] { invSqrt2, 0, 0, -invSqrt2 },
            "psi_plus" => new Complex[] { 0, invSqrt2, invSqrt2, 0 },
            "psi_minus" => new Complex[] { 0, invSqrt2, -invSqrt2, 0 },
            _ => throw new ArgumentException($"Unknown Bell state type: {bellType}")
        };

        return new PhotonicState(stateVector, normalize: false);
    }

    /// <summary>
    /// Create a GHZ (Greenberger-Horne-Zeilinger) state.
    /// |GHZ⟩ = (|0...0⟩ + |1...1⟩)/√2
    /// </summary>
    /// <param name="numQubits">Number of qubits</param>
    /// <returns>PhotonicState representing the GHZ state</returns>
    public static PhotonicState CreateGhzState(int numQubits)
    {
        if (numQubits < 2)
        {
            throw new ArgumentException("GHZ state requires at least 2 qubits");
        }

        int size = 1 << numQubits;
        var stateVector = new Complex[size];
        double invSqrt2 = 1.0 / Math.Sqrt(2);
        stateVector[0] = invSqrt2;
        stateVector[size - 1] = invSqrt2;

        return new PhotonicState(stateVector, normalize: false);
    }

    /// <summary>
    /// Compute the entanglement entropy (von Neumann entropy) of a subsystem.
    /// S(ρ_A) = -Tr(ρ_A log₂(ρ_A))
    /// </summary>
    /// <param name="state">Quantum state</param>
    /// <param name="subsystemQubits">List of qubit indices for subsystem A</param>
    /// <returns>Entanglement entropy in bits</returns>
    public static double EntanglementEntropy(PhotonicState state, List<int> subsystemQubits)
    {
        if (subsystemQubits.Count == 0)
        {
            return 0.0;
        }

        // Get all qubits not in subsystem
        var allQubits = Enumerable.Range(0, state.NumQubits).ToHashSet();
        var tracedQubits = allQubits.Except(subsystemQubits).ToList();

        if (tracedQubits.Count == 0)
        {
            // No qubits to trace out, state is pure
            return 0.0;
        }

        // Compute reduced density matrix
        var rhoReduced = state.PartialTrace(tracedQubits);

        // Convert to MathNet matrix for eigenvalue computation
        var matrixSize = rhoReduced.GetLength(0);
        var matrix = Matrix<Complex>.Build.Dense(matrixSize, matrixSize);
        for (int i = 0; i < matrixSize; i++)
        {
            for (int j = 0; j < matrixSize; j++)
            {
                matrix[i, j] = rhoReduced[i, j];
            }
        }

        // Compute eigenvalues
        var evd = matrix.Evd();
        var eigenvalues = evd.EigenValues;

        // Compute von Neumann entropy
        double entropy = 0.0;
        foreach (var eigenval in eigenvalues)
        {
            double eigenvalReal = eigenval.Real;
            if (eigenvalReal > 1e-10)
            {
                entropy -= eigenvalReal * Math.Log2(eigenvalReal);
            }
        }

        return entropy;
    }

    /// <summary>
    /// Compute Schmidt decomposition for bipartite system.
    /// </summary>
    /// <param name="state">Quantum state of n qubits</param>
    /// <param name="partition">Number of qubits in subsystem A (rest go to subsystem B)</param>
    /// <returns>Tuple of (schmidt_coefficients, basis_A, basis_B)</returns>
    public static (double[] coefficients, Complex[,] basisA, Complex[,] basisB) SchmidtDecomposition(
        PhotonicState state, 
        int partition)
    {
        if (partition <= 0 || partition >= state.NumQubits)
        {
            throw new ArgumentException("Invalid partition");
        }

        // Reshape state vector into matrix
        int dimA = 1 << partition;
        int dimB = 1 << (state.NumQubits - partition);

        var psiMatrix = Matrix<Complex>.Build.Dense(dimA, dimB);
        for (int i = 0; i < dimA; i++)
        {
            for (int j = 0; j < dimB; j++)
            {
                psiMatrix[i, j] = state.StateVector[i * dimB + j];
            }
        }

        // Perform SVD
        var svd = psiMatrix.Svd();
        var U = svd.U;
        var Vh = svd.VT;

        // Convert singular values to array - S is a Vector<Complex> in Complex SVD
        var sVector = svd.S;
        double[] schmidtCoeffs = new double[sVector.Count];
        for (int i = 0; i < sVector.Count; i++)
        {
            schmidtCoeffs[i] = sVector[i].Real;  // Take real part of singular values
        }

        // Convert U and V to 2D arrays
        var basisA = new Complex[U.RowCount, U.ColumnCount];
        for (int i = 0; i < U.RowCount; i++)
        {
            for (int j = 0; j < U.ColumnCount; j++)
            {
                basisA[i, j] = U[i, j];
            }
        }

        var basisB = new Complex[Vh.ColumnCount, Vh.RowCount];
        for (int i = 0; i < Vh.ColumnCount; i++)
        {
            for (int j = 0; j < Vh.RowCount; j++)
            {
                basisB[i, j] = Complex.Conjugate(Vh[j, i]);
            }
        }

        return (schmidtCoeffs, basisA, basisB);
    }

    /// <summary>
    /// Check if a two-qubit state is entangled using Schmidt decomposition.
    /// </summary>
    /// <param name="state">Two-qubit quantum state</param>
    /// <param name="tolerance">Numerical tolerance</param>
    /// <returns>True if state is entangled</returns>
    public static bool IsEntangled(PhotonicState state, double tolerance = 1e-10)
    {
        if (state.NumQubits != 2)
        {
            throw new ArgumentException("Entanglement check implemented for two-qubit states only");
        }

        var (schmidtCoeffs, _, _) = SchmidtDecomposition(state, 1);

        // State is entangled if more than one Schmidt coefficient is non-zero
        int nonZeroCoeffs = schmidtCoeffs.Count(c => c > tolerance);

        return nonZeroCoeffs > 1;
    }

    /// <summary>
    /// Compute concurrence for a two-qubit state (measure of entanglement).
    /// </summary>
    /// <param name="state">Two-qubit quantum state</param>
    /// <returns>Concurrence value (0 for separable, 1 for maximally entangled)</returns>
    public static double Concurrence(PhotonicState state)
    {
        if (state.NumQubits != 2)
        {
            throw new ArgumentException("Concurrence defined for two-qubit states only");
        }

        // Get density matrix
        var rho = state.DensityMatrix();

        // Spin-flip operator: σ_y ⊗ σ_y
        var sigmaY = new Complex[,]
        {
            { Complex.Zero, new Complex(0, -1) },
            { new Complex(0, 1), Complex.Zero }
        };
        var spinFlip = TensorProduct(sigmaY, sigmaY);

        // Compute R = ρ * (σ_y ⊗ σ_y) * ρ* * (σ_y ⊗ σ_y)
        var rhoConj = ConjugateMatrix(rho);
        var rhoTilde = MatrixMultiply(MatrixMultiply(spinFlip, rhoConj), spinFlip);
        var R = MatrixMultiply(rho, rhoTilde);

        // Convert to MathNet matrix for eigenvalue computation
        var matrix = Matrix<Complex>.Build.Dense(4, 4);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                matrix[i, j] = R[i, j];
            }
        }

        // Get eigenvalues
        var evd = matrix.Evd();
        var eigenvalues = evd.EigenValues
            .Select(e => Math.Sqrt(Math.Max(e.Real, 0)))
            .OrderByDescending(e => e)
            .ToArray();

        // Concurrence
        double C = Math.Max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3]);

        return C;
    }

    // Helper methods

    private static Complex[,] TensorProduct(Complex[,] a, Complex[,] b)
    {
        int aRows = a.GetLength(0);
        int aCols = a.GetLength(1);
        int bRows = b.GetLength(0);
        int bCols = b.GetLength(1);

        var result = new Complex[aRows * bRows, aCols * bCols];

        for (int i = 0; i < aRows; i++)
        {
            for (int j = 0; j < aCols; j++)
            {
                for (int k = 0; k < bRows; k++)
                {
                    for (int l = 0; l < bCols; l++)
                    {
                        result[i * bRows + k, j * bCols + l] = a[i, j] * b[k, l];
                    }
                }
            }
        }

        return result;
    }

    private static Complex[,] ConjugateMatrix(Complex[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        var result = new Complex[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = Complex.Conjugate(matrix[i, j]);
            }
        }

        return result;
    }

    private static Complex[,] MatrixMultiply(Complex[,] a, Complex[,] b)
    {
        int aRows = a.GetLength(0);
        int aCols = a.GetLength(1);
        int bCols = b.GetLength(1);

        var result = new Complex[aRows, bCols];
        for (int i = 0; i < aRows; i++)
        {
            for (int j = 0; j < bCols; j++)
            {
                Complex sum = Complex.Zero;
                for (int k = 0; k < aCols; k++)
                {
                    sum += a[i, k] * b[k, j];
                }
                result[i, j] = sum;
            }
        }
        return result;
    }
}

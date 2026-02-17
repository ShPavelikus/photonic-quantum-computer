using System.Numerics;

namespace PhotonicQuantumComputer;

/// <summary>
/// Interface for quantum gates.
/// </summary>
public interface IQuantumGate
{
    /// <summary>
    /// Unitary matrix representation of the gate
    /// </summary>
    Complex[,] Matrix { get; }

    /// <summary>
    /// Number of qubits the gate acts on
    /// </summary>
    int NumQubits { get; }

    /// <summary>
    /// Apply the gate to a quantum state.
    /// </summary>
    /// <param name="state">Input quantum state</param>
    /// <param name="targetQubits">List of target qubit indices</param>
    /// <returns>New quantum state after applying the gate</returns>
    PhotonicState Apply(PhotonicState state, int[]? targetQubits = null);

    /// <summary>
    /// Check if the gate is unitary (U†U = I).
    /// </summary>
    /// <param name="tolerance">Numerical tolerance</param>
    /// <returns>True if gate is unitary</returns>
    bool IsUnitary(double tolerance = 1e-10);

    /// <summary>
    /// Return the Hermitian conjugate (adjoint) of the gate.
    /// </summary>
    /// <returns>Adjoint gate U†</returns>
    IQuantumGate Dagger();
}

/// <summary>
/// Base class for quantum gates.
/// </summary>
public abstract class QuantumGateBase : IQuantumGate
{
    /// <summary>
    /// Unitary matrix representation of the gate
    /// </summary>
    public Complex[,] Matrix { get; protected set; }

    /// <summary>
    /// Number of qubits the gate acts on
    /// </summary>
    public int NumQubits { get; protected set; }

    protected QuantumGateBase(Complex[,] matrix)
    {
        Matrix = matrix;
        int size = matrix.GetLength(0);
        NumQubits = (int)Math.Log2(size);

        if ((1 << NumQubits) != size || matrix.GetLength(1) != size)
        {
            throw new ArgumentException($"Gate matrix must be square and size must be a power of 2, got {size}");
        }
    }

    /// <summary>
    /// Check if the gate is unitary (U†U = I).
    /// </summary>
    public bool IsUnitary(double tolerance = 1e-10)
    {
        int size = Matrix.GetLength(0);
        var product = MatrixMultiply(Matrix, ConjugateTranspose(Matrix));

        // Check if product is identity
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                Complex expected = i == j ? Complex.One : Complex.Zero;
                if ((product[i, j] - expected).Magnitude > tolerance)
                {
                    return false;
                }
            }
        }
        return true;
    }

    /// <summary>
    /// Apply the gate to a quantum state.
    /// </summary>
    public PhotonicState Apply(PhotonicState state, int[]? targetQubits = null)
    {
        if (targetQubits == null)
        {
            // Apply to all qubits
            if (NumQubits != state.NumQubits)
            {
                throw new ArgumentException($"Gate acts on {NumQubits} qubits but state has {state.NumQubits}");
            }
            var newVector = MatrixVectorMultiply(Matrix, state.StateVector);
            return new PhotonicState(newVector, normalize: false);
        }
        else
        {
            // Apply to specific qubits
            if (targetQubits.Length != NumQubits)
            {
                throw new ArgumentException($"Gate acts on {NumQubits} qubits but {targetQubits.Length} targets provided");
            }

            // Build full matrix with gate on target qubits
            var fullMatrix = EmbedGate(state.NumQubits, targetQubits);
            var newVector = MatrixVectorMultiply(fullMatrix, state.StateVector);
            return new PhotonicState(newVector, normalize: false);
        }
    }

    /// <summary>
    /// Embed gate matrix into larger Hilbert space.
    /// This implementation fixes Bug #1 - now works for ANY qubit pairs, including non-adjacent.
    /// </summary>
    /// <param name="totalQubits">Total number of qubits in the system</param>
    /// <param name="targetQubits">Indices of target qubits</param>
    /// <returns>Full matrix acting on all qubits</returns>
    protected Complex[,] EmbedGate(int totalQubits, int[] targetQubits)
    {
        if (targetQubits.Any(q => q >= totalQubits || q < 0))
        {
            throw new ArgumentException("Target qubit index out of range");
        }

        if (targetQubits.Length != NumQubits)
        {
            throw new ArgumentException($"Gate acts on {NumQubits} qubits but {targetQubits.Length} targets provided");
        }

        // For single-qubit gates
        if (NumQubits == 1)
        {
            int target = targetQubits[0];
            // Build tensor product: I ⊗ ... ⊗ I ⊗ U ⊗ I ⊗ ... ⊗ I
            Complex[,] result = new Complex[,] { { Complex.One } };
            for (int i = 0; i < totalQubits; i++)
            {
                if (i == target)
                {
                    result = TensorProduct(result, Matrix);
                }
                else
                {
                    result = TensorProduct(result, Identity2x2());
                }
            }
            return result;
        }

        // For multi-qubit gates - FIXED: Now supports non-adjacent qubits
        // Use direct index mapping approach
        int fullSize = 1 << totalQubits;
        var fullMatrix = new Complex[fullSize, fullSize];

        // For each basis state |i⟩, compute the output state
        for (int i = 0; i < fullSize; i++)
        {
            // Extract target qubit bits
            int gateInput = 0;
            for (int k = 0; k < NumQubits; k++)
            {
                int targetQubit = targetQubits[k];
                int bit = (i >> targetQubit) & 1;
                gateInput |= (bit << k);
            }

            // Apply gate matrix to these bits
            for (int gateOutput = 0; gateOutput < (1 << NumQubits); gateOutput++)
            {
                if (Matrix[gateOutput, gateInput].Magnitude < 1e-15)
                    continue;

                // Construct output basis state
                int j = i;
                for (int k = 0; k < NumQubits; k++)
                {
                    int targetQubit = targetQubits[k];
                    int newBit = (gateOutput >> k) & 1;
                    // Clear the bit at targetQubit position
                    j &= ~(1 << targetQubit);
                    // Set the new bit value
                    j |= (newBit << targetQubit);
                }

                fullMatrix[j, i] += Matrix[gateOutput, gateInput];
            }
        }

        return fullMatrix;
    }

    /// <summary>
    /// Return the Hermitian conjugate (adjoint) of the gate.
    /// </summary>
    public virtual IQuantumGate Dagger()
    {
        return new GenericGate(ConjugateTranspose(Matrix));
    }

    // Helper methods for matrix operations

    private static Complex[,] ConjugateTranspose(Complex[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        var result = new Complex[cols, rows];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[j, i] = Complex.Conjugate(matrix[i, j]);
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

    private static Complex[] MatrixVectorMultiply(Complex[,] matrix, Complex[] vector)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        if (cols != vector.Length)
        {
            throw new ArgumentException("Matrix and vector dimensions don't match");
        }

        var result = new Complex[rows];
        for (int i = 0; i < rows; i++)
        {
            Complex sum = Complex.Zero;
            for (int j = 0; j < cols; j++)
            {
                sum += matrix[i, j] * vector[j];
            }
            result[i] = sum;
        }
        return result;
    }

    /// <summary>
    /// Compute tensor product (Kronecker product) of two matrices.
    /// </summary>
    public static Complex[,] TensorProduct(Complex[,] a, Complex[,] b)
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

    private static Complex[,] Identity2x2()
    {
        return new Complex[,]
        {
            { Complex.One, Complex.Zero },
            { Complex.Zero, Complex.One }
        };
    }

    public override string ToString()
    {
        return $"{GetType().Name}({NumQubits}-qubit)";
    }
}

// Generic gate wrapper
internal class GenericGate : QuantumGateBase
{
    public GenericGate(Complex[,] matrix) : base(matrix) { }
}

// Single-qubit gates

/// <summary>
/// Hadamard gate: Creates superposition.
/// Implemented with 50/50 beam splitter in photonics.
/// Matrix: (1/√2) [[1,  1], [1, -1]]
/// </summary>
public class HadamardGate : QuantumGateBase
{
    public HadamardGate() : base(CreateMatrix()) { }

    private static Complex[,] CreateMatrix()
    {
        double invSqrt2 = 1.0 / Math.Sqrt(2);
        return new Complex[,]
        {
            { new Complex(invSqrt2, 0), new Complex(invSqrt2, 0) },
            { new Complex(invSqrt2, 0), new Complex(-invSqrt2, 0) }
        };
    }
}

/// <summary>
/// Pauli-X gate (NOT gate): Bit flip.
/// Matrix: [[0, 1], [1, 0]]
/// </summary>
public class PauliXGate : QuantumGateBase
{
    public PauliXGate() : base(new Complex[,]
    {
        { Complex.Zero, Complex.One },
        { Complex.One, Complex.Zero }
    })
    { }
}

/// <summary>
/// Pauli-Y gate: Bit and phase flip.
/// Matrix: [[0, -i], [i,  0]]
/// </summary>
public class PauliYGate : QuantumGateBase
{
    public PauliYGate() : base(new Complex[,]
    {
        { Complex.Zero, new Complex(0, -1) },
        { new Complex(0, 1), Complex.Zero }
    })
    { }
}

/// <summary>
/// Pauli-Z gate: Phase flip.
/// Matrix: [[1,  0], [0, -1]]
/// </summary>
public class PauliZGate : QuantumGateBase
{
    public PauliZGate() : base(new Complex[,]
    {
        { Complex.One, Complex.Zero },
        { Complex.Zero, new Complex(-1, 0) }
    })
    { }
}

/// <summary>
/// Phase gate: Applies phase rotation.
/// Matrix: [[1, 0], [0, e^(iφ)]]
/// </summary>
public class PhaseGate : QuantumGateBase
{
    /// <summary>
    /// Phase angle φ in radians
    /// </summary>
    public double Phase { get; }

    public PhaseGate(double phase) : base(CreateMatrix(phase))
    {
        Phase = phase;
    }

    private static Complex[,] CreateMatrix(double phase)
    {
        return new Complex[,]
        {
            { Complex.One, Complex.Zero },
            { Complex.Zero, Complex.FromPolarCoordinates(1.0, phase) }
        };
    }
}

/// <summary>
/// S gate (Phase gate with φ = π/2).
/// Matrix: [[1, 0], [0, i]]
/// </summary>
public class SGate : QuantumGateBase
{
    public SGate() : base(new Complex[,]
    {
        { Complex.One, Complex.Zero },
        { Complex.Zero, Complex.ImaginaryOne }
    })
    { }
}

/// <summary>
/// T gate (Phase gate with φ = π/4).
/// Matrix: [[1, 0], [0, e^(iπ/4)]]
/// </summary>
public class TGate : QuantumGateBase
{
    public TGate() : base(CreateMatrix()) { }

    private static Complex[,] CreateMatrix()
    {
        return new Complex[,]
        {
            { Complex.One, Complex.Zero },
            { Complex.Zero, Complex.FromPolarCoordinates(1.0, Math.PI / 4) }
        };
    }
}

/// <summary>
/// General rotation gate around axis on Bloch sphere.
/// R(θ, φ, λ) is the general single-qubit rotation.
/// </summary>
public class RotationGate : QuantumGateBase
{
    public double Theta { get; }
    public double Phi { get; }
    public double Lambda { get; }

    public RotationGate(double theta, double phi, double lambda) : base(CreateMatrix(theta, phi, lambda))
    {
        Theta = theta;
        Phi = phi;
        Lambda = lambda;
    }

    private static Complex[,] CreateMatrix(double theta, double phi, double lambda)
    {
        double cos = Math.Cos(theta / 2);
        double sin = Math.Sin(theta / 2);

        return new Complex[,]
        {
            { new Complex(cos, 0), -Complex.FromPolarCoordinates(1.0, lambda) * sin },
            { Complex.FromPolarCoordinates(1.0, phi) * sin, Complex.FromPolarCoordinates(1.0, phi + lambda) * cos }
        };
    }
}

// Two-qubit gates

/// <summary>
/// CNOT (Controlled-NOT) gate.
/// Implemented through Hong-Ou-Mandel interference in photonics.
/// Matrix: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
/// </summary>
public class CnotGate : QuantumGateBase
{
    public CnotGate() : base(new Complex[,]
    {
        { Complex.One, Complex.Zero, Complex.Zero, Complex.Zero },
        { Complex.Zero, Complex.One, Complex.Zero, Complex.Zero },
        { Complex.Zero, Complex.Zero, Complex.Zero, Complex.One },
        { Complex.Zero, Complex.Zero, Complex.One, Complex.Zero }
    })
    { }
}

/// <summary>
/// Controlled-Z gate.
/// Matrix: [[1, 0, 0,  0], [0, 1, 0,  0], [0, 0, 1,  0], [0, 0, 0, -1]]
/// </summary>
public class CzGate : QuantumGateBase
{
    public CzGate() : base(new Complex[,]
    {
        { Complex.One, Complex.Zero, Complex.Zero, Complex.Zero },
        { Complex.Zero, Complex.One, Complex.Zero, Complex.Zero },
        { Complex.Zero, Complex.Zero, Complex.One, Complex.Zero },
        { Complex.Zero, Complex.Zero, Complex.Zero, new Complex(-1, 0) }
    })
    { }
}

/// <summary>
/// SWAP gate: Exchanges two qubits.
/// Matrix: [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
/// </summary>
public class SwapGate : QuantumGateBase
{
    public SwapGate() : base(new Complex[,]
    {
        { Complex.One, Complex.Zero, Complex.Zero, Complex.Zero },
        { Complex.Zero, Complex.Zero, Complex.One, Complex.Zero },
        { Complex.Zero, Complex.One, Complex.Zero, Complex.Zero },
        { Complex.Zero, Complex.Zero, Complex.Zero, Complex.One }
    })
    { }
}

/// <summary>
/// Utility class for gate operations.
/// </summary>
public static class QuantumGates
{
    /// <summary>
    /// Compute tensor product of multiple gates.
    /// </summary>
    /// <param name="gates">Variable number of quantum gates</param>
    /// <returns>Tensor product gate</returns>
    public static IQuantumGate TensorProduct(params IQuantumGate[] gates)
    {
        if (gates.Length == 0)
        {
            throw new ArgumentException("At least one gate required");
        }

        Complex[,] result = gates[0].Matrix;
        for (int i = 1; i < gates.Length; i++)
        {
            result = QuantumGateBase.TensorProduct(result, gates[i].Matrix);
        }

        return new GenericGate(result);
    }
}

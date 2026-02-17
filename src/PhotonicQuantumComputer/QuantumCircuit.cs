using System.Text;

namespace PhotonicQuantumComputer;

/// <summary>
/// Quantum circuit for composing and executing quantum operations.
/// </summary>
public class QuantumCircuit
{
    /// <summary>
    /// Number of qubits in the circuit
    /// </summary>
    public int NumQubits { get; }

    /// <summary>
    /// List of operations (gate, target qubits)
    /// </summary>
    private readonly List<(IQuantumGate gate, int[] targets)> _operations;

    /// <summary>
    /// List of qubits to measure
    /// </summary>
    private readonly List<int> _measurements;

    /// <summary>
    /// Final state after execution (if no measurements)
    /// </summary>
    public PhotonicState? State { get; private set; }

    /// <summary>
    /// Initialize a quantum circuit.
    /// </summary>
    /// <param name="numQubits">Number of qubits in the circuit</param>
    public QuantumCircuit(int numQubits)
    {
        if (numQubits <= 0)
        {
            throw new ArgumentException("Number of qubits must be positive");
        }

        NumQubits = numQubits;
        _operations = new List<(IQuantumGate, int[])>();
        _measurements = new List<int>();
        State = null;
    }

    /// <summary>
    /// Add a gate to the circuit.
    /// </summary>
    /// <param name="gate">Quantum gate to add</param>
    /// <param name="targetQubits">List of target qubit indices</param>
    public QuantumCircuit AddGate(IQuantumGate gate, params int[] targetQubits)
    {
        if (targetQubits.Length == 0)
        {
            throw new ArgumentException("Must specify target qubits");
        }

        if (targetQubits.Max() >= NumQubits || targetQubits.Min() < 0)
        {
            throw new ArgumentException("Target qubit index out of range");
        }

        if (targetQubits.Length != gate.NumQubits)
        {
            throw new ArgumentException($"Gate requires {gate.NumQubits} qubits but {targetQubits.Length} provided");
        }

        _operations.Add((gate, targetQubits));
        return this;  // Fluent API
    }

    /// <summary>
    /// Add Hadamard gate.
    /// </summary>
    public QuantumCircuit H(int qubit)
    {
        return AddGate(new HadamardGate(), qubit);
    }

    /// <summary>
    /// Add Pauli-X gate.
    /// </summary>
    public QuantumCircuit X(int qubit)
    {
        return AddGate(new PauliXGate(), qubit);
    }

    /// <summary>
    /// Add Pauli-Y gate.
    /// </summary>
    public QuantumCircuit Y(int qubit)
    {
        return AddGate(new PauliYGate(), qubit);
    }

    /// <summary>
    /// Add Pauli-Z gate.
    /// </summary>
    public QuantumCircuit Z(int qubit)
    {
        return AddGate(new PauliZGate(), qubit);
    }

    /// <summary>
    /// Add S gate.
    /// </summary>
    public QuantumCircuit S(int qubit)
    {
        return AddGate(new SGate(), qubit);
    }

    /// <summary>
    /// Add T gate.
    /// </summary>
    public QuantumCircuit T(int qubit)
    {
        return AddGate(new TGate(), qubit);
    }

    /// <summary>
    /// Add phase gate.
    /// </summary>
    public QuantumCircuit Phase(int qubit, double angle)
    {
        return AddGate(new PhaseGate(angle), qubit);
    }

    /// <summary>
    /// Add CNOT gate.
    /// </summary>
    public QuantumCircuit Cnot(int control, int target)
    {
        return AddGate(new CnotGate(), control, target);
    }

    /// <summary>
    /// Add CZ gate.
    /// </summary>
    public QuantumCircuit Cz(int control, int target)
    {
        return AddGate(new CzGate(), control, target);
    }

    /// <summary>
    /// Add SWAP gate.
    /// </summary>
    public QuantumCircuit Swap(int qubit1, int qubit2)
    {
        return AddGate(new SwapGate(), qubit1, qubit2);
    }

    /// <summary>
    /// Add measurement operation.
    /// </summary>
    public QuantumCircuit Measure(int qubit)
    {
        if (qubit < 0 || qubit >= NumQubits)
        {
            throw new ArgumentException("Qubit index out of range");
        }
        _measurements.Add(qubit);
        return this;
    }

    /// <summary>
    /// Add measurement of all qubits.
    /// </summary>
    public QuantumCircuit MeasureAll()
    {
        _measurements.Clear();
        _measurements.AddRange(Enumerable.Range(0, NumQubits));
        return this;
    }

    /// <summary>
    /// Execute the circuit.
    /// </summary>
    /// <param name="initialState">Initial quantum state (default: |0...0⟩)</param>
    /// <param name="shots">Number of times to run the circuit</param>
    /// <returns>Dictionary with measurement results</returns>
    public Dictionary<string, int> Run(PhotonicState? initialState = null, int shots = 1)
    {
        initialState ??= PhotonicState.ZeroState(NumQubits);

        if (initialState.NumQubits != NumQubits)
        {
            throw new ArgumentException("Initial state has wrong number of qubits");
        }

        var results = new Dictionary<string, int>();

        for (int shot = 0; shot < shots; shot++)
        {
            // Start with initial state
            var state = initialState.Copy();

            // Apply all gates
            foreach (var (gate, targets) in _operations)
            {
                state = gate.Apply(state, targets);
            }

            // Perform measurements
            if (_measurements.Count > 0)
            {
                var measurementResult = new List<string>();
                var sortedMeasurements = _measurements.OrderBy(q => q).ToList();

                foreach (var qubit in sortedMeasurements)
                {
                    var (outcome, collapsedState) = Measurement.MeasureComputationalBasis(state, qubit);
                    state = collapsedState;
                    measurementResult.Add(outcome.ToString());
                }

                string resultStr = string.Join("", measurementResult);
                if (results.ContainsKey(resultStr))
                {
                    results[resultStr]++;
                }
                else
                {
                    results[resultStr] = 1;
                }
            }
            else
            {
                // No measurements, store final state
                State = state;
            }
        }

        return results;
    }

    /// <summary>
    /// Get the final state vector without measurement.
    /// </summary>
    /// <param name="initialState">Initial quantum state (default: |0...0⟩)</param>
    /// <returns>Final quantum state</returns>
    public PhotonicState GetStatevector(PhotonicState? initialState = null)
    {
        initialState ??= PhotonicState.ZeroState(NumQubits);

        if (initialState.NumQubits != NumQubits)
        {
            throw new ArgumentException("Initial state has wrong number of qubits");
        }

        var state = initialState.Copy();

        // Apply all gates
        foreach (var (gate, targets) in _operations)
        {
            state = gate.Apply(state, targets);
        }

        return state;
    }

    /// <summary>
    /// Generate ASCII art representation of the circuit.
    /// </summary>
    /// <returns>String representation of the circuit</returns>
    public string Draw()
    {
        var lines = new string[NumQubits];
        for (int i = 0; i < NumQubits; i++)
        {
            lines[i] = $"q{i}: ";
        }

        // Simple visualization
        foreach (var (gate, targets) in _operations)
        {
            string gateName = gate.GetType().Name.Replace("Gate", "");

            for (int i = 0; i < NumQubits; i++)
            {
                if (targets.Contains(i))
                {
                    lines[i] += $"─[{gateName}]─";
                }
                else
                {
                    lines[i] += new string('─', gateName.Length + 4);
                }
            }
        }

        // Add measurements
        if (_measurements.Count > 0)
        {
            for (int i = 0; i < NumQubits; i++)
            {
                if (_measurements.Contains(i))
                {
                    lines[i] += "─[M]";
                }
                else
                {
                    lines[i] += "────";
                }
            }
        }

        return string.Join("\n", lines);
    }

    /// <summary>
    /// String representation of circuit.
    /// </summary>
    public override string ToString()
    {
        return $"QuantumCircuit({NumQubits} qubits, {_operations.Count} operations)";
    }
}

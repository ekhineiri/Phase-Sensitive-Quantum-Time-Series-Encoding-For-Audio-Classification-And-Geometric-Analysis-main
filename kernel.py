import importlib
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from feature_maps import build_feature_map


def fidelity_circuit(x_a, x_b, feature_map):
    qc_a = build_feature_map(feature_map=feature_map, data=x_a)
    qc_b = build_feature_map(feature_map=feature_map, data=x_b)

    num_qubits = qc_a.num_qubits

    qc = QuantumCircuit(num_qubits)
    qc.compose(qc_a, inplace=True)
    qc.compose(qc_b.inverse(), inplace=True)
    #qc.measure(range(num_qubits), range(num_qubits))
    return qc

def get_fidelity(x_a, x_b, feature_map, backend="statevector", shots=1024, seed=42):
    qc = fidelity_circuit(x_a=x_a, x_b=x_b, feature_map=feature_map)
    if backend.lower() == "statevector":
        state = Statevector.from_instruction(qc)
        return np.abs(state.data[0]) ** 2
    elif backend.lower() == "aer":
        aer = importlib.import_module("qiskit_aer")
        simulator = aer.AerSimulator(seed_simulator=seed)
        compiled = transpile(qc, simulator)
        result = simulator.run(compiled, shots=shots).result()
        counts = result.get_counts()
        zero_key = "0" * qc.num_qubits
        return float(counts.get(zero_key, 0) / shots)
    else:
        raise ValueError("Unsupported backend. Choose 'statevector' or 'aer'.")



def get_kernel_matrix(X1, feature_map, X2=None, backend="statevector", shots=1024):
    # Caso 1: un solo dataset -> matriz simétrica
    if X2 is None:
        n = len(X1)
        K = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                value = get_fidelity(
                    x_a=X1[i],
                    x_b=X1[j],
                    feature_map=feature_map,
                    backend=backend,
                    shots=shots
                )

                K[i, j] = value
                K[j, i] = value

    # Caso 2: dos datasets -> matriz rectangular
    else:
        n1 = len(X1)
        n2 = len(X2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = get_fidelity(
                    x_a=X1[i],
                    x_b=X2[j],
                    feature_map=feature_map,
                    backend=backend,
                    shots=shots
                )

    return K



# version rápida?

def statevectors(X, feature_map):
    states = []

    for x in X:
        qc = build_feature_map(feature_map, x)
        state = Statevector.from_instruction(qc)
        states.append(state.data)

    return np.array(states)   # shape: (N, 2^n)


def fidelity_kernel_matrix(states_X, states_Y=None):
    if states_Y is None:
        states_Y = states_X

    n1 = len(states_X)
    n2 = len(states_Y)

    K = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            K[i, j] = np.abs(np.vdot(states_X[i], states_Y[j])) ** 2

    return K

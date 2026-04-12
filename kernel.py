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


def _build_kernel_matrix_statevector(
    X_a,
    X_b,
    feature_map,
    entanglement,
):
    A = np.asarray(X_a, dtype=float)
    B = np.asarray(X_b, dtype=float)

    states_a = []
    for sample in A:
        circuit = build_feature_map(feature_map=feature_map, data=sample, entanglement=entanglement)
        states_a.append(Statevector.from_instruction(circuit).data)

    states_b = []
    for sample in B:
        circuit = build_feature_map(feature_map=feature_map, data=sample, entanglement=entanglement)
        states_b.append(Statevector.from_instruction(circuit).data)

    overlaps = np.asarray(states_a) @ np.conjugate(np.asarray(states_b)).T
    return np.abs(overlaps) ** 2


def _build_kernel_matrix_aer(
    X_a,
    X_b,
    feature_map,
    entanglement,
    shots,
    seed,
):
    aer = importlib.import_module("qiskit_aer")
    simulator = aer.AerSimulator(seed_simulator=seed)
    A = np.asarray(X_a, dtype=float)
    B = np.asarray(X_b, dtype=float)
    K = np.zeros((len(A), len(B)), dtype=np.float64)

    for i in range(len(A)):
        for j in range(len(B)):
            K[i, j] = _estimate_fidelity_with_shots(
                x_a=A[i],
                x_b=B[j],
                feature_map=feature_map,
                entanglement=entanglement,
                simulator=simulator,
                shots=shots,
            )
    return K


def build_kernel_matrix(
    X_a,
    X_b,
    feature_map,
    entanglement,
    backend="aer",
    shots=1024,
    seed=42,
):
    backend_value = (backend or "").lower()
    if backend_value == "statevector":
        return _build_kernel_matrix_statevector(
            X_a=X_a,
            X_b=X_b,
            feature_map=feature_map,
            entanglement=entanglement,
        )

    if backend_value == "aer":
        return _build_kernel_matrix_aer(
            X_a=X_a,
            X_b=X_b,
            feature_map=feature_map,
            entanglement=entanglement,
            shots=shots,
            seed=seed,
        )

    raise ValueError("Unsupported backend. Choose 'statevector' or 'aer'.")

def build_qtse_kernel_matrix(A_audio, A_t, B_audio=None, B_t=None, n_qubits=8):
    """
    A_audio: array/list de shape (N,16) con valores 0..15
    A_t:     array/list de shape (N,16) con tiempos 0..15
    B_*: si None, usa A (kernel cuadrado train-train)
    """
    if B_audio is None:
        B_audio, B_t = A_audio, A_t

    states_a = []
    for audio, t in zip(A_audio, A_t):
        psi = Statevector.from_instruction(qtse(n_qubits=n_qubits, audio=np.asarray(audio), t=np.asarray(t))).data
        states_a.append(psi)

    states_b = []
    for audio, t in zip(B_audio, B_t):
        psi = Statevector.from_instruction(qtse(n_qubits=n_qubits, audio=np.asarray(audio), t=np.asarray(t))).data
        states_b.append(psi)

    A = np.asarray(states_a)
    B = np.asarray(states_b)
    overlaps = A @ np.conjugate(B).T
    return np.abs(overlaps) ** 2


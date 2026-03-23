from __future__ import annotations

import importlib
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from feature_maps import build_feature_map
import numpy as np
from qiskit.quantum_info import Statevector
from feature_maps import qtse


def _fidelity_circuit(x_a: np.ndarray, x_b: np.ndarray, feature_map: str, entanglement: str) -> QuantumCircuit:
    circ_a = build_feature_map(feature_map=feature_map, data=x_a, entanglement=entanglement)
    circ_b = build_feature_map(feature_map=feature_map, data=x_b, entanglement=entanglement)

    n_qubits = circ_a.num_qubits
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.compose(circ_a, inplace=True)
    qc.compose(circ_b.inverse(), inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def _estimate_fidelity_with_shots(
    x_a: np.ndarray,
    x_b: np.ndarray,
    feature_map: str,
    entanglement: str,
    simulator,
    shots: int,
) -> float:
    qc = _fidelity_circuit(x_a=x_a, x_b=x_b, feature_map=feature_map, entanglement=entanglement)
    compiled = transpile(qc, simulator)
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()
    zero_key = "0" * qc.num_qubits
    return float(counts.get(zero_key, 0) / shots)


def _build_kernel_matrix_statevector(
    X_a: np.ndarray,
    X_b: np.ndarray,
    feature_map: str,
    entanglement: str,
) -> np.ndarray:
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
    X_a: np.ndarray,
    X_b: np.ndarray,
    feature_map: str,
    entanglement: str,
    shots: int,
    seed: int | None,
) -> np.ndarray:
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
    X_a: np.ndarray,
    X_b: np.ndarray,
    feature_map: str,
    entanglement: str,
    backend: str = "aer",
    shots: int = 1024,
    seed: int | None = 42,
) -> np.ndarray:
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


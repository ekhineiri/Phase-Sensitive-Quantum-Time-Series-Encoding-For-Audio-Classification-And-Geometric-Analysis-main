from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


LAST_FOUR_SUPERPOSITION_QUBITS = 8


def _validate_data(n_qubits: int, data: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(data, dtype=float).ravel()
    if arr.size != n_qubits:
        raise ValueError(f"data must have length equal to n_qubits ({n_qubits}); got {arr.size}")
    return arr


def _apply_entanglement(qc: QuantumCircuit, entanglement: str, gate: str) -> None:
    ent = (entanglement or "").lower()
    if ent == "chain":
        pairs = [(i, i + 1) for i in range(qc.num_qubits - 1)]
    elif ent == "ring":
        pairs = [(i, i + 1) for i in range(qc.num_qubits - 1)]
        if qc.num_qubits > 1:
            pairs.append((qc.num_qubits - 1, 0))
    elif ent == "all_to_all":
        pairs = [(i, j) for i in range(qc.num_qubits) for j in range(i + 1, qc.num_qubits)]
    else:
        raise ValueError("Unsupported entanglement pattern. Choose 'chain', 'ring' or 'all_to_all'.")

    for control, target in pairs:
        if gate == "cx":
            qc.cx(control, target)
        elif gate == "cz":
            qc.cz(control, target)
        else:
            raise ValueError("Unsupported entangling gate. Choose 'cx' or 'cz'.")


def ry_feature_map(n_qubits: int, data: np.ndarray | list[float], entanglement: str = "ring") -> QuantumCircuit:
    arr = _validate_data(n_qubits, data)
    qc = QuantumCircuit(n_qubits)

    for qubit in range(n_qubits):
        qc.ry(arr[qubit], qubit)

    _apply_entanglement(qc, entanglement=entanglement, gate="cx")

    for qubit in range(n_qubits):
        qc.ry(arr[qubit], qubit)

    return qc


def phase_feature_map(n_qubits: int, data: np.ndarray | list[float], entanglement: str = "ring") -> QuantumCircuit:
    arr = _validate_data(n_qubits, data)
    qc = QuantumCircuit(n_qubits)

    for qubit in range(n_qubits):
        qc.h(qubit)

    for qubit in range(n_qubits):
        qc.rz(arr[qubit], qubit)

    _apply_entanglement(qc, entanglement=entanglement, gate="cz")

    for qubit in range(n_qubits):
        qc.rz(arr[qubit], qubit)

    return qc


def qtse(n_qubits: int, audio: np.ndarray, t: np.ndarray | list[float]) -> QuantumCircuit:
    # QTSE: 4 qubits de amplitud A (0..3) + 4 qubits de tiempo T (4..7).
    # Para cada valor de tiempo t_i, se aplican MCX que escriben A_i condicionado al estado |t_i>.
    if n_qubits < 8:
        raise ValueError("qtse requiere al menos 8 qubits: 4 para A y 4 para T.")

    audio_arr = np.asarray(audio, dtype=int).ravel()
    t_arr = np.asarray(t, dtype=int).ravel()
    if audio_arr.size != t_arr.size:
        raise ValueError(f"audio y t deben tener la misma longitud; got {audio_arr.size} y {t_arr.size}.")
    if audio_arr.size == 0:
        raise ValueError("audio y t no pueden estar vacios.")

    if np.any((audio_arr < 0) | (audio_arr > 15)):
        raise ValueError("Los valores de audio deben estar cuantizados en [0, 15].")
    if np.any((t_arr < 0) | (t_arr > 15)):
        raise ValueError("Los valores de tiempo deben estar en [0, 15].")

    qc = QuantumCircuit(n_qubits)
    a_qubits = [0, 1, 2, 3]
    t_qubits = [4, 5, 6, 7]

    # Superposicion uniforme de los 16 tiempos.
    for qubit in t_qubits:
        qc.h(qubit)

    # Escribir A condicionado por cada estado base de T (qROM con MCX).
    for t_value, a_value in zip(t_arr, audio_arr):
        t_bits = np.binary_repr(int(t_value), width=4)
        a_bits = np.binary_repr(int(a_value), width=4)

        # Convertir controles sobre |0> en controles efectivos via X ... MCX ... X.
        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

        # Si un bit de A debe ser 1 para este t, aplicar MCX hacia ese target de A.
        for a_idx, abit in enumerate(a_bits):
            if abit == "1":
                qc.mcx(t_qubits, a_qubits[a_idx])

        # Deshacer X en los controles abiertos.
        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

    return qc


def build_feature_map(feature_map: str, data: np.ndarray | list[float], entanglement: str = "ring") -> QuantumCircuit:
    arr = np.asarray(data, dtype=float).ravel()
    n_qubits = arr.size
    fm = feature_map.lower()

    if fm == "ry":
        return ry_feature_map(n_qubits=n_qubits, data=arr, entanglement=entanglement)
    if fm in {"rz", "phase"}:
        return phase_feature_map(n_qubits=n_qubits, data=arr, entanglement=entanglement)
    if fm in {"qtse", "ry_h_last4", "ry_hadamard_last4"}:
        raise ValueError("QTSE requiere audio y t por separado. Usa qtse(n_qubits=8, audio=..., t=...).")

    raise ValueError("Unsupported feature map. Choose 'ry', 'rz', or 'qtse'.")

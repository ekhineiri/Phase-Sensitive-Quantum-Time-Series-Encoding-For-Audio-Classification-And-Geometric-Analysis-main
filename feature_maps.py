import numpy as np
from qiskit import QuantumCircuit
import os

def ry_feature_map(n_qubits, data):
    data = np.asarray(data)
    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        qc.ry(data[i], i)

    # ring entanglement
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.cx(n_qubits - 1, 0)

    # re-encoding
    for i in range(n_qubits):
        qc.ry(data[i], i)

    return qc


def rz_feature_map(n_qubits, data):
    data = np.asarray(data)
    qc = QuantumCircuit(n_qubits)

    # encoding
    for i in range(n_qubits):
        qc.rz(data[i], i)

    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.cx(n_qubits - 1, 0)

    # re-encoding
    for i in range(n_qubits):
        qc.rz(data[i], i)

    return qc


def draw_circuit(qc, save_path="circuits/circuit.png"):

    # crear carpeta si no existe
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # dibujar circuito
    fig = qc.draw(output="mpl")

    # guardar
    fig.savefig(save_path)

    print(f"Circuit saved to {save_path}")


def build_feature_map(feature_map, data):
    n_qubits = len(data)

    if(feature_map == "ry"):
        return ry_feature_map(n_qubits=n_qubits, data=data)
    if(feature_map in {"rz", "phase"}):
        return rz_feature_map(n_qubits=n_qubits, data=data)
    if(feature_map == "qtse"):
        return qtse(data=data)
    
def qtse(data):
    data_arr = np.asarray(data, dtype=int).ravel()

    n_qubits = 8
    qc = QuantumCircuit(n_qubits)
    a_qubits = [0, 1, 2, 3]
    t_qubits = [4, 5, 6, 7]

    for qubit in t_qubits:
        qc.h(qubit)

    for t_value, a_value in enumerate(data_arr):
        t_bits = np.binary_repr(int(t_value), width=4)
        a_bits = np.binary_repr(int(a_value), width=4)

        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

        for a_idx, abit in enumerate(a_bits):
            if abit == "1":
                qc.mcx(t_qubits, a_qubits[a_idx])

        for bit_idx, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[bit_idx])

    return qc


def qtse2(n_qubits, audio, t):
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


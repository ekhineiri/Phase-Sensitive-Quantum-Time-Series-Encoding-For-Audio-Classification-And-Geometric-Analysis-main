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


def qtse_timbre_phase1(data_a, data_p):
    am_arr = np.asarray(data_a, dtype=int).ravel()
    ph_arr = np.asarray(data_p, dtype=int).ravel()

    n_qubits = 12
    qc = QuantumCircuit(n_qubits)

    a_qubits = [0,1,2,3]
    p_qubits = [4,5,6,7]
    t_qubits = [8,9,10,11]

    for t_value, (a_value, p_value) in enumerate(zip(am_arr, ph_arr)):
        t_bits = np.binary_repr(int(t_value), width=4)
        a_bits = np.binary_repr(int(a_value), width=4)
        p_bits = np.binary_repr(int(p_value), width=4)

        # activar controles para seleccionar el estado |t_value>
        for i, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[i])

        # escribir amplitud
        for i, bit in enumerate(a_bits):
            if bit == "1":
                qc.mcx(t_qubits, a_qubits[i])

        # escribir fase
        for i, bit in enumerate(p_bits):
            if bit == "1":
                qc.mcx(t_qubits, p_qubits[i])

        # deshacer X de selección temporal
        for i, bit in enumerate(t_bits):
            if bit == "0":
                qc.x(t_qubits[i])

    return qc



def draw_circuit(qc, save_path="circuits/circuit.png"):

    # crear carpeta si no existe
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # dibujar circuito
    fig = qc.draw(output="mpl")

    # guardar
    fig.savefig(save_path)

    print(f"Circuit saved to {save_path}")


def build_feature_map(feature_map, data,data_p=None):
    n_qubits = len(data)

    if(feature_map == "ry"):
        return ry_feature_map(n_qubits=n_qubits, data=data)
    if(feature_map in {"rz", "phase"}):
        return rz_feature_map(n_qubits=n_qubits, data=data)
    if(feature_map == "qtse"):
        return qtse(data=data)
    if(feature_map == "qtse_timbre_phase1"):
        return qtse_timbre_phase1(data_a=data, data_p=data_p)
    
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


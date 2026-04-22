from pathlib import Path

from matplotlib.pyplot import clf
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from datos_sin import generate_sin_phase, visualize_dataset, print_dataset, generate_sin_frequency
from feature_maps import ry_feature_map, rz_feature_map, draw_circuit, qtse
from kernel import fidelity_circuit, get_fidelity, get_kernel_matrix, statevectors, fidelity_kernel_matrix
from preprocess import preprocess, load_aeon_gunpoint, preprocess_phase
import numpy as np


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

def plot_data(X, y=None, save_path="preprocessed_data_plot.png"):
    plt.figure(figsize=(10, 5))

    for i in range(len(X)):
        if y is not None:
            if y[i] == 0:
                plt.plot(X[i], color="blue", alpha=0.7)
            else:
                plt.plot(X[i], color="red", alpha=0.7)
        else:
            plt.plot(X[i], marker="o", alpha=0.8)

    # leyenda manual si hay labels
    if y is not None:
        plt.plot([], [], color="blue", label="class 0")
        plt.plot([], [], color="red", label="class 1")
        plt.legend()
    else:
        plt.legend(["samples"])

    plt.title("Preprocessed Data")
    plt.xlabel("Frame Index")
    plt.ylabel("Quantized Value")
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()


def take_per_class(X, y, n_per_class=5):
    classes = np.unique(y)

    X_out = []
    y_out = []

    for c in classes:
        idx = np.where(y == c)[0][:n_per_class]
        X_out.append(X[idx])
        y_out.append(y[idx])

    X_out = np.vstack(X_out)
    y_out = np.concatenate(y_out)

    return X_out, y_out

def svm_classification(K_train, y_train, K_test, y_test):
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

def svm_classic_kernel(X_train, y_train, X_test, y_test, kernel_type="rbf"):
    clf = SVC(kernel=kernel_type)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Kernel: {kernel_type}")
    print("Accuracy:", acc)

def build_train_kernel(X_train, feature_map):
    states_train = statevectors(X_train, feature_map)
    K_train = fidelity_kernel_matrix(states_train)
    return K_train, states_train

def build_test_kernel(X_test, states_train, feature_map):
    states_test = statevectors(X_test, feature_map)
    K_test = fidelity_kernel_matrix(states_test, states_train)
    return K_test

def main():    

    # load preprocessed data
    #out_dir = Path("preprocessed_data")
    #X_train = np.load(out_dir / "gunpoint_train.npz")["X"]
    #y_train = np.load(out_dir / "gunpoint_train.npz")["y"]
    #X_test = np.load(out_dir / "gunpoint_test.npz")["X"]
    #y_test = np.load(out_dir / "gunpoint_test.npz")["y"]

    X_train, y_train, X_test, y_test = load_aeon_gunpoint()

    X_train_small, y_train_small = take_per_class(X_train, y_train, n_per_class=5)
    X_test_small, y_test_small = take_per_class(X_test, y_test, n_per_class=5)

    X_train_small = preprocess(X_train_small, sr=100)
    X_test_small = preprocess(X_test_small, sr=100)

    np.savetxt("preproc_ampl_Xtrain.txt", X_train_small, fmt="%.6f")
    np.savetxt("preproc_ampl_Xtest.txt", X_test_small, fmt="%.6f")

    # cojer un dato y pasar por qtse y dibujar circuito

    qc = qtse(data=X_train_small[0])
    draw_circuit(qc, save_path="circuits/qtse_circuit.png")

    #X_train_small, y_train_small = take_per_class(X_train, y_train, n_per_class=5)
    #X_test_small, y_test_small = take_per_class(X_test, y_test, n_per_class=5)

    print(X_train_small.shape)
    print(X_test_small.shape)

    plot_data(X_test_small, y_test_small)


    #K_train = get_kernel_matrix(X_train_small, feature_map="qtse", backend="statevector")
    #np.savetxt("qtse_ampl_Ktrain.txt", K_train, fmt="%.6f")
    #K_test = get_kernel_matrix(X_test_small, feature_map="qtse", backend="statevector", X2=X_train_small)
    #np.savetxt("qtse_ampl_Ktest.txt", K_test, fmt="%.6f")

    K_train, states_train = build_train_kernel(X_train_small, feature_map="qtse")
    np.savetxt("qtse_ampl_Ktrain.txt", K_train, fmt="%.6f")
    K_test = build_test_kernel(X_test_small, states_train, feature_map="qtse")
    np.savetxt("qtse_ampl_Ktest.txt", K_test, fmt="%.6f")
    
    svm_classification(K_train, y_train_small, K_test, y_test_small)

    #K_train = process(out_dir / "gunpoint_train.npz", encoding="qtse")
    #K_test = process(out_dir / "gunpoint_test.npz", encoding="qtse")

    #svm_classic_kernel(X_train_small, y_train_small, X_test_small, y_test_small, kernel_type="rbf")


if __name__ == "__main__":
    main()
from pathlib import Path

from matplotlib.pyplot import clf
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from datos_sin import generate_sin_phase, visualize_dataset, print_dataset, generate_sin_frequency
from feature_maps import ry_feature_map, rz_feature_map, draw_circuit, qtse,qtse_timbre_phase1
from kernel import fidelity_circuit, get_fidelity, get_kernel_matrix, statevectors, fidelity_kernel_matrix
from preprocess import preprocess, load_aeon_gunpoint, preprocess_phase, load_aeon_hearbeat, load_aeon_ECG200
import numpy as np
from sklearn.model_selection import train_test_split


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


def take_per_class(X, y, n_per_class):
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
    #print("Accuracy:", acc)

    return acc

def svm_classic_kernel(X_train, y_train, X_test, y_test, kernel_type="rbf"):
    clf = SVC(kernel=kernel_type)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Y test:", y_test)
    print("Y pred:", y_pred)
    #print(f"Kernel: {kernel_type}")
    #print("Accuracy:", acc)
    return acc

def build_train_kernel(X_train, feature_map):
    states_train = statevectors(X_train, feature_map)
    K_train = fidelity_kernel_matrix(states_train)
    return K_train, states_train

def build_test_kernel(X_test, states_train, feature_map):
    states_test = statevectors(X_test, feature_map)
    K_test = fidelity_kernel_matrix(states_test, states_train)
    return K_test

def run_one_holdout(X_train_raw, y_train, X_test_raw, y_test, test_size=0.3,seed=42,backend="statevector"):

    # preprocess data
    X_train_am = preprocess(X_train_raw, sr=100)
    X_test_am = preprocess(X_test_raw, sr=100)

    X_train_ph = preprocess_phase(X_train_raw, sr=100)
    X_test_ph = preprocess_phase(X_test_raw, sr=100)

    # build kernels
    K_train_qtse = get_kernel_matrix(X_train_am, feature_map="qtse", backend=backend)
    K_test_qtse = get_kernel_matrix(X_test_am, feature_map="qtse", backend=backend, X2=X_train_am)
    K_train_qtse_timbre_phase1 = get_kernel_matrix(X_train_am, feature_map="qtse_timbre_phase1", backend=backend, X1_p=X_train_ph)
    K_test_qtse_timbre_phase1 = get_kernel_matrix(X_test_am, feature_map="qtse_timbre_phase1", backend=backend, X2=X_train_am, X1_p=X_test_ph, X2_p=X_train_ph)

    # classification
    acc_qtse1 = svm_classification(K_train_qtse, y_train, K_test_qtse, y_test)
    acc_qtse2 = svm_classification(K_train_qtse_timbre_phase1, y_train, K_test_qtse_timbre_phase1, y_test)
    acc_classic = svm_classic_kernel(X_train_raw, y_train, X_test_raw, y_test, kernel_type="rbf")

    return acc_qtse1, acc_qtse2, acc_classic





def run_repeated_holout(n_sizes,n_runs, test_size=0.3, seed=42, backend="statevector"):

    # cargar datos
    X_tr, y_tr, X_te, y_te = load_aeon_ECG200()
    X_all = np.vstack((X_tr, X_te))
    y_all = np.concatenate((y_tr, y_te))

    acc_qtse1_mean_list = []
    acc_qtse2_mean_list = []
    acc_classic_mean_list = []

    acc_qtse1_std_list = []
    acc_qtse2_std_list = []
    acc_classic_std_list = []

    for size in n_sizes:

        print(f"Running for size {size} per class")
        X_sub, y_sub = take_per_class(X_all, y_all, n_per_class=size)

        acc_qtse1_list = []
        acc_qtse2_list = []
        acc_classic_list = []

        for run in range(n_runs):
            # dividir datos
            print("Run", run + 1)
            X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=test_size, random_state=seed + run)

            acc_qtse1, acc_qtse2, acc_classic = run_one_holdout(X_train, y_train, X_test, y_test, test_size=test_size, seed=seed + run, backend=backend)

            acc_qtse1_list.append(acc_qtse1)
            acc_qtse2_list.append(acc_qtse2)
            acc_classic_list.append(acc_classic)
            print(f"QTSE Accuracy: {acc_qtse1:.4f}")
            print(f"QTSE Timbre+Phase Accuracy: {acc_qtse2:.4f}")
            print(f"Classic RBF Accuracy: {acc_classic:.4f}")
            print("-" * 30)


        acc_qtse1_mean = np.mean(acc_qtse1_list)
        acc_qtse2_mean = np.mean(acc_qtse2_list)
        acc_classic_mean = np.mean(acc_classic_list)

        acc_qtse1_std = np.std(acc_qtse1_list)
        acc_qtse2_std = np.std(acc_qtse2_list)
        acc_classic_std = np.std(acc_classic_list)

        acc_qtse1_mean_list.append(acc_qtse1_mean)
        acc_qtse2_mean_list.append(acc_qtse2_mean)
        acc_classic_mean_list.append(acc_classic_mean)

        acc_qtse1_std_list.append(acc_qtse1_std)
        acc_qtse2_std_list.append(acc_qtse2_std)
        acc_classic_std_list.append(acc_classic_std)

    # print results
    print("Results:")
    for i, size in enumerate(n_sizes):
        print(f"Size {size} per class:")
        print(f"QTSE Accuracy: {acc_qtse1_mean_list[i]:.4f} ± {acc_qtse1_std_list[i]:.4f}")
        print(f"QTSE Timbre+Phase Accuracy: {acc_qtse2_mean_list[i]:.4f} ± {acc_qtse2_std_list[i]:.4f}")
        print(f"Classic RBF Accuracy: {acc_classic_mean_list[i]:.4f} ± {acc_classic_std_list[i]:.4f}")
        print("-" * 30)

def main():    

    #X_train, y_train, X_test, y_test = load_aeon_gunpoint()

    #X_train_small, y_train_small = take_per_class(X_train, y_train, n_per_class=5)
    #X_test_small, y_test_small = take_per_class(X_test, y_test, n_per_class=5)

    #X_train_small = preprocess(X_train_small, sr=100)
    #X_test_small = preprocess(X_test_small, sr=100)

    #np.savetxt("preproc_ampl_Xtrain.txt", X_train_small, fmt="%.6f")
    #np.savetxt("preproc_ampl_Xtest.txt", X_test_small, fmt="%.6f")
    #np.savetxt("preproc_y_train.txt", y_train_small, fmt="%d")
    #np.savetxt("preproc_y_test.txt", y_test_small, fmt="%d")

    #cargar datos preprocesados de preproc_ampl_Xtrain.txt y preproc_ampl_Xtest.txt y preproc_phase_Xtrain.txt y preproc_phase_Xtest.txt
    #X_train_ampl = np.loadtxt("preproc_ampl_Xtrain.txt")
    #X_test_ampl = np.loadtxt("preproc_ampl_Xtest.txt")
    #X_train_phase = np.loadtxt("preproc_phase_Xtrain.txt")
    #X_test_phase = np.loadtxt("preproc_phase_Xtest.txt")

    # build kernels
    #K_train_qtse = get_kernel_matrix(X_train_ampl, feature_map="qtse", backend="statevector")
    #K_test_qtse = get_kernel_matrix(X_test_ampl, feature_map="qtse", backend="statevector", X2=X_train_ampl)
    #K_train_qtse_timbre_phase1 = get_kernel_matrix(X_train_ampl, feature_map="qtse_timbre_phase1", backend="statevector", X1_p=X_train_phase)
    #K_test_qtse_timbre_phase1 = get_kernel_matrix(X_test_ampl, feature_map="qtse_timbre_phase1", backend="statevector", X2=X_train_ampl, X1_p=X_test_phase, X2_p=X_train_phase)

    # classification
    #svm_classification(K_train_qtse, y_train_small, K_test_qtse, y_test_small)
    #svm_classification(K_train_qtse_timbre_phase1, y_train_small, K_test_qtse_timbre_phase1, y_test_small)

    #ver resultados de clasificación y comparar con SVM clásico
    #svm_classic_kernel(X_train_ampl, y_train_small, X_test_ampl, y_test_small, kernel_type="rbf")

    # ejecutar prueba rápida para ver que la función run_repeated_holdout funciona
    n_sizes = [10]
    run_repeated_holout(n_sizes=n_sizes,n_runs=5, test_size=0.3, seed=42, backend="statevector")


    # ejecutar holdout repetido grande
    #n_sizes = [20, 50, 70, 100]
    #run_repeated_holout(n_sizes=n_sizes,n_runs=10, test_size=0.3, seed=42, backend="statevector")
    #run_one_holdout(X_train, y_train, X_test, y_test, test_size=0.3, seed=42, backend="statevector")

    # Ejecutar solo classic kernel para 10 de train y 10 de test
    #X_train, y_train, X_test, y_test = load_aeon_ECG200()

    #X_sub, y_sub = take_per_class(X_train, y_train, n_per_class=10)
    #print()
    #X_test_sub, y_test_sub = take_per_class(X_test, y_test, n_per_class=10)

    #acc_classic = svm_classic_kernel(X_sub, y_sub, X_test_sub, y_test_sub, kernel_type="rbf")
    #print(f"Classic RBF Accuracy: {acc_classic:.4f}")

    





if __name__ == "__main__":
    main()
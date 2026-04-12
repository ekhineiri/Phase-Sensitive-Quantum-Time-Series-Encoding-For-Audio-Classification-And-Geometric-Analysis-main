from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from kernel import build_kernel_matrix, build_qtse_kernel_matrix


# Cambia estos valores para elegir qué ejecutar.
DATASET = "ordered_permuted"  # "phase" | "freq" | "temporal_order" | "chirp_direction" | "time_reverse" | "ordered_permuted"
ENCODINGS = ["qtse", "ry", "phase"]
N_QUBITS = 8
N_TRAIN = 10


DATASET_PATHS = {
    "phase": "data/synthetic/qtse_10_samples.npz",
    "freq": "data/synthetic/qtse_freq_samples.npz",
    "temporal_order": "data/synthetic/qtse_temporal_order_samples.npz",
    "chirp_direction": "data/synthetic/qtse_chirp_direction_samples.npz",
    "time_reverse": "data/synthetic/qtse_time_reverse_samples.npz",
    "ordered_permuted": "data/synthetic/qtse_ordered_vs_permuted_samples.npz",
}


def absolute_path(path_text):
    return Path(to_absolute_path(path_text))


def create_run_dir(cfg):
    now = datetime.now()
    base = absolute_path(str(cfg.run.output_dir))
    run_dir = base / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def kernel_stats(kernel_matrix, labels):
    intra = []
    inter = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                intra.append(float(kernel_matrix[i, j]))
            else:
                inter.append(float(kernel_matrix[i, j]))

    intra_mean = float(np.mean(intra)) if intra else float("nan")
    inter_mean = float(np.mean(inter)) if inter else float("nan")
    ratio = intra_mean / inter_mean if inter_mean > 1e-12 else float("nan")

    return {
        "intra_class_mean": intra_mean,
        "inter_class_mean": inter_mean,
        "intra_inter_ratio": ratio,
        "diag_mean": float(np.mean(np.diag(kernel_matrix))),
    }


def split_train_test(labels, n_train):
    n_per_class = n_train // 2
    class0 = np.where(labels == 0)[0]
    class1 = np.where(labels == 1)[0]

    train_idx = np.concatenate([class0[:n_per_class], class1[:n_per_class]])
    test_idx = np.concatenate([class0[n_per_class:], class1[n_per_class:]])

    return train_idx, test_idx, n_per_class


def save_kernel_text(
    out_dir,
    encoding,
    label,
    class_names,
    labels,
    kernel_matrix,
    stats,
    y_test,
    y_pred,
    acc,
    n_per_class,
):
    txt_path = out_dir / f"{encoding}_kernel_matrix_{label}.txt"
    lines = [
        f"dataset: {label}",
        f"encoding: {encoding}",
        f"clases: {class_names}",
        f"etiquetas: {labels.tolist()}",
        "",
        "Kernel matrix:",
        np.array2string(kernel_matrix, precision=6, suppress_small=True),
        "",
        "Separability stats:",
        f"  intra_class_mean : {stats['intra_class_mean']:.6f}",
        f"  inter_class_mean : {stats['inter_class_mean']:.6f}",
        f"  intra_inter_ratio: {stats['intra_inter_ratio']:.6f}",
        f"  diag_mean        : {stats['diag_mean']:.6f}",
        "",
        "SVM (kernel precalculado):",
        f"  train : {2 * n_per_class} muestras ({n_per_class} por clase)",
        f"  test  : {len(y_test)} muestras",
        f"  y_test : {y_test.tolist()}",
        f"  y_pred : {y_pred.tolist()}",
        f"  accuracy : {acc:.6f}",
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Kernel (texto) guardado en: {txt_path}")


def save_kernel_heatmap(out_dir, encoding, label, kernel_matrix, labels):
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(kernel_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"{encoding.upper()} Kernel Matrix - {label}")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        ax.set_xticks(np.arange(kernel_matrix.shape[1]))
        ax.set_yticks(np.arange(kernel_matrix.shape[0]))

        half = len(labels) // 2
        ax.axhline(half - 0.5, color="red", linewidth=1.2, linestyle="--")
        ax.axvline(half - 0.5, color="red", linewidth=1.2, linestyle="--")

        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("K[i, j]")

        fig.tight_layout()
        fig_path = out_dir / f"{encoding}_kernel_matrix_{label}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Heatmap del kernel guardado en: {fig_path}")
    except Exception as exc:
        print(f"No se pudo guardar heatmap del kernel: {exc}")


def run_kernel_experiment(out_dir, dataset_path, label, encoding, n_qubits, n_train):
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No existe {dataset_path}. Ejecuta primero datos_sin.py para generarlo."
        )

    data = np.load(dataset_path, allow_pickle=True)
    audio = np.asarray(data["audio"], dtype=int)
    time_steps = np.asarray(data["t"], dtype=int)
    labels = np.asarray(data["y"], dtype=int)
    class_names = list(data["class_names"])

    encoding = encoding.lower()
    if encoding == "qtse":
        kernel_matrix = build_qtse_kernel_matrix(audio, time_steps, n_qubits=n_qubits)
    elif encoding in {"ry", "phase"}:
        angles = (audio.astype(float) / 15.0) * np.pi
        kernel_matrix = build_kernel_matrix(
            angles,
            angles,
            feature_map=encoding,
            entanglement="ring",
            backend="statevector",
        )
    else:
        raise ValueError("ENCODING desconocido. Usa 'qtse', 'ry' o 'phase'.")

    print(f"\n{'=' * 60}")
    print(f"ANALISIS KERNEL {encoding.upper()} - dataset: {label}")
    print(f"clases: {class_names}")
    print(f"etiquetas: {labels.tolist()}")
    print(f"{'=' * 60}")
    print("Matriz de kernel:")
    print(np.array2string(kernel_matrix, precision=4, suppress_small=True))
    print(f"Shape kernel: {kernel_matrix.shape}")

    stats = kernel_stats(kernel_matrix, labels)
    print("\nEstadisticas de separabilidad:")
    print(f"Media intra-clase : {stats['intra_class_mean']:.4f}")
    print(f"Media inter-clase : {stats['inter_class_mean']:.4f}")
    print(f"Ratio intra/inter : {stats['intra_inter_ratio']:.4f}")
    print(f"Media diagonal    : {stats['diag_mean']:.4f}")

    train_idx, test_idx, n_per_class = split_train_test(labels, n_train)
    kernel_train = kernel_matrix[np.ix_(train_idx, train_idx)]
    kernel_test = kernel_matrix[np.ix_(test_idx, train_idx)]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    svm = SVC(kernel="precomputed", C=1.0)
    svm.fit(kernel_train, y_train)
    y_pred = svm.predict(kernel_test)
    acc = float(accuracy_score(y_test, y_pred))

    print("\nSVM (kernel precalculado):")
    print(f"Train    : {len(y_train)} muestras ({n_per_class} por clase)")
    print(f"Test     : {len(y_test)} muestras")
    print(f"y_test   : {y_test.tolist()}")
    print(f"y_pred   : {y_pred.tolist()}")
    print(f"Accuracy : {acc:.4f}")

    matrix_path = out_dir / f"{encoding}_kernel_matrix_{label}.npy"
    np.save(matrix_path, kernel_matrix)
    print(f"\nKernel guardado en: {matrix_path}")

    save_kernel_text(
        out_dir,
        encoding,
        label,
        class_names,
        labels,
        kernel_matrix,
        stats,
        y_test,
        y_pred,
        acc,
        n_per_class,
    )
    save_kernel_heatmap(out_dir, encoding, label, kernel_matrix, labels)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    if DATASET not in DATASET_PATHS:
        raise ValueError(f"DATASET desconocido: {DATASET}")

    dataset_path = absolute_path(DATASET_PATHS[DATASET])
    run_dir = create_run_dir(cfg)
    print(f"Carpeta de ejecucion: {run_dir}")

    for encoding in ENCODINGS:
        run_kernel_experiment(
            run_dir,
            dataset_path=dataset_path,
            label=DATASET,
            encoding=encoding,
            n_qubits=N_QUBITS,
            n_train=N_TRAIN,
        )


if __name__ == "__main__":
    main()
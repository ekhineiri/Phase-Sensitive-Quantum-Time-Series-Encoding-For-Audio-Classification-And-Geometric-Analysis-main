from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from kernel import build_kernel_matrix


def load_processed_dataset(path: str | Path) -> dict[str, np.ndarray]:
    dataset = np.load(Path(path), allow_pickle=True)
    return {key: dataset[key] for key in dataset.files}


def prepare_run_data(dataset: dict[str, np.ndarray], cfg: DictConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    available = [str(name) for name in dataset["feature_names"].tolist()]

    if cfg.feature_names:
        selected_indices = np.asarray([available.index(name) for name in cfg.feature_names], dtype=np.int64)
    else:
        selected_indices = np.arange(min(cfg.n_qubits, len(available)), dtype=np.int64)

    selected_features = [available[i] for i in selected_indices]
    X_quantum = np.asarray(dataset["X_angle"] if cfg.use_angles else dataset["X_raw"])[:, selected_indices]
    X_raw = np.asarray(dataset["X_raw"])[:, selected_indices]
    y = np.asarray(dataset["y"], dtype=np.int64)
    return X_quantum, X_raw, y, selected_features


def summarize_result(result: dict[str, object]) -> dict[str, object]:
    return {
        "feature_map": result["feature_map"],
        "accuracy": result["accuracy"],
        "train_shape": result["train_shape"],
        "test_shape": result["test_shape"],
        "selected_features": result["selected_features"],
    }


def _split_and_subsample(
    y: np.ndarray,
    test_size: float,
    random_state: int,
    max_train_samples: int | None,
    max_test_samples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y), dtype=np.int64)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state, stratify=y)

    if max_train_samples is not None and max_train_samples < len(train_idx):
        train_idx, _ = train_test_split(
            train_idx,
            train_size=max_train_samples,
            random_state=random_state,
            stratify=y[train_idx],
        )

    if max_test_samples is not None and max_test_samples < len(test_idx):
        test_idx, _ = train_test_split(
            test_idx,
            train_size=max_test_samples,
            random_state=random_state,
            stratify=y[test_idx],
        )

    return train_idx, test_idx


def run_classical_baseline(
    X_raw: np.ndarray,
    y: np.ndarray,
    selected_features: list[str],
    cfg: DictConfig,
) -> dict[str, object]:
    train_idx, test_idx = _split_and_subsample(
        y=y,
        test_size=float(cfg.test_size),
        random_state=int(cfg.random_state),
        max_train_samples=cfg.max_train_samples,
        max_test_samples=cfg.max_test_samples,
    )

    X_train = X_raw[train_idx]
    X_test = X_raw[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=float(cfg.classical_c), gamma=str(cfg.classical_gamma)),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "feature_map": "classical_rbf",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "selected_features": selected_features,
    }


def run_quantum_svm_pipeline(
    X_quantum: np.ndarray,
    y: np.ndarray,
    selected_features: list[str],
    feature_map: str,
    cfg: DictConfig,
) -> dict[str, object]:
    train_idx, test_idx = _split_and_subsample(
        y=y,
        test_size=float(cfg.test_size),
        random_state=int(cfg.random_state),
        max_train_samples=cfg.max_train_samples,
        max_test_samples=cfg.max_test_samples,
    )

    X_train = X_quantum[train_idx]
    X_test = X_quantum[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    shots = int(cfg.get("shots", 1024))
    simulator_seed = cfg.get("simulator_seed", 42)
    kernel_backend = str(cfg.get("kernel_backend", "aer"))

    K_train = build_kernel_matrix(
        X_train,
        X_train,
        feature_map=feature_map,
        entanglement=str(cfg.entanglement),
        backend=kernel_backend,
        shots=shots,
        seed=int(simulator_seed) if simulator_seed is not None else None,
    )
    K_test = build_kernel_matrix(
        X_test,
        X_train,
        feature_map=feature_map,
        entanglement=str(cfg.entanglement),
        backend=kernel_backend,
        shots=shots,
        seed=int(simulator_seed) if simulator_seed is not None else None,
    )

    model = SVC(kernel="precomputed", C=float(cfg.svm_c))
    model.fit(K_train, y_train)
    y_pred = model.predict(K_test)

    return {
        "feature_map": feature_map,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "selected_features": selected_features,
    }

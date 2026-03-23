from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from feature_maps import qtse
from kernel import build_feature_map, build_qtse_kernel_matrix
from preprocess import DEFAULT_BANDS, build_dataset, load_eea, preprocess_qtse
from svm import (
    load_processed_dataset,
    prepare_run_data,
    run_classical_baseline,
    run_quantum_svm_pipeline,
    summarize_result,
)


def _cfg_path(path_value: str) -> Path:
    return Path(to_absolute_path(path_value))


def _outputs_dir(cfg: DictConfig) -> Path:
    out_dir = _cfg_path(str(cfg.run.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _preprocess(cfg: DictConfig) -> None:
    d = cfg.data
    dataset = build_dataset(
        data_root=_cfg_path(str(d.root)),
        fs=float(d.fs),
        window_sec=float(d.window_sec),
        step_sec=float(d.step_sec),
        bands=DEFAULT_BANDS,
        class_dirs=list(d.classes),
        max_files_per_class=d.max_files_per_class,
    )
    output_path = _cfg_path(str(d.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)
    print(f"Preprocesamiento completado: {output_path}")
    print(f"Ventanas totales: {dataset['X_raw'].shape[0]}")
    print(f"Features por ventana: {dataset['X_raw'].shape[1]}")


def _run(cfg: DictConfig) -> None:
    dataset = load_processed_dataset(_cfg_path(str(cfg.run.dataset)))
    X_quantum, X_raw, y, selected_features = prepare_run_data(dataset, cfg.run)
    out_root = _outputs_dir(cfg)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_maps = ["ry", "rz", "qtme"] if cfg.run.feature_map == "all" else [cfg.run.feature_map]
    all_results: list[dict[str, object]] = []

    if not cfg.run.skip_classical_baseline:
        result = run_classical_baseline(
            X_raw=X_raw,
            y=y,
            selected_features=selected_features,
            cfg=cfg.run,
        )
        all_results.append(result)
        print("Classical baseline:")
        print(json.dumps(summarize_result(result), indent=2, default=str))
        print(result["report"])

    for fm in feature_maps:
        result = run_quantum_svm_pipeline(
            X_quantum=X_quantum,
            y=y,
            selected_features=selected_features,
            feature_map=fm,
            cfg=cfg.run,
        )
        all_results.append(result)
        print(f"Quantum pipeline ({fm}):")
        print(json.dumps(summarize_result(result), indent=2, default=str))
        print(result["report"])

        # Save one circuit drawing per feature map using the first sample.
        sample = np.asarray(X_quantum[0], dtype=float)
        circuit = build_feature_map(feature_map=fm, data=sample, entanglement=str(cfg.run.entanglement))
        try:
            from qiskit.visualization import circuit_drawer

            fig = circuit_drawer(circuit, output="mpl")
            fig_path = run_dir / f"circuit_{fm}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"Circuito guardado en: {fig_path}")
        except Exception:
            txt_path = run_dir / f"circuit_{fm}.txt"
            txt_path.write_text(str(circuit.draw(output="text")), encoding="utf-8")
            print(f"Circuito (texto) guardado en: {txt_path}")

    results_path = run_dir / "results.json"
    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "config": OmegaConf.to_container(cfg.run, resolve=True),
        "results": [
            {
                **summarize_result(result),
                "report": result["report"],
            }
            for result in all_results
        ],
    }
    results_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Resultados guardados en: {results_path}")


def _run_qtse_only(cfg: DictConfig) -> None:
    data_root = _cfg_path(str(cfg.data.root))

    signal_path = None
    for class_name in cfg.data.classes:
        class_path = data_root / str(class_name)
        if not class_path.exists():
            continue
        files = sorted(class_path.glob("*.eea"))
        if files:
            signal_path = files[0]
            break

    if signal_path is None:
        raise FileNotFoundError(f"No se encontraron archivos .eea en {data_root}")

    signal = load_eea(signal_path)
    t, audio = preprocess_qtse(
        signal=signal,
        fs=float(cfg.data.fs),
        n_mfcc=13,
        n_frames=16,
        n_mels=40,
        scalar_mode="energy",
    )

    circuit = qtse(n_qubits=8, audio=audio, t=t)
    print(f"QTSE input file: {signal_path}")
    print(f"t (16): {t.tolist()}")
    print(f"audio (16, cuantizado): {audio.tolist()}")
    print(circuit.draw(output="text"))

    out_dir = _outputs_dir(cfg)
    txt_path = out_dir / "circuit_qtse.txt"
    txt_path.write_text(str(circuit.draw(output="text")), encoding="utf-8")
    print(f"Circuito (texto) guardado en: {txt_path}")

    try:
        from qiskit.visualization import circuit_drawer

        fig = circuit_drawer(circuit, output="mpl")
        fig_path = out_dir / "circuit_qtse.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Circuito guardado en: {fig_path}")
    except Exception as exc:
        print(f"No se pudo guardar imagen del circuito: {exc}")


def _run_qtse_kernel_matrix(cfg: DictConfig) -> None:
    dataset_path = _cfg_path("data/synthetic/qtse_10_samples.npz")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No existe {dataset_path}. Ejecuta primero datos_sin.py para generar los 10 datos sintéticos."
        )

    data = np.load(dataset_path, allow_pickle=True)
    audio = np.asarray(data["audio"], dtype=int)
    t = np.asarray(data["t"], dtype=int)

    K = build_qtse_kernel_matrix(audio, t, n_qubits=8)

    print("Matriz de kernel QTSE:")
    print(np.array2string(K, precision=4, suppress_small=True))
    print(f"Shape kernel: {K.shape}")

    out_dir = _outputs_dir(cfg)
    matrix_path = out_dir / "qtse_kernel_matrix_10x10.npy"
    np.save(matrix_path, K)
    print(f"Kernel guardado en: {matrix_path}")

    txt_path = out_dir / "qtse_kernel_matrix_10x10.txt"
    txt_path.write_text(np.array2string(K, precision=6, suppress_small=True), encoding="utf-8")
    print(f"Kernel (texto) guardado en: {txt_path}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(K, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title("QTSE Kernel Matrix (10x10)")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        ax.set_xticks(np.arange(K.shape[1]))
        ax.set_yticks(np.arange(K.shape[0]))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("K[i, j]")
        fig.tight_layout()
        fig_path = out_dir / "qtse_kernel_matrix_10x10.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap del kernel guardado en: {fig_path}")
    except Exception as exc:
        print(f"No se pudo guardar heatmap del kernel: {exc}")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Flujo anterior temporalmente desactivado; se conserva comentado.
    # if cfg.command == "preprocess":
    #     _preprocess(cfg)
    # elif cfg.command == "run":
    #     _run(cfg)
    # elif cfg.command == "full":
    #     _preprocess(cfg)
    #     _run(cfg)
    # else:
    #     raise ValueError(f"Comando desconocido: {cfg.command!r}. Usa preprocess | run | full")

    # Ejecutar cálculo de kernel QTSE sobre los 10 datos sintéticos.
    _run_qtse_kernel_matrix(cfg)


if __name__ == "__main__":
    main()

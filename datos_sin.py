from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def quantize_to_4bit(values: np.ndarray) -> np.ndarray:
    """Mapea un vector continuo a enteros en [0, 15]."""
    arr = np.asarray(values, dtype=float)
    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    if v_max - v_min < 1e-12:
        return np.zeros_like(arr, dtype=int)
    normalized = (arr - v_min) / (v_max - v_min)
    return np.clip(np.round(normalized * 15), 0, 15).astype(int)


def generate_qtse_synthetic_samples(
    n_items: int = 10,
    n_frames: int = 16,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Genera muestras sintéticas QTSE etiquetadas de forma phase-dependent."""
    rng = np.random.default_rng(seed)

    t_cont = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    t_discrete = np.arange(n_frames, dtype=int)

    audio_samples: list[np.ndarray] = []
    time_samples: list[np.ndarray] = []
    frequencies: list[float] = []
    phases: list[float] = []
    amplitudes: list[float] = []
    labels: list[int] = []

    # Clase 0 y 1 comparten rango de frecuencia; la diferencia principal es la fase.
    # Clase 0: fase en [0, pi), clase 1: fase en [pi, 2*pi).
    for idx in range(n_items):
        label = 0 if idx < (n_items // 2) else 1
        omega = float(rng.uniform(1.0, 3.0))

        if label == 0:
            phase = float(rng.uniform(0.0, np.pi))
        else:
            phase = float(rng.uniform(np.pi, 2.0 * np.pi))
        amplitude = float(rng.uniform(0.7, 1.3))
        noise = rng.normal(0.0, 0.08, size=n_frames)

        waveform = amplitude * np.sin(omega * t_cont + phase) + noise
        audio_q = quantize_to_4bit(waveform)

        audio_samples.append(audio_q)
        time_samples.append(t_discrete.copy())
        frequencies.append(omega)
        phases.append(phase)
        amplitudes.append(amplitude)
        labels.append(label)

    return {
        "audio": np.asarray(audio_samples, dtype=int),
        "t": np.asarray(time_samples, dtype=int),
        "frequencies": np.asarray(frequencies, dtype=float),
        "phases": np.asarray(phases, dtype=float),
        "amplitudes": np.asarray(amplitudes, dtype=float),
        "y": np.asarray(labels, dtype=np.int64),
        "class_names": np.asarray(["phase_0_pi", "phase_pi_2pi"]),
    }


def print_generated_samples(data: dict[str, np.ndarray]) -> None:
    """Imprime los 10 datos generados para inspección rápida."""
    audio = data["audio"]
    t = data["t"]
    y = data["y"]
    class_names = data["class_names"]

    print("=" * 80)
    print("10 DATOS QTSE GENERADOS")
    print("=" * 80)
    for i in range(audio.shape[0]):
        print(f"Muestra {i:02d}")
        print(f"  clase : {int(y[i])} ({class_names[int(y[i])]})")
        print(f"  t     : {t[i].tolist()}")
        print(f"  audio : {audio[i].tolist()}")


def visualize_qtse_samples(data: dict[str, np.ndarray]) -> None:
    """Visualiza los 10 datos: líneas por muestra + heatmap 10x16."""
    audio = data["audio"]
    t = data["t"][0]
    y = data["y"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    class_colors = {0: "tab:blue", 1: "tab:orange"}
    for i in range(audio.shape[0]):
        c = class_colors[int(y[i])]
        ax.plot(t, audio[i], marker="o", linewidth=1.4, alpha=0.8, color=c, label=f"sample {i} (y={int(y[i])})")
    ax.set_title("10 series QTSE cuantizadas")
    ax.set_xlabel("frame t (0..15)")
    ax.set_ylabel("audio cuantizado (0..15)")
    ax.set_xticks(np.arange(0, 16, 1))
    ax.set_ylim(-0.5, 15.5)
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    im = ax2.imshow(audio, aspect="auto", cmap="viridis", vmin=0, vmax=15)
    ax2.set_title("Heatmap audio (10x16)")
    ax2.set_xlabel("frame t")
    ax2.set_ylabel("sample")
    ax2.set_xticks(np.arange(0, 16, 1))
    ax2.set_yticks(np.arange(0, 10, 1))
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("valor audio")

    plt.tight_layout()

    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "qtse_10_samples.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura guardada en: {out_path}")
    plt.show()


if __name__ == "__main__":
    dataset = generate_qtse_synthetic_samples(n_items=10, n_frames=16, seed=42)

    np.savez_compressed(
        "data/synthetic/qtse_10_samples.npz",
        audio=dataset["audio"],
        t=dataset["t"],
        y=dataset["y"],
        class_names=dataset["class_names"],
        frequencies=dataset["frequencies"],
        phases=dataset["phases"],
        amplitudes=dataset["amplitudes"],
    )
    print("Archivo guardado en: data/synthetic/qtse_10_samples.npz")

    print_generated_samples(dataset)
    visualize_qtse_samples(dataset)

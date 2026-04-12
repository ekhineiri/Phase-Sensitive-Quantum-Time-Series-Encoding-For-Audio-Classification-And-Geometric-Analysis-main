from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os


def quantize_to_4bit(values):
    """Mapea un vector continuo a enteros en [0, 15]."""
    arr = np.asarray(values, dtype=float)
    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    if v_max - v_min < 1e-12:
        return np.zeros_like(arr, dtype=int)
    normalized = (arr - v_min) / (v_max - v_min)
    return np.clip(np.round(normalized * 15), 0, 15).astype(int)


def generate_sin_phase(n_items=10, n_frames=16, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2*np.pi, n_frames, endpoint=False)

    y = np.zeros(n_items, dtype=int)
    y[n_items // 2:] = 1

    omega = np.full(n_items, 2.0)        # constante
    amplitude = np.ones(n_items) * 1.0    # constante
    noise = np.zeros((n_items, n_frames)) # sin ruido

    # SOLO cambia la fase
    phase = np.zeros(n_items)
    phase[y == 0] = rng.uniform(0.0, np.pi, size=(y == 0).sum())
    phase[y == 1] = rng.uniform(np.pi, 2*np.pi, size=(y == 1).sum())

    waveform = amplitude[:, None] * np.sin(omega[:, None] * t + phase[:, None]) + noise

    X = np.array([quantize_to_4bit(w) for w in waveform])

    return X, y

import os
import numpy as np

def print_dataset(X, y, save_path=None):

    lines = []
    def log(text=""):
        print(text)
        lines.append(text)

    log("=== DATASET INFO ===")
    log(f"Shape X: {X.shape}")
    log(f"Shape y: {y.shape}")
    log(f"Num clases: {len(np.unique(y))}")
    log()

    log("=== SAMPLE DATA ===")
    n_samples = len(X)

    for i in range(min(n_samples, len(X))):
        log(f"Sample {i} | class={y[i]}")
        log(str(X[i]))
        log("-" * 30)

    # Guardar
    if save_path is not None:
        base, _ = os.path.splitext(save_path)

        #guardar log TXT
        txt_path = base + "_log.txt"
        os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

        #guardar dataset
        npz_path = base + ".npz"
        np.savez(npz_path, X=X, y=y)

        print(f"\nSaved log to: {txt_path}")
        print(f"Saved dataset to: {npz_path}")

def visualize_dataset(X, y, save_path=None):
    import matplotlib.pyplot as plt
    import os

    plt.figure(figsize=(10, 6))

    for i in range(len(X)):
        if y[i] == 0:
            plt.plot(X[i], color="blue", alpha=0.7)
        else:
            plt.plot(X[i], color="red", alpha=0.7)

    plt.title("Dataset samples")
    plt.xlabel("Time / Frame")
    plt.ylabel("Value")
    plt.grid(True)

    # leyenda simple
    plt.plot([], [], color="blue", label="class 0")
    plt.plot([], [], color="red", label="class 1")
    plt.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved to {save_path}")

    plt.show()


def generate_sin_frequency(n_items=10, n_frames=16, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2*np.pi, n_frames, endpoint=False)

    y = np.zeros(n_items, dtype=int)
    y[n_items // 2:] = 1

    phase = np.full(n_items, np.pi / 4)     # fase constante
    amplitude = np.ones(n_items) * 1.0      # amplitud constante
    noise = np.zeros((n_items, n_frames))   # sin ruido

    # SOLO cambia la frecuencia
    omega = np.zeros(n_items)
    omega[y == 0] = rng.uniform(1.0, 2.0, size=(y == 0).sum())   # baja frecuencia
    omega[y == 1] = rng.uniform(4.0, 6.0, size=(y == 1).sum())   # alta frecuencia

    waveform = amplitude[:, None] * np.sin(omega[:, None] * t + phase[:, None]) + noise

    X = np.array([quantize_to_4bit(w) for w in waveform])

    return X, y
   


def generate_qtse_temporal_order_samples(
    n_items=10,
    n_frames=16,
    seed=42,
):
    """Genera muestras sintéticas QTSE etiquetadas por orden temporal de picos.

    Clase 0: pico positivo en la primera mitad,  pico negativo en la segunda  (↑ luego ↓).
    Clase 1: pico negativo en la primera mitad,  pico positivo en la segunda  (↓ luego ↑).

    La distinción es puramente temporal: las mismas frecuencias y amplitudes se
    usan en ambas clases; sólo cambia el orden en el tiempo del burst positivo
    y el burst negativo.
    """
    rng = np.random.default_rng(seed)

    t_norm = np.linspace(0.0, 1.0, n_frames, endpoint=False)  # [0, 1) normalizado
    t_discrete = np.arange(n_frames, dtype=int)

    def gaussian_bump(center, width=0.12):
        return np.exp(-((t_norm - center) ** 2) / (2.0 * width ** 2))

    audio_samples: list[np.ndarray] = []
    time_samples: list[np.ndarray] = []
    peak_positions: list[tuple[float, float]] = []
    amplitudes: list[float] = []
    labels: list[int] = []

    for idx in range(n_items):
        label = 0 if idx < (n_items // 2) else 1
        amplitude = float(rng.uniform(0.7, 1.3))
        noise = rng.normal(0.0, 0.06, size=n_frames)
        width = float(rng.uniform(0.08, 0.14))

        # Centros siempre en sus mitades respectivas para que el orden sea inequívoco.
        first_half_center = float(rng.uniform(0.12, 0.38))   # primera mitad
        second_half_center = float(rng.uniform(0.62, 0.88))  # segunda mitad

        up_bump = amplitude * gaussian_bump(first_half_center, width)
        down_bump = amplitude * gaussian_bump(second_half_center, width)

        if label == 0:
            # Pico ↑ primero, pico ↓ después.
            waveform = up_bump - down_bump + noise
        else:
            # Pico ↓ primero, pico ↑ después (orden invertido).
            waveform = -up_bump + down_bump + noise

        audio_q = quantize_to_4bit(waveform)

        audio_samples.append(audio_q)
        time_samples.append(t_discrete.copy())
        peak_positions.append((first_half_center, second_half_center))
        amplitudes.append(amplitude)
        labels.append(label)

    return {
        "audio": np.asarray(audio_samples, dtype=int),
        "t": np.asarray(time_samples, dtype=int),
        "amplitudes": np.asarray(amplitudes, dtype=float),
        "y": np.asarray(labels, dtype=np.int64),
        "class_names": np.asarray(["up_then_down", "down_then_up"]),
    }


def visualize_qtse_temporal_order_samples(data):
    """Visualiza los datos temporal-order: líneas por muestra + heatmap."""
    audio = data["audio"]
    t = data["t"][0]
    y = data["y"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    class_colors = {0: "tab:blue", 1: "tab:orange"}
    class_labels_used: set[int] = set()
    for i in range(audio.shape[0]):
        lbl = int(y[i])
        c = class_colors[lbl]
        class_name = str(data["class_names"][lbl])
        show_label = class_name not in class_labels_used
        ax.plot(
            t,
            audio[i],
            marker="o",
            linewidth=1.4,
            alpha=0.8,
            color=c,
            label=class_name if show_label else "_nolegend_",
        )
        class_labels_used.add(class_name)
    ax.set_title("Series QTSE por orden temporal de picos")
    ax.set_xlabel("frame t (0..15)")
    ax.set_ylabel("audio cuantizado (0..15)")
    ax.set_xticks(np.arange(0, n_frames := audio.shape[1], 1))
    ax.set_ylim(-0.5, 15.5)
    ax.axvline(n_frames / 2 - 0.5, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    im = ax2.imshow(audio, aspect="auto", cmap="viridis", vmin=0, vmax=15)
    ax2.set_title("Heatmap audio temporal-order")
    ax2.set_xlabel("frame t")
    ax2.set_ylabel("sample")
    ax2.set_xticks(np.arange(0, audio.shape[1], 1))
    ax2.set_yticks(np.arange(0, audio.shape[0], 1))
    ax2.axvline(audio.shape[1] / 2 - 0.5, color="red", linewidth=1.2, linestyle="--")
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("valor audio")

    plt.tight_layout()

    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "qtse_temporal_order_samples.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura guardada en: {out_path}")
    plt.show()


def _visualize_binary_qtse_dataset(
    data,
    title,
    out_filename,
):
    """Visualiza un dataset binario QTSE (lineas + heatmap)."""
    audio = data["audio"]
    t = data["t"][0]
    y = data["y"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    class_colors = {0: "tab:blue", 1: "tab:orange"}
    class_labels_used: set[str] = set()
    for i in range(audio.shape[0]):
        lbl = int(y[i])
        class_name = str(data["class_names"][lbl])
        show_label = class_name not in class_labels_used
        ax.plot(
            t,
            audio[i],
            marker="o",
            linewidth=1.4,
            alpha=0.85,
            color=class_colors[lbl],
            label=class_name if show_label else "_nolegend_",
        )
        class_labels_used.add(class_name)

    ax.set_title(title)
    ax.set_xlabel("frame t (0..15)")
    ax.set_ylabel("audio cuantizado (0..15)")
    ax.set_xticks(np.arange(0, audio.shape[1], 1))
    ax.set_ylim(-0.5, 15.5)
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax2 = axes[1]
    im = ax2.imshow(audio, aspect="auto", cmap="viridis", vmin=0, vmax=15)
    ax2.set_title("Heatmap audio")
    ax2.set_xlabel("frame t")
    ax2.set_ylabel("sample")
    ax2.set_xticks(np.arange(0, audio.shape[1], 1))
    ax2.set_yticks(np.arange(0, audio.shape[0], 1))
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("valor audio")

    plt.tight_layout()

    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / out_filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigura guardada en: {out_path}")
    plt.show()


def generate_qtse_chirp_direction_samples(
    n_items=10,
    n_frames=16,
    seed=42,
):
    """Clase 0: chirp ascendente, clase 1: chirp descendente."""
    rng = np.random.default_rng(seed)
    t_norm = np.linspace(0.0, 1.0, n_frames, endpoint=False)
    t_discrete = np.arange(n_frames, dtype=int)

    audio_samples: list[np.ndarray] = []
    time_samples: list[np.ndarray] = []
    start_freqs: list[float] = []
    end_freqs: list[float] = []
    amplitudes: list[float] = []
    labels: list[int] = []

    for idx in range(n_items):
        label = 0 if idx < (n_items // 2) else 1

        f_low = float(rng.uniform(1.0, 2.0))
        f_high = float(rng.uniform(4.0, 6.0))
        if label == 0:
            f_start, f_end = f_low, f_high
        else:
            f_start, f_end = f_high, f_low

        k = f_end - f_start
        phase = 2.0 * np.pi * (f_start * t_norm + 0.5 * k * t_norm**2)
        amplitude = float(rng.uniform(0.75, 1.25))
        noise = rng.normal(0.0, 0.05, size=n_frames)

        waveform = amplitude * np.sin(phase + np.pi / 6.0) + noise
        audio_q = quantize_to_4bit(waveform)

        audio_samples.append(audio_q)
        time_samples.append(t_discrete.copy())
        start_freqs.append(f_start)
        end_freqs.append(f_end)
        amplitudes.append(amplitude)
        labels.append(label)

    return {
        "audio": np.asarray(audio_samples, dtype=int),
        "t": np.asarray(time_samples, dtype=int),
        "start_freqs": np.asarray(start_freqs, dtype=float),
        "end_freqs": np.asarray(end_freqs, dtype=float),
        "amplitudes": np.asarray(amplitudes, dtype=float),
        "y": np.asarray(labels, dtype=np.int64),
        "class_names": np.asarray(["chirp_up", "chirp_down"]),
    }


def visualize_qtse_chirp_direction_samples(data):
    """Visualiza dataset chirp ascendente vs descendente."""
    _visualize_binary_qtse_dataset(
        data=data,
        title="Series QTSE: chirp ascendente vs descendente",
        out_filename="qtse_chirp_direction_samples.png",
    )


def generate_qtse_time_reverse_samples(
    n_items=10,
    n_frames=16,
    seed=42,
):
    """Clase 0: senal original, clase 1: la misma senal invertida en el tiempo."""
    rng = np.random.default_rng(seed)
    t_cont = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    t_discrete = np.arange(n_frames, dtype=int)

    audio_samples: list[np.ndarray] = []
    time_samples: list[np.ndarray] = []
    frequencies: list[float] = []
    phases: list[float] = []
    amplitudes: list[float] = []
    labels: list[int] = []

    n_pairs = n_items // 2
    for _ in range(n_pairs):
        omega = float(rng.uniform(1.0, 3.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        amplitude = float(rng.uniform(0.75, 1.25))
        envelope = 0.65 + 0.35 * np.sin(0.5 * t_cont + np.pi / 7.0)
        noise = rng.normal(0.0, 0.05, size=n_frames)

        base = amplitude * envelope * np.sin(omega * t_cont + phase) + noise
        audio_base_q = quantize_to_4bit(base)
        audio_reverse_q = audio_base_q[::-1].copy()

        audio_samples.append(audio_base_q)
        time_samples.append(t_discrete.copy())
        frequencies.append(omega)
        phases.append(phase)
        amplitudes.append(amplitude)
        labels.append(0)

        audio_samples.append(audio_reverse_q)
        time_samples.append(t_discrete.copy())
        frequencies.append(omega)
        phases.append(phase)
        amplitudes.append(amplitude)
        labels.append(1)

    if len(audio_samples) < n_items:
        extra = generate_qtse_time_reverse_samples(n_items=2, n_frames=n_frames, seed=seed + 999)
        audio_samples.append(extra["audio"][0])
        time_samples.append(t_discrete.copy())
        frequencies.append(float(extra["frequencies"][0]))
        phases.append(float(extra["phases"][0]))
        amplitudes.append(float(extra["amplitudes"][0]))
        labels.append(0)

    return {
        "audio": np.asarray(audio_samples[:n_items], dtype=int),
        "t": np.asarray(time_samples[:n_items], dtype=int),
        "frequencies": np.asarray(frequencies[:n_items], dtype=float),
        "phases": np.asarray(phases[:n_items], dtype=float),
        "amplitudes": np.asarray(amplitudes[:n_items], dtype=float),
        "y": np.asarray(labels[:n_items], dtype=np.int64),
        "class_names": np.asarray(["time_forward", "time_reverse"]),
    }


def generate_qtse_ordered_vs_permuted_samples(
    n_items=10,
    n_frames=16,
    seed=42,
):
    """Clase 0: serie ordenada temporalmente, clase 1: la misma serie permutada."""
    rng = np.random.default_rng(seed)
    t_cont = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    t_discrete = np.arange(n_frames, dtype=int)

    audio_samples: list[np.ndarray] = []
    time_samples: list[np.ndarray] = []
    frequencies: list[float] = []
    phases: list[float] = []
    amplitudes: list[float] = []
    labels: list[int] = []

    n_pairs = n_items // 2
    for _ in range(n_pairs):
        omega = float(rng.uniform(1.0, 3.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        amplitude = float(rng.uniform(0.75, 1.25))
        noise = rng.normal(0.0, 0.05, size=n_frames)

        ordered = amplitude * np.sin(omega * t_cont + phase) + 0.3 * np.sin(2.0 * omega * t_cont) + noise
        ordered_q = quantize_to_4bit(ordered)
        perm_q = ordered_q[rng.permutation(n_frames)]

        audio_samples.append(ordered_q)
        time_samples.append(t_discrete.copy())
        frequencies.append(omega)
        phases.append(phase)
        amplitudes.append(amplitude)
        labels.append(0)

        audio_samples.append(perm_q)
        time_samples.append(t_discrete.copy())
        frequencies.append(omega)
        phases.append(phase)
        amplitudes.append(amplitude)
        labels.append(1)

    if len(audio_samples) < n_items:
        extra = generate_qtse_ordered_vs_permuted_samples(n_items=2, n_frames=n_frames, seed=seed + 888)
        audio_samples.append(extra["audio"][0])
        time_samples.append(t_discrete.copy())
        frequencies.append(float(extra["frequencies"][0]))
        phases.append(float(extra["phases"][0]))
        amplitudes.append(float(extra["amplitudes"][0]))
        labels.append(0)

    return {
        "audio": np.asarray(audio_samples[:n_items], dtype=int),
        "t": np.asarray(time_samples[:n_items], dtype=int),
        "frequencies": np.asarray(frequencies[:n_items], dtype=float),
        "phases": np.asarray(phases[:n_items], dtype=float),
        "amplitudes": np.asarray(amplitudes[:n_items], dtype=float),
        "y": np.asarray(labels[:n_items], dtype=np.int64),
        "class_names": np.asarray(["ordered", "permuted"]),
    }
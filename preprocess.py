import re
from pathlib import Path
from scipy.fft import dct
import numpy as np


DEFAULT_BANDS = {
	"delta": (0.5, 4.0),
	"theta": (4.0, 8.0),
	"alpha": (8.0, 13.0),
	"beta": (13.0, 30.0),
	"gamma": (30.0, 45.0),
}


def load_eea(path: Path) -> np.ndarray:
	try:
		data = np.loadtxt(path)
		data = np.asarray(data, dtype=np.float64).reshape(-1)
		return data[np.isfinite(data)]
	except Exception:
		values = []
		with open(path, "r", errors="replace") as file:
			for line in file:
				for token in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line):
					try:
						values.append(float(token))
					except ValueError:
						continue
		arr = np.asarray(values, dtype=np.float64)
		return arr[np.isfinite(arr)]


def split_windows(signal: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
	if signal.size < window_size:
		return np.empty((0, window_size), dtype=np.float64)

	starts = np.arange(0, signal.size - window_size + 1, step_size)
	windows = np.stack([signal[start : start + window_size] for start in starts], axis=0)
	return windows


def _band_mask(freqs: np.ndarray, low: float, high: float) -> np.ndarray:
	return (freqs >= low) & (freqs < high)


def extract_features(window: np.ndarray, fs: float, bands: dict[str, tuple[float, float]]) -> tuple[np.ndarray, list[str]]:
	window = np.asarray(window, dtype=np.float64)
	centered = window - np.mean(window)

	energy = float(np.sum(centered**2))
	rms = float(np.sqrt(np.mean(centered**2)))
	var = float(np.var(centered))
	std = float(np.std(centered))
	zcr = float(np.mean(np.abs(np.diff(np.signbit(centered)))))
	mobility = float(np.sqrt(np.var(np.diff(centered)) / (var + 1e-12))) if centered.size > 1 else 0.0

	spectrum = np.fft.rfft(centered)
	freqs = np.fft.rfftfreq(centered.size, d=1.0 / fs)
	psd = (np.abs(spectrum) ** 2) / (centered.size * fs)

	total_power = float(np.sum(psd) + 1e-12)
	norm_psd = psd / total_power
	spectral_entropy = float(-np.sum(norm_psd * np.log(norm_psd + 1e-12)))
	spectral_centroid = float(np.sum(freqs * psd) / total_power)

	feature_names = [
		"energy",
		"rms",
		"variance",
		"std",
		"zcr",
		"mobility",
		"spectral_entropy",
		"spectral_centroid",
	]
	features = [energy, rms, var, std, zcr, mobility, spectral_entropy, spectral_centroid]

	band_powers = []
	for band_name, (low, high) in bands.items():
		mask = _band_mask(freqs, low, high)
		band_power = float(np.sum(psd[mask])) if np.any(mask) else 0.0
		rel_power = band_power / total_power
		features.extend([band_power, rel_power])
		feature_names.extend([f"psd_{band_name}", f"rel_psd_{band_name}"])
		band_powers.append(band_power)

	features.append(float(np.sum(band_powers)))
	feature_names.append("band_power_sum")

	return np.asarray(features, dtype=np.float64), feature_names


def normalize_to_angle(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	mins = np.min(features, axis=0)
	maxs = np.max(features, axis=0)
	denom = maxs - mins

	scaled = np.zeros_like(features)
	non_constant = denom > 1e-12
	scaled[:, non_constant] = (features[:, non_constant] - mins[non_constant]) / denom[non_constant]
	scaled[:, ~non_constant] = 0.5

	angular = (scaled * 2.0 * np.pi) - np.pi
	return angular, mins, maxs


def build_dataset(
	data_root: Path,
	fs: float,
	window_sec: float,
	step_sec: float,
	bands: dict[str, tuple[float, float]],
	class_dirs: list[str] | tuple[str, ...] = ("norm", "sch"),
	max_files_per_class: int | None = None,
) -> dict[str, np.ndarray]:
	class_dirs = list(class_dirs)
	class_to_id = {name: idx for idx, name in enumerate(class_dirs)}

	window_size = int(round(window_sec * fs))
	step_size = int(round(step_sec * fs))
	if window_size <= 1:
		raise ValueError("window_sec es demasiado pequeño para la frecuencia de muestreo dada.")
	if step_size <= 0:
		raise ValueError("step_sec debe producir al menos 1 muestra por salto.")

	all_features = []
	all_labels = []
	all_label_names = []
	all_file_names = []
	all_window_index = []
	feature_names = None

	for class_name in class_dirs:
		class_path = data_root / class_name
		if not class_path.exists():
			continue

		files = sorted(class_path.glob("*.eea"))
		if max_files_per_class is not None:
			files = files[:max_files_per_class]
		for file_path in files:
			signal = load_eea(file_path)
			if signal.size < window_size:
				continue

			windows = split_windows(signal, window_size=window_size, step_size=step_size)
			for w_idx, window in enumerate(windows):
				feats, names = extract_features(window, fs=fs, bands=bands)
				if feature_names is None:
					feature_names = names

				all_features.append(feats)
				all_labels.append(class_to_id[class_name])
				all_label_names.append(class_name)
				all_file_names.append(file_path.name)
				all_window_index.append(w_idx)

	if not all_features:
		raise RuntimeError("No se pudieron extraer ventanas. Verifica fs/window_sec y los archivos de entrada.")

	x_raw = np.vstack(all_features)
	y = np.asarray(all_labels, dtype=np.int64)
	y_name = np.asarray(all_label_names)
	file_names = np.asarray(all_file_names)
	window_index = np.asarray(all_window_index, dtype=np.int64)
	feature_names_arr = np.asarray(feature_names)

	x_angle, x_min, x_max = normalize_to_angle(x_raw)

	return {
		"X_raw": x_raw,
		"X_angle": x_angle,
		"y": y,
		"y_name": y_name,
		"file_name": file_names,
		"window_index": window_index,
		"feature_names": feature_names_arr,
		"feature_min": x_min,
		"feature_max": x_max,
		"class_names": np.asarray(class_dirs),
		"fs": np.asarray([fs], dtype=np.float64),
		"window_size_samples": np.asarray([window_size], dtype=np.int64),
		"step_size_samples": np.asarray([step_size], dtype=np.int64),
	}



### QTSE




def _mel_filterbank(n_filters: int, n_fft: int, fs: float) -> np.ndarray:
    """Construye un banco de filtros Mel de shape (n_filters, n_fft//2 + 1)."""
    def hz_to_mel(hz):  return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel  = hz_to_mel(0.0)
    high_mel = hz_to_mel(fs / 2.0)
    mel_pts  = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_pts   = mel_to_hz(mel_pts)
    bin_pts  = np.floor(hz_pts / (fs / n_fft)).astype(int)
    n_bins   = n_fft // 2 + 1

    filterbank = np.zeros((n_filters, n_bins))
    for m in range(1, n_filters + 1):
        f_left, f_center, f_right = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(f_left, f_center):
            filterbank[m - 1, k] = (k - f_left) / max(f_center - f_left, 1)
        for k in range(f_center, f_right):
            filterbank[m - 1, k] = (f_right - k) / max(f_right - f_center, 1)
    return filterbank


def extract_mfcc_frame(frame: np.ndarray, fs: float, n_mfcc: int, n_mels: int = 40) -> np.ndarray:
    """Calcula los coeficientes MFCC de un frame de señal."""
    windowed = frame * np.hamming(len(frame))
    n_fft = max(512, 2 ** int(np.ceil(np.log2(len(frame)))))
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2

    filterbank = _mel_filterbank(n_mels, n_fft, fs)
    mel_energy = np.dot(filterbank, spectrum)
    log_mel = np.log(mel_energy + 1e-10)

    mfcc = dct(log_mel, type=2, norm="ortho")[:n_mfcc]
    return mfcc


def preprocess_qtse(
    signal: np.ndarray,
    fs: float,
    n_mfcc: int = 13,
    n_frames: int = 16,
    n_mels: int = 40,
    scalar_mode: str = "energy",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocesa una señal 1D para el QTSE feature map.

    1. Divide la señal en n_frames frames solapados
    2. Calcula MFCC de n_mfcc dimensiones por frame
    3. Extrae un escalar por frame de los MFCCs
    4. Cuantiza los escalares a enteros [0, 15] (4-bit para QTSE)

    Args:
        signal:      señal 1D de entrada (e.g., EEG/audio)
        fs:          frecuencia de muestreo en Hz
        n_mfcc:      número de coeficientes MFCC (dimensión D)
        n_frames:    número de frames (debe ser 16 para QTSE con 4 qubits de tiempo)
        n_mels:      número de filtros Mel
        scalar_mode: cómo colapsar los D MFCCs a un escalar:
                     - "energy"  → suma de cuadrados (energía espectral)
                     - "mean"    → media de los coeficientes
                     - "first"   → primer coeficiente (energía de banda 0)

    Returns:
        t:     np.ndarray shape (n_frames,) — índices de tiempo enteros [0..n_frames-1]
        audio: np.ndarray shape (n_frames,) — escalares cuantizados [0..15]
    """
    frame_size = len(signal) // n_frames
    if frame_size == 0:
        raise ValueError(
            f"La señal ({len(signal)} muestras) es demasiado corta para {n_frames} frames."
        )

    # Paso 1: Dividir en n_frames y calcular MFCC por frame -> shape (n_frames, n_mfcc)
    mfccs = np.zeros((n_frames, n_mfcc))
    for i in range(n_frames):
        start = i * frame_size
        frame = signal[start : start + frame_size]
        mfccs[i] = extract_mfcc_frame(frame, fs=fs, n_mfcc=n_mfcc, n_mels=n_mels)

    # Paso 2: Colapsar D MFCCs a un escalar por frame
    if scalar_mode == "energy":
        scalars = np.sum(mfccs ** 2, axis=1)
    elif scalar_mode == "mean":
        scalars = np.mean(mfccs, axis=1)
    elif scalar_mode == "first":
        scalars = mfccs[:, 0]
    else:
        raise ValueError(f"scalar_mode '{scalar_mode}' no soportado. Elige 'energy', 'mean' o 'first'.")

    # Paso 3: Cuantizar a [0, 15] (enteros de 4 bits para QTSE)
    s_min, s_max = scalars.min(), scalars.max()
    if s_max - s_min < 1e-12:
        audio_quantized = np.zeros(n_frames, dtype=int)
    else:
        audio_quantized = np.round(
            (scalars - s_min) / (s_max - s_min) * 15
        ).astype(int)

    # t = índices enteros [0, 1, ..., 15]
    t = np.arange(n_frames, dtype=int)

    return t, audio_quantized


import argparse
import re
from pathlib import Path

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
) -> dict[str, np.ndarray]:
	class_dirs = ["norm", "sch"]
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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Preprocesamiento EEG con features por ventana y normalización angular.")
	parser.add_argument("--data", default="data", help="Directorio raíz que contiene subdirectorios norm/ y sch/")
	parser.add_argument("--output", default="data/processed_features.npz", help="Ruta de salida del dataset procesado")
	parser.add_argument("--fs", type=float, required=True, help="Frecuencia de muestreo en Hz")
	parser.add_argument("--window-sec", type=float, default=2.0, help="Duración de ventana en segundos")
	parser.add_argument("--step-sec", type=float, default=1.0, help="Salto entre ventanas en segundos")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	data_root = Path(args.data)
	output_path = Path(args.output)

	dataset = build_dataset(
		data_root=data_root,
		fs=args.fs,
		window_sec=args.window_sec,
		step_sec=args.step_sec,
		bands=DEFAULT_BANDS,
	)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(output_path, **dataset)

	print("Preprocesamiento completado")
	print(f"Salida: {output_path}")
	print(f"Ventanas totales: {dataset['X_raw'].shape[0]}")
	print(f"Features por ventana: {dataset['X_raw'].shape[1]}")
	unique, counts = np.unique(dataset["y_name"], return_counts=True)
	print("Distribución por clase:", {str(k): int(v) for k, v in zip(unique, counts)})
	print("Rango angular global X_angle:", float(np.min(dataset["X_angle"])), float(np.max(dataset["X_angle"])))


if __name__ == "__main__":
	main()

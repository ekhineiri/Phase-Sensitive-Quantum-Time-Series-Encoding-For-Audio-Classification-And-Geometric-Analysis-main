"""EEG preprocessing utilities.

Functions:
- `band_power_psd` : compute band power (Welch) per channel and band
- `energy_feature` : compute signal energy per channel
- `window_to_feature_vector` : convert one window to a fixed feature vector
- `normalize_features_to_angles` : scale features to [-pi, pi]
- `preprocess_dataset` : process many windows and save to disk

The code depends only on `numpy` and `scipy.signal`.
"""

from typing import Iterable, List, Sequence, Tuple, Optional

import numpy as np
from scipy.signal import welch


def band_power_psd(window: np.ndarray, sf: float, bands: Sequence[Tuple[float, float]]) -> np.ndarray:
	"""Compute band power using Welch PSD for each channel.

	Parameters:
	- window: array shape (n_channels, n_samples)
	- sf: sampling frequency (Hz)
	- bands: sequence of (low, high) frequency band tuples

	Returns:
	- array shape (n_channels, len(bands)) with power per band
	"""
	if window.ndim != 2:
		raise ValueError("window must be 2D array with shape (n_channels, n_samples)")
	n_channels, _ = window.shape
	band_powers = np.zeros((n_channels, len(bands)), dtype=float)

	for ch in range(n_channels):
		f, Pxx = welch(window[ch], fs=sf, nperseg=min(256, window.shape[1]))
		for i, (low, high) in enumerate(bands):
			mask = (f >= low) & (f <= high)
			band_powers[ch, i] = np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0

	return band_powers


def energy_feature(window: np.ndarray) -> np.ndarray:
	"""Compute energy per channel for a window.

	Parameters:
	- window: array shape (n_channels, n_samples)

	Returns:
	- array shape (n_channels,) with energy per channel
	"""
	if window.ndim != 2:
		raise ValueError("window must be 2D array with shape (n_channels, n_samples)")
	return np.sum(window ** 2, axis=1)


def window_to_feature_vector(window: np.ndarray, sf: float, bands: Sequence[Tuple[float, float]]) -> np.ndarray:
	"""Convert a single EEG window to a fixed 1D feature vector.

	Features concatenated as: [band_powers_channel0..., band_powers_channel1..., ..., energy_channel0..., energy_channel1...]

	Parameters:
	- window: (n_channels, n_samples)
	- sf: sampling frequency
	- bands: list of frequency bands

	Returns:
	- 1D numpy array feature vector
	"""
	bp = band_power_psd(window, sf, bands)  # shape (n_channels, n_bands)
	en = energy_feature(window)  # shape (n_channels,)
	return np.concatenate([bp.ravel(order="C"), en])


def normalize_features_to_angles(features: np.ndarray, vmin: Optional[np.ndarray] = None, vmax: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Normalize features to angular range [-pi, pi].

	If `vmin`/`vmax` are not provided, they are computed per-feature across the rows.

	Mapping is linear: x -> ( (x - vmin) / (vmax - vmin) ) * 2π - π

	Parameters:
	- features: shape (n_windows, n_features)
	- vmin: optional array shape (n_features,) of minima
	- vmax: optional array shape (n_features,) of maxima

	Returns:
	- angles: normalized features in [-pi, pi]
	- vmin: used minima
	- vmax: used maxima
	"""
	features = np.asarray(features, dtype=float)
	if features.ndim != 2:
		raise ValueError("features must be 2D array (n_windows, n_features)")

	if vmin is None:
		vmin = np.min(features, axis=0)
	else:
		vmin = np.asarray(vmin, dtype=float)
	if vmax is None:
		vmax = np.max(features, axis=0)
	else:
		vmax = np.asarray(vmax, dtype=float)

	# prevent division by zero
	span = vmax - vmin
	span[span == 0] = 1.0

	scaled = (features - vmin) / span
	angles = scaled * (2.0 * np.pi) - np.pi
	return angles, vmin, vmax


def preprocess_dataset(windows: Iterable[np.ndarray], sf: float, bands: Sequence[Tuple[float, float]] = ((1,4),(4,8),(8,12),(12,30),(30,45)), out_file: str = "data/processed_features.npz") -> dict:
	"""Preprocess many windows and save normalized angular features.

	Parameters:
	- windows: iterable of windows, each of shape (n_channels, n_samples)
	- sf: sampling frequency
	- bands: frequency bands to extract (default: delta, theta, alpha, beta, low-gamma)
	- out_file: file path to save compressed features and normalization params

	Returns:
	- dict with keys: `angles`, `vmin`, `vmax`, `bands`
	"""
	feats = []
	for w in windows:
		feats.append(window_to_feature_vector(np.asarray(w), sf, bands))
	feats = np.vstack(feats)  # shape (n_windows, n_features)

	angles, vmin, vmax = normalize_features_to_angles(feats)

	# ensure output directory exists
	import os
	os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

	np.savez_compressed(out_file, angles=angles, vmin=vmin, vmax=vmax, bands=np.array(bands, dtype=float))

	return {"angles": angles, "vmin": vmin, "vmax": vmax, "bands": bands}


if __name__ == "__main__":
	# quick self-test example
	sf = 128.0
	n_channels = 4
	n_samples = 256
	n_windows = 10
	rng = np.random.default_rng(42)
	windows = [rng.standard_normal((n_channels, n_samples)) for _ in range(n_windows)]
	out = preprocess_dataset(windows, sf)
	print("Processed", out["angles"].shape, "-> saved to data/processed_features.npz")


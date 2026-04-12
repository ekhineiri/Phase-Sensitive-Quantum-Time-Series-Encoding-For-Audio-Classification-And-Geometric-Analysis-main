import numpy as np
import librosa
from aeon.datasets import load_gunpoint
import numpy as np

def load_aeon_gunpoint():

	X_train, y_train = load_gunpoint(split="train")
	X_test, y_test = load_gunpoint(split="test")

	# quitar canal
	X_train = X_train[:, 0, :]
	X_test = X_test[:, 0, :]

	# labels numéricas
	classes = np.unique(y_train)
	mapping = {c:i for i,c in enumerate(classes)}

	y_train = np.array([mapping[v] for v in y_train])
	y_test = np.array([mapping[v] for v in y_test])

	return X_train, y_train, X_test, y_test


def preprocess(X, sr, n_frames=16, n_mfcc=13):

    X_out = []

    for signal in X:
        signal = np.asarray(signal, dtype=float)

        # MFCC -> (n_mfcc, T)
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=n_mfcc,
            hop_length=max(1, len(signal) // n_frames),
            n_fft=min(2048, len(signal))
        )

        # Queremos exactamente 16 frames
        if mfcc.shape[1] < n_frames:
            pad = n_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
        else:
            mfcc = mfcc[:, :n_frames]

        # 1 scalar por frame = media de coeficientes
        vec = mfcc.mean(axis=0)   # longitud 16

        # normalizar a [0,15]
        mn, mx = vec.min(), vec.max()

        if mx - mn < 1e-12:
            vec_q = np.zeros(n_frames, dtype=int)
        else:
            vec_norm = (vec - mn) / (mx - mn)
            vec_q = np.floor(vec_norm * 15).astype(int)

        X_out.append(vec_q)

    return np.asarray(X_out)



def preprocess_phase(X, sr, n_frames=16):

    X_out = []

    for signal in X:
        signal = np.asarray(signal, dtype=float)

        # STFT compleja
        stft = librosa.stft(
            y=signal,
            n_fft=min(2048, len(signal)),
            hop_length=max(1, len(signal) // n_frames)
        )

        # Fase
        phase = np.angle(stft)   # (freq_bins, T)

        # Media por frame
        vec = phase.mean(axis=0)

        # Queremos exactamente 16 frames
        if len(vec) < n_frames:
            vec = np.pad(vec, (0, n_frames - len(vec)))
        else:
            vec = vec[:n_frames]

        # Normalizar a [0,15]
        mn, mx = vec.min(), vec.max()

        if mx - mn < 1e-12:
            vec_q = np.zeros(n_frames, dtype=int)
        else:
            vec_norm = (vec - mn) / (mx - mn)
            vec_q = np.floor(vec_norm * 15).astype(int)

        X_out.append(vec_q)

    return np.asarray(X_out)
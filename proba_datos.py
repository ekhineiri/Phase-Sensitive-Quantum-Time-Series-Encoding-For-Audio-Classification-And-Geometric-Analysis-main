import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random
import re

def load_eea(path):
    try:
        return np.loadtxt(path)
    except Exception:
        vals = []
        with open(path, "r", errors="replace") as f:
            for line in f:
                for tok in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line):
                    try:
                        vals.append(float(tok))
                    except:
                        pass
        return np.array(vals)

def load_class(dirpath):
    files = sorted(Path(dirpath).glob("*.eea"))
    sigs = [load_eea(f) for f in files]
    return files, sigs

def plot_comparison(norm_files, norm_sigs, sch_files, sch_sigs, n_samples, fs):
    random.seed(0)
    ns = min(n_samples, len(norm_sigs), len(sch_sigs))
    norm_idx = random.sample(range(len(norm_sigs)), ns)
    sch_idx  = random.sample(range(len(sch_sigs)), ns)

    # Overlay mean +/- std (truncate to common min length)
    min_len = min(min((s.size for s in norm_sigs), default=0),
                  min((s.size for s in sch_sigs), default=0))
    if min_len == 0:
        raise SystemExit("No signals found or empty files.")
    t = (np.arange(min_len) / fs) if fs else np.arange(min_len)

    norm_mat = np.vstack([s[:min_len] for s in norm_sigs])
    sch_mat  = np.vstack([s[:min_len] for s in sch_sigs])
    norm_mean, norm_std = np.nanmean(norm_mat, axis=0), np.nanstd(norm_mat, axis=0)
    sch_mean,  sch_std  = np.nanmean(sch_mat, axis=0),  np.nanstd(sch_mat, axis=0)

    plt.figure(figsize=(12, 6 + ns*1.5))
    ax1 = plt.subplot2grid((2+ns, 2), (0,0), colspan=2)
    ax1.fill_between(t, norm_mean-norm_std, norm_mean+norm_std, color='C0', alpha=0.2)
    ax1.plot(t, norm_mean, label='norm mean', color='C0')
    ax1.fill_between(t, sch_mean-sch_std, sch_mean+sch_std, color='C1', alpha=0.2)
    ax1.plot(t, sch_mean, label='sch mean', color='C1')
    ax1.set_title("Mean ± std (truncated to common length)")
    ax1.set_xlabel("Time (s)" if fs else "Sample index")
    ax1.legend()

    # Plot sample pairs
    for i in range(ns):
        axn = plt.subplot2grid((2+ns,2), (1+i,0))
        axs = plt.subplot2grid((2+ns,2), (1+i,1))
        s_n = norm_sigs[norm_idx[i]]
        s_s = sch_sigs[sch_idx[i]]
        tn = (np.arange(s_n.size)/fs) if fs else np.arange(s_n.size)
        ts = (np.arange(s_s.size)/fs) if fs else np.arange(s_s.size)
        axn.plot(tn, s_n, color='C0', lw=0.8); axn.set_title(f"norm: {Path(norm_files[norm_idx[i]]).name}"); axn.set_xlabel('')
        axs.plot(ts, s_s, color='C1', lw=0.8); axs.set_title(f"sch: {Path(sch_files[sch_idx[i]]).name}"); axs.set_xlabel('')
    plt.tight_layout()
    plt.show()

def main():
    p = argparse.ArgumentParser(description="Comparar señales de data/norm y data/sch")
    p.add_argument("--data", default="data", help="directorio raíz con subdirs norm/ y sch/")
    p.add_argument("--n-samples", type=int, default=6, help="número de pares de muestras a mostrar")
    p.add_argument("--fs", type=float, default=0.0, help="frecuencia de muestreo (Hz). 0 = índice de muestra")
    args = p.parse_args()

    base = Path(args.data)
    norm_files, norm_sigs = load_class(base / "norm")
    sch_files,  sch_sigs  = load_class(base / "sch")
    fs = args.fs if args.fs > 0 else None

    plot_comparison(norm_files, norm_sigs, sch_files, sch_sigs, args.n_samples, fs)

if __name__ == "__main__":
    main()
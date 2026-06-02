"""
evaluation.py
=============
Evaluation metrics for music reconstruction from EEG, following Daly (2023).

Metrics
-------
1. Pearson correlation in the time domain  (r_time)
2. Pearson correlation in the frequency domain (r_freq)  — correlation of
   power spectra computed from FFT.
3. SSIM of time-frequency spectrograms  (ssim)
4. Rank accuracy  (rank_acc)  — for each trial, rank similarity of the
   decoded signal vs. all N original stimuli; report mean rank percentage
   and significance via bootstrap (4000 shuffles).

All metrics return the score and a p-value.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, spectrogram
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bandpass_for_eval(signal: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Band-pass 0.035–4.75 Hz as specified in Daly (2023) for trial-level analysis."""
    b, a = butter(4, [0.035, 4.75], btype="bandpass", fs=fs)
    min_len = 3 * max(len(a), len(b))
    if len(signal) <= min_len:
        return signal
    return filtfilt(b, a, signal).astype(np.float32)


def _safe_pearsonr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Pearson r with a guard for zero-variance arrays."""
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0, 1.0
    r, p = pearsonr(a.ravel(), b.ravel())
    return float(r), float(p)


def _power_spectrum(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided power spectrum of *signal* via FFT.

    Parameters
    ----------
    signal : np.ndarray, 1-D
    fs : float
        Sample rate (Hz).

    Returns
    -------
    freqs : np.ndarray
    power : np.ndarray
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(signal)) ** 2
    return freqs, power


def _compute_spectrogram(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: int = 128,
) -> np.ndarray:
    """Compute short-time Fourier transform magnitude spectrogram.

    Parameters
    ----------
    signal : np.ndarray, 1-D
    fs : float
    nperseg : int
    noverlap : int

    Returns
    -------
    S : np.ndarray, shape (freq_bins, time_frames)
        Log-magnitude spectrogram.
    """
    _, _, Zxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    S = 10.0 * np.log10(np.abs(Zxx) + 1e-10)
    return S


# ---------------------------------------------------------------------------
# Per-trial metrics
# ---------------------------------------------------------------------------

def time_domain_correlation(
    predicted: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float]:
    """Pearson r between predicted and target waveforms.

    Parameters
    ----------
    predicted : np.ndarray, 1-D
    target : np.ndarray, 1-D

    Returns
    -------
    r : float
    p : float
    """
    return _safe_pearsonr(predicted, target)


def frequency_domain_correlation(
    predicted: np.ndarray,
    target: np.ndarray,
    fs: float = 100.0,
) -> Tuple[float, float]:
    """Pearson r between power spectra of predicted and target.

    Parameters
    ----------
    predicted : np.ndarray, 1-D
    target : np.ndarray, 1-D
    fs : float

    Returns
    -------
    r : float
    p : float
    """
    _, power_pred = _power_spectrum(predicted, fs)
    _, power_tgt = _power_spectrum(target, fs)
    return _safe_pearsonr(power_pred, power_tgt)


def spectrogram_ssim(
    predicted: np.ndarray,
    target: np.ndarray,
    fs: float = 100.0,
    nperseg: int = 64,
    noverlap: int = 32,
) -> float:
    """SSIM between spectrograms of predicted and target.

    Parameters
    ----------
    predicted : np.ndarray, 1-D
    target : np.ndarray, 1-D
    fs : float
    nperseg : int
    noverlap : int

    Returns
    -------
    ssim_val : float
        SSIM in [−1, 1]; 1 = perfect match.
    """
    S_pred = _compute_spectrogram(predicted, fs, nperseg=nperseg, noverlap=noverlap)
    S_tgt = _compute_spectrogram(target, fs, nperseg=nperseg, noverlap=noverlap)

    # Normalise to [0, 1]
    vmin = min(S_pred.min(), S_tgt.min())
    vmax = max(S_pred.max(), S_tgt.max())
    rng = vmax - vmin
    if rng < 1e-10:
        return 0.0

    S_pred_n = (S_pred - vmin) / rng
    S_tgt_n = (S_tgt - vmin) / rng

    val = ssim(S_pred_n, S_tgt_n, data_range=1.0)
    return float(val)


# ---------------------------------------------------------------------------
# Rank accuracy
# ---------------------------------------------------------------------------

def rank_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    metric: str = "correlation",
) -> Tuple[float, np.ndarray]:
    """Compute the rank accuracy across trials.

    For each trial i, rank the similarity between the predicted signal and all
    N target signals.  The rank of the correct target is recorded.  Mean rank
    accuracy (as percentage) = mean of (N − rank + 1) / N × 100.

    Parameters
    ----------
    predictions : np.ndarray, shape (N, T)
        Decoded signals for N trials.
    targets : np.ndarray, shape (N, T)
        Ground-truth stimuli for N trials.
    metric : str
        Similarity metric: ``'correlation'`` or ``'mse'``.

    Returns
    -------
    mean_rank_acc : float
        Mean rank accuracy (%).
    ranks : np.ndarray, shape (N,)
        Rank of the correct target for each trial (1 = best).
    """
    n_trials = len(predictions)
    ranks = np.zeros(n_trials, dtype=int)

    for i in range(n_trials):
        pred_i = predictions[i]
        similarities = np.zeros(n_trials)
        for j in range(n_trials):
            if metric == "correlation":
                r, _ = _safe_pearsonr(pred_i, targets[j])
                similarities[j] = r
            elif metric == "mse":
                similarities[j] = -np.mean((pred_i - targets[j]) ** 2)
            else:
                raise ValueError(f"Unknown metric: {metric!r}")

        # Rank in descending order (highest similarity = rank 1)
        sorted_idx = np.argsort(similarities)[::-1]
        rank_i = int(np.where(sorted_idx == i)[0][0]) + 1  # 1-indexed
        ranks[i] = rank_i

    # Rank accuracy: proportion of trials where correct stimulus ranks in top k
    # Paper uses mean rank percentage
    mean_rank_acc = np.mean((n_trials - ranks + 1) / n_trials) * 100.0
    return float(mean_rank_acc), ranks


def bootstrap_rank_significance(
    predictions: np.ndarray,
    targets: np.ndarray,
    observed_rank_acc: float,
    n_shuffles: int = 4000,
    metric: str = "correlation",
    rng_seed: int = 42,
) -> Tuple[float, np.ndarray]:
    """Bootstrap null distribution for rank accuracy.

    Randomly shuffles the correspondence between predictions and targets
    *n_shuffles* times to build a null distribution, then computes the
    one-tailed p-value.

    Parameters
    ----------
    predictions : np.ndarray, shape (N, T)
    targets : np.ndarray, shape (N, T)
    observed_rank_acc : float
        The observed rank accuracy (%) to test against.
    n_shuffles : int
        Number of permutations (paper uses 4000).
    metric : str
    rng_seed : int

    Returns
    -------
    p_value : float
    null_dist : np.ndarray, shape (n_shuffles,)
    """
    rng = np.random.default_rng(rng_seed)
    n_trials = len(predictions)
    null_dist = np.zeros(n_shuffles)

    for s in range(n_shuffles):
        perm = rng.permutation(n_trials)
        shuffled_preds = predictions[perm]
        acc, _ = rank_accuracy(shuffled_preds, targets, metric=metric)
        null_dist[s] = acc

    # One-tailed p-value: proportion of null values ≥ observed
    p_value = float(np.mean(null_dist >= observed_rank_acc))
    return p_value, null_dist


# ---------------------------------------------------------------------------
# Aggregate evaluation across folds/subjects
# ---------------------------------------------------------------------------

def evaluate_trial(
    predicted: np.ndarray,
    target: np.ndarray,
    fs: float = 100.0,
) -> Dict[str, float]:
    """Compute all single-trial metrics.

    Parameters
    ----------
    predicted : np.ndarray, 1-D
    target : np.ndarray, 1-D
    fs : float

    Returns
    -------
    metrics : dict with keys 'r_time', 'p_time', 'r_freq', 'p_freq', 'ssim'
    """
    # Apply 0.035-4.75 Hz band-pass before correlation (Daly 2023)
    pred_bp = _bandpass_for_eval(predicted, fs=fs)
    tgt_bp = _bandpass_for_eval(target, fs=fs)
    r_t, p_t = time_domain_correlation(pred_bp, tgt_bp)
    r_f, p_f = frequency_domain_correlation(pred_bp, tgt_bp, fs=fs)
    ssim_val = spectrogram_ssim(pred_bp, tgt_bp, fs=fs)
    return {
        "r_time": r_t,
        "p_time": p_t,
        "r_freq": r_f,
        "p_freq": p_f,
        "ssim": ssim_val,
    }


def evaluate_all(
    predictions: np.ndarray,
    targets: np.ndarray,
    fs: float = 100.0,
    n_bootstrap: int = 4000,
    rank_metric: str = "correlation",
    verbose: bool = True,
) -> Dict:
    """Run the full evaluation suite on N trials.

    Parameters
    ----------
    predictions : np.ndarray, shape (N, T)
    targets : np.ndarray, shape (N, T)
    fs : float
    n_bootstrap : int
    rank_metric : str
    verbose : bool

    Returns
    -------
    results : dict with keys:
        'r_time_mean', 'r_time_std', 'r_time_p',
        'r_freq_mean', 'r_freq_std', 'r_freq_p',
        'ssim_mean', 'ssim_std',
        'rank_acc', 'rank_acc_p', 'ranks', 'null_dist'
    """
    n_trials = len(predictions)
    r_times, p_times, r_freqs, p_freqs, ssims = [], [], [], [], []

    for i in range(n_trials):
        m = evaluate_trial(predictions[i], targets[i], fs=fs)
        r_times.append(m["r_time"])
        p_times.append(m["p_time"])
        r_freqs.append(m["r_freq"])
        p_freqs.append(m["p_freq"])
        ssims.append(m["ssim"])

    r_times = np.array(r_times)
    r_freqs = np.array(r_freqs)
    ssims = np.array(ssims)

    # Aggregate Pearson r significance: Fisher z-transform then t-test
    from scipy.stats import ttest_1samp
    _, p_r_time = ttest_1samp(r_times, 0.0)
    _, p_r_freq = ttest_1samp(r_freqs, 0.0)

    # Rank accuracy
    rank_acc, ranks = rank_accuracy(predictions, targets, metric=rank_metric)
    rank_p, null_dist = bootstrap_rank_significance(
        predictions, targets, rank_acc, n_shuffles=n_bootstrap, metric=rank_metric
    )

    results = {
        "r_time_mean": float(np.mean(r_times)),
        "r_time_std": float(np.std(r_times)),
        "r_time_p": float(p_r_time),
        "r_freq_mean": float(np.mean(r_freqs)),
        "r_freq_std": float(np.std(r_freqs)),
        "r_freq_p": float(p_r_freq),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims)),
        "rank_acc": rank_acc,
        "rank_acc_p": rank_p,
        "ranks": ranks,
        "null_dist": null_dist,
    }

    if verbose:
        print(f"r_time   = {results['r_time_mean']:.4f} ± {results['r_time_std']:.4f}"
              f"  (p = {results['r_time_p']:.4f})")
        print(f"r_freq   = {results['r_freq_mean']:.4f} ± {results['r_freq_std']:.4f}"
              f"  (p = {results['r_freq_p']:.4f})")
        print(f"SSIM     = {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
        print(f"Rank acc = {results['rank_acc']:.2f}%  (p = {results['rank_acc_p']:.4f})")

    return results


def print_results_table(results: Dict) -> None:
    """Print a formatted summary table of evaluation results."""
    print("\n" + "=" * 60)
    print("  Metric           Mean ± SD            p-value")
    print("=" * 60)
    print(f"  r_time        {results['r_time_mean']:+.4f} ± {results['r_time_std']:.4f}"
          f"      {results['r_time_p']:.4e}")
    print(f"  r_freq        {results['r_freq_mean']:+.4f} ± {results['r_freq_std']:.4f}"
          f"      {results['r_freq_p']:.4e}")
    print(f"  SSIM          {results['ssim_mean']:+.4f} ± {results['ssim_std']:.4f}")
    print(f"  Rank accuracy {results['rank_acc']:.2f}%"
          f"                   {results['rank_acc_p']:.4e}")
    print("=" * 60)

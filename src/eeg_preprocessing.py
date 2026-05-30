"""
eeg_preprocessing.py
====================
EEG preprocessing pipeline following Daly (2023):

1. Pick EEG channels only (drop EOG, EMG, misc).
2. Set average reference.
3. Bandpass filter (1–40 Hz).
4. Resample to 1000 Hz.
5. ICA-based artefact removal (FastICA or SOBI via MNE).
6. Return cleaned Raw and fitted ICA object.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import mne
import numpy as np
from mne.preprocessing import ICA


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess_eeg(
    raw: mne.io.Raw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    target_sfreq: float = 1000.0,
    n_ica_components: Optional[int] = None,
    ica_method: str = "fastica",
    random_state: int = 42,
    reject_by_annotation: bool = True,
    verbose: bool = False,
) -> Tuple[mne.io.Raw, ICA]:
    """Full EEG preprocessing pipeline.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data (will be copied, not modified in-place).
    l_freq : float
        High-pass cutoff frequency (Hz).
    h_freq : float
        Low-pass cutoff frequency (Hz).
    target_sfreq : float
        Target sample rate after resampling (Hz).
    n_ica_components : int or None
        Number of ICA components. If None, uses ``min(n_channels - 1, 25)``.
    ica_method : str
        ``'fastica'`` or ``'infomax'`` or ``'picard'``.
    random_state : int
    reject_by_annotation : bool
        Whether to respect bad-segment annotations during ICA fitting.
    verbose : bool

    Returns
    -------
    raw_clean : mne.io.Raw
        Cleaned EEG data at *target_sfreq* Hz.
    ica : mne.preprocessing.ICA
        Fitted ICA decomposition.
    """
    raw = raw.copy()

    # 1. Pick EEG channels
    raw.pick_types(eeg=True, eog=False, emg=False, stim=False, exclude="bads")
    n_channels = len(raw.ch_names)

    # 2. Set average reference
    raw.set_eeg_reference("average", projection=True, verbose=verbose)
    raw.apply_proj()

    # 3. Bandpass filter
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", fir_window="hamming",
               verbose=verbose)

    # 4. Resample
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=verbose)

    # 5. ICA
    if n_ica_components is None:
        n_ica_components = min(n_channels - 1, 25)

    ica = ICA(
        n_components=n_ica_components,
        method=ica_method,
        random_state=random_state,
        max_iter="auto",
        fit_params=None,
        verbose=verbose,
    )

    # Fit ICA (use a 1 Hz high-pass copy to improve convergence)
    raw_for_fit = raw.copy().filter(1.0, None, verbose=verbose)
    ica.fit(raw_for_fit, reject_by_annotation=reject_by_annotation, verbose=verbose)
    del raw_for_fit

    # Auto-detect eye-movement / heartbeat artefacts
    excluded = _auto_detect_artefact_components(raw, ica, verbose=verbose)
    ica.exclude = excluded

    # Apply ICA (subtract artefact components)
    raw_clean = ica.apply(raw.copy(), verbose=verbose)

    return raw_clean, ica


# ---------------------------------------------------------------------------
# Artefact detection helpers
# ---------------------------------------------------------------------------

def _auto_detect_artefact_components(
    raw: mne.io.Raw,
    ica: ICA,
    eog_threshold: float = 3.0,
    ecg_threshold: float = 3.0,
    verbose: bool = False,
) -> List[int]:
    """Automatically detect EOG and ECG artefact ICA components.

    Uses MNE's built-in correlation-based methods when EOG/ECG channels are
    present; falls back to variance-based heuristics otherwise.

    Parameters
    ----------
    raw : mne.io.Raw
    ica : ICA
    eog_threshold : float
        Z-score threshold for EOG artefact detection.
    ecg_threshold : float
        Z-score threshold for ECG artefact detection.
    verbose : bool

    Returns
    -------
    exclude : list of int
        Component indices to exclude.
    """
    exclude: List[int] = []

    # EOG
    eog_chs = mne.pick_types(raw.info, meg=False, eog=True)
    if len(eog_chs) > 0:
        try:
            eog_idx, _ = ica.find_bads_eog(
                raw, threshold=eog_threshold, verbose=verbose
            )
            exclude.extend(eog_idx)
        except Exception as e:
            warnings.warn(f"EOG detection failed: {e}")
    else:
        # Heuristic: components with high high-frequency variance
        exclude.extend(_variance_heuristic(ica, raw, n_components=2))

    # ECG
    ecg_chs = mne.pick_types(raw.info, meg=False, ecg=True)
    if len(ecg_chs) > 0:
        try:
            ecg_idx, _ = ica.find_bads_ecg(
                raw, threshold=ecg_threshold, verbose=verbose
            )
            exclude.extend(ecg_idx)
        except Exception as e:
            warnings.warn(f"ECG detection failed: {e}")

    return list(set(exclude))


def _variance_heuristic(
    ica: ICA,
    raw: mne.io.Raw,
    n_components: int = 2,
) -> List[int]:
    """Return ICA component indices with the highest absolute max amplitude.

    Used as a fallback when no EOG channels are available.
    """
    sources = ica.get_sources(raw).get_data()  # (n_components, n_times)
    amplitudes = np.max(np.abs(sources), axis=1)
    worst = np.argsort(amplitudes)[::-1][:n_components]
    return list(worst)


# ---------------------------------------------------------------------------
# Additional utilities
# ---------------------------------------------------------------------------

def apply_ica(
    raw: mne.io.Raw,
    ica: ICA,
    exclude: Optional[List[int]] = None,
    verbose: bool = False,
) -> mne.io.Raw:
    """Apply a previously fitted ICA to new raw data.

    Parameters
    ----------
    raw : mne.io.Raw
    ica : ICA
        Fitted ICA object.
    exclude : list of int or None
        Component indices to exclude. If None, uses ``ica.exclude``.
    verbose : bool

    Returns
    -------
    raw_clean : mne.io.Raw
    """
    if exclude is not None:
        ica.exclude = exclude
    return ica.apply(raw.copy(), verbose=verbose)


def bandpass_filter(
    raw: mne.io.Raw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    verbose: bool = False,
) -> mne.io.Raw:
    """Apply a zero-phase FIR bandpass filter.

    Parameters
    ----------
    raw : mne.io.Raw
    l_freq, h_freq : float
    verbose : bool

    Returns
    -------
    raw_filtered : mne.io.Raw
    """
    return raw.copy().filter(l_freq=l_freq, h_freq=h_freq,
                             method="fir", fir_window="hamming",
                             verbose=verbose)


def resample_eeg(
    raw: mne.io.Raw,
    target_sfreq: float = 1000.0,
    verbose: bool = False,
) -> mne.io.Raw:
    """Resample EEG to *target_sfreq* Hz.

    Parameters
    ----------
    raw : mne.io.Raw
    target_sfreq : float
    verbose : bool

    Returns
    -------
    raw_resampled : mne.io.Raw
    """
    if raw.info["sfreq"] == target_sfreq:
        return raw.copy()
    return raw.copy().resample(target_sfreq, verbose=verbose)


def preprocess_epochs(
    epochs: mne.Epochs,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    target_sfreq: float = 1000.0,
    n_ica_components: Optional[int] = None,
    ica_method: str = "fastica",
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[mne.Epochs, ICA]:
    """Preprocess epoched EEG.

    Converts epochs to a temporary Raw, preprocesses, then re-epochs.
    In practice it is more efficient to preprocess the continuous Raw before
    epoching; this function is provided for convenience.

    Parameters
    ----------
    epochs : mne.Epochs
    l_freq, h_freq, target_sfreq, n_ica_components, ica_method, random_state
    verbose : bool

    Returns
    -------
    epochs_clean : mne.Epochs
    ica : ICA
    """
    # Fit ICA on epochs data directly
    n_channels = len(epochs.ch_names)
    if n_ica_components is None:
        n_ica_components = min(n_channels - 1, 25)

    ica = ICA(
        n_components=n_ica_components,
        method=ica_method,
        random_state=random_state,
        max_iter="auto",
        verbose=verbose,
    )
    # Filter epochs first
    epochs_filtered = epochs.copy().filter(
        l_freq=l_freq, h_freq=h_freq, verbose=verbose
    )
    ica.fit(epochs_filtered, verbose=verbose)

    # Auto-exclude
    eog_idx, _ = ica.find_bads_eog(epochs_filtered, verbose=verbose) if \
        len(mne.pick_types(epochs.info, meg=False, eog=True)) > 0 else ([], None)
    ica.exclude = list(set(eog_idx))

    epochs_clean = ica.apply(epochs_filtered.copy(), verbose=verbose)

    # Resample
    if epochs_clean.info["sfreq"] != target_sfreq:
        epochs_clean = epochs_clean.resample(target_sfreq, verbose=verbose)

    return epochs_clean, ica

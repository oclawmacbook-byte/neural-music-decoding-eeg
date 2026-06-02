"""
pipeline.py
===========
Main analysis pipeline orchestrating all steps with 3×3 cross-fold validation,
following Daly (2023).

Cross-validation scheme
-----------------------
The dataset has 3 runs per subject.  The paper uses a leave-one-run-out
(3-fold) cross-validation:

    Fold 1: train on runs 2 & 3, test on run 1
    Fold 2: train on runs 1 & 3, test on run 2
    Fold 3: train on runs 1 & 2, test on run 3

Each fold trains one biLSTM per subject (or optionally pooled across subjects).
The "3×3" refers to 3 subjects × 3 folds in the compact experiment reported in
the paper, but the code supports arbitrary numbers of subjects.

Feature matrix shape: (n_trials, 124, n_samples_100Hz)
Audio target shape:   (n_trials, n_samples_100Hz)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.signal import hilbert

from .data_loader import (
    list_subjects,
    load_subject_runs,
    load_events_for_run,
    load_audio_stimuli,
    epoch_eeg_by_stimulus,
)
from .eeg_preprocessing import preprocess_eeg
from .source_localization import (
    build_forward,
    extract_features_from_epochs,
)
from .models import BiLSTMDecoder, train_model, predict, save_model, load_model
from .evaluation import evaluate_all, print_results_table


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_RUNS = 3
AUDIO_SR = 1000      # Hz — stimulus loaded at this rate
DECODING_SR = 100    # Hz — target for model (downsample ×10)
DECIM_FACTOR = int(AUDIO_SR / DECODING_SR)  # = 10

MODEL_INPUT_SIZE = 124
MODEL_HIDDEN_SIZE = 250
MODEL_NUM_LAYERS = 4

TRAIN_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3


# ---------------------------------------------------------------------------
# Audio preparation helpers
# ---------------------------------------------------------------------------

def compute_audio_envelope(audio: np.ndarray) -> np.ndarray:
    """Compute the amplitude envelope via Hilbert transform (Daly 2023).

    The paper targets the audio *envelope*, not the raw waveform.
    """
    return np.abs(hilbert(audio)).astype(np.float32)


def prepare_audio_target(
    audio_waveform: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Trim or zero-pad audio to *n_samples* and normalise to [−1, 1].

    Parameters
    ----------
    audio_waveform : np.ndarray, 1-D  (at DECODING_SR Hz)
    n_samples : int

    Returns
    -------
    audio : np.ndarray, 1-D, length n_samples
    """
    if len(audio_waveform) >= n_samples:
        audio = audio_waveform[:n_samples]
    else:
        audio = np.pad(audio_waveform, (0, n_samples - len(audio_waveform)))

    # Normalise
    m = np.max(np.abs(audio))
    if m > 1e-8:
        audio = audio / m
    return audio.astype(np.float32)


def downsample_audio(
    audio: np.ndarray,
    factor: int = DECIM_FACTOR,
) -> np.ndarray:
    """Simple downsampling by integer factor (anti-alias not applied here;
    the audio was already loaded at AUDIO_SR which serves as the source)."""
    return audio[::factor]


# ---------------------------------------------------------------------------
# Single-subject feature + target extraction
# ---------------------------------------------------------------------------

def extract_subject_data(
    subject_dir: Path,
    dipole_coords_mni: np.ndarray,
    stimuli: Dict[str, np.ndarray],
    run_ids: List[int],
    fmt: str = "auto",
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract feature matrices and audio targets for one subject / run set.

    Parameters
    ----------
    subject_dir : Path
    dipole_coords_mni : np.ndarray, shape (4, 3)  — MNI mm
    stimuli : dict  — ``{stem: waveform at AUDIO_SR Hz}``
    run_ids : list of int  — which runs to include
    fmt : str  — EEG format
    verbose : bool

    Returns
    -------
    X : np.ndarray, shape (n_trials, n_time, 124)
        Feature matrices, with time as the second axis (biLSTM input).
    y : np.ndarray, shape (n_trials, n_time)
        Corresponding audio targets at DECODING_SR Hz.
    trial_names : list of str
        Stimulus file names for each trial.
    """
    X_list, y_list, names = [], [], []

    for run_id in run_ids:
        # Load and preprocess EEG
        try:
            from .data_loader import load_eeg
            raw = load_eeg(subject_dir, run_id, fmt=fmt)
        except FileNotFoundError as e:
            warnings.warn(str(e))
            continue

        raw_clean, ica = preprocess_eeg(raw, verbose=verbose)
        events_df = load_events_for_run(subject_dir, run_id)

        if events_df.empty:
            warnings.warn(f"No events for {subject_dir.name} run {run_id}; skipping.")
            continue

        # Epoch per music trial
        try:
            epochs = epoch_eeg_by_stimulus(raw_clean, events_df, tmin=0.0, tmax=40.0)
        except Exception as e:
            warnings.warn(f"Epoching failed: {e}")
            continue

        # Build forward model (BEM if available, else spherical)
        fwd = build_forward(raw_clean.info, dipole_coords_mni)

        # Extract features (returns shape: n_epochs, 124, n_time_100Hz)
        feats = extract_features_from_epochs(
            epochs, fwd, target_sfreq=float(DECODING_SR), verbose=verbose
        )
        # Transpose to (n_epochs, n_time, 124) for LSTM
        feats = np.transpose(feats, (0, 2, 1))

        # Match audio targets
        stim_file_col = "stim_file" if "stim_file" in events_df.columns else None
        music_events = events_df[events_df["trial_type"] == "music"].reset_index(drop=True)

        for i in range(feats.shape[0]):
            n_time = feats.shape[1]

            # Get stimulus name
            stim_name = None
            if stim_file_col and i < len(music_events):
                stim_path = str(music_events.loc[i, stim_file_col])
                stim_name = Path(stim_path).stem

            if stim_name and stim_name in stimuli:
                audio_raw = stimuli[stim_name]
            elif len(stimuli) > 0:
                # Fall back to cycling through available stimuli
                idx = (len(X_list)) % len(stimuli)
                stim_name = list(stimuli.keys())[idx]
                audio_raw = stimuli[stim_name]
            else:
                # No audio found; use zeros
                audio_raw = np.zeros(n_time, dtype=np.float32)
                stim_name = f"unknown_{len(X_list)}"

            # Compute envelope then downsample to DECODING_SR (Daly 2023)
            audio_env = compute_audio_envelope(audio_raw)
            audio_dec = downsample_audio(audio_env, factor=DECIM_FACTOR)
            audio_tgt = prepare_audio_target(audio_dec, n_time)

            X_list.append(feats[i])         # (n_time, 124)
            y_list.append(audio_tgt)        # (n_time,)
            names.append(stim_name or "")

    if not X_list:
        return np.empty((0, 0, 124)), np.empty((0, 0)), []

    X = np.stack(X_list, axis=0)   # (n_trials, n_time, 124)
    y = np.stack(y_list, axis=0)   # (n_trials, n_time)
    return X, y, names


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_subject(
    subject_dir: Path,
    dipole_coords_mni: np.ndarray,
    stimuli: Dict[str, np.ndarray],
    n_runs: int = N_RUNS,
    fmt: str = "auto",
    n_train_epochs: int = TRAIN_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict:
    """Leave-one-run-out cross-validation for one subject.

    Parameters
    ----------
    subject_dir : Path
    dipole_coords_mni : np.ndarray
    stimuli : dict
    n_runs : int
    fmt : str
    n_train_epochs : int
    batch_size : int
    lr : float
    device : str or None
    output_dir : Path or None
        If provided, save per-fold model weights and predictions here.
    verbose : bool

    Returns
    -------
    fold_results : dict with keys ``fold_1``, ``fold_2``, ``fold_3`` (metrics dicts)
        and ``mean`` (averaged across folds).
    """
    fold_results = {}
    all_preds, all_targets = [], []

    for test_run in range(1, n_runs + 1):
        train_runs = [r for r in range(1, n_runs + 1) if r != test_run]
        fold_name = f"fold_{test_run}"

        if verbose:
            print(f"\n{'─'*50}")
            print(f"  {subject_dir.name}  |  Fold {test_run}  "
                  f"(test run {test_run}, train runs {train_runs})")
            print(f"{'─'*50}")

        # Extract train data
        X_train, y_train, _ = extract_subject_data(
            subject_dir, dipole_coords_mni, stimuli, train_runs, fmt=fmt, verbose=False
        )
        # Extract test data
        X_test, y_test, trial_names = extract_subject_data(
            subject_dir, dipole_coords_mni, stimuli, [test_run], fmt=fmt, verbose=False
        )

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            warnings.warn(f"Insufficient data for {fold_name}; skipping.")
            continue

        # Train model
        model = BiLSTMDecoder(
            input_size=MODEL_INPUT_SIZE,
            hidden_size=MODEL_HIDDEN_SIZE,
            num_layers=MODEL_NUM_LAYERS,
        )
        model, history = train_model(
            model,
            X_train,
            y_train,
            n_epochs=n_train_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            verbose=verbose,
        )

        # Save model
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = output_dir / f"{subject_dir.name}_{fold_name}.pt"
            save_model(model, str(ckpt_path))

        # Predict
        preds = predict(model, X_test, device=device)  # (n_test, n_time)

        # Evaluate
        metrics = evaluate_all(
            preds, y_test, fs=float(DECODING_SR), verbose=verbose
        )
        metrics["trial_names"] = trial_names
        metrics["train_history"] = history
        fold_results[fold_name] = metrics

        all_preds.append(preds)
        all_targets.append(y_test)

    # Compute mean metrics across folds
    if fold_results:
        mean_metrics = _average_fold_metrics(fold_results)
        fold_results["mean"] = mean_metrics
        if verbose:
            print(f"\n{'='*50}")
            print(f"  {subject_dir.name}  — Mean across {n_runs} folds")
            print_results_table(mean_metrics)

    return fold_results


def _average_fold_metrics(fold_results: Dict) -> Dict:
    """Average scalar metrics across folds (excluding non-scalar keys)."""
    scalar_keys = [
        "r_time_mean", "r_time_std", "r_time_p",
        "r_freq_mean", "r_freq_std", "r_freq_p",
        "ssim_mean", "ssim_std",
        "rank_acc", "rank_acc_p",
    ]
    mean_metrics: Dict = {}
    for k in scalar_keys:
        vals = [v[k] for v in fold_results.values()
                if isinstance(v, dict) and k in v]
        if vals:
            mean_metrics[k] = float(np.mean(vals))
    return mean_metrics


# ---------------------------------------------------------------------------
# Full dataset pipeline
# ---------------------------------------------------------------------------

def run_eegfmri_pipeline(
    dataset_dir: Path,
    output_dir: Path,
    n_subjects: int = 18,
    n_runs: int = N_RUNS,
    dipole_coords_mni: Optional[np.ndarray] = None,
    fmt: str = "brainvision",
    device: Optional[str] = None,
    n_train_epochs: int = TRAIN_EPOCHS,
    fmri_tr: float = 2.0,
    verbose: bool = True,
) -> Dict:
    """Run the full EEG-fMRI pipeline.

    If *dipole_coords_mni* is None, the fMRI GLM is run first to derive them.

    Parameters
    ----------
    dataset_dir : Path
    output_dir : Path
    n_subjects : int
    n_runs : int
    dipole_coords_mni : np.ndarray or None
    fmt : str
    device : str or None
    n_train_epochs : int
    fmri_tr : float
        fMRI repetition time (seconds). ds002725 uses TR=2.0 s (Daly 2023).
    verbose : bool

    Returns
    -------
    all_results : dict  — per-subject fold metrics
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = list_subjects(dataset_dir)[:n_subjects]
    stimuli = load_audio_stimuli(dataset_dir, target_sr=AUDIO_SR)

    # 1. fMRI analysis (if dipoles not provided)
    if dipole_coords_mni is None:
        if verbose:
            print("Running fMRI GLM analysis to identify dipole locations …")
        from .fmri_analysis import run_full_fmri_pipeline
        dipole_coords_mni, group_t_map, _ = run_full_fmri_pipeline(
            dataset_dir,
            subject_dirs,
            n_runs=n_runs,
            t_r=fmri_tr,
            verbose=verbose,
        )
        # Save
        np.save(str(output_dir / "dipole_locations.npy"), dipole_coords_mni)
        if verbose:
            print(f"Dipole locations saved to {output_dir / 'dipole_locations.npy'}")
    else:
        if verbose:
            print(f"Using provided dipole locations:\n{dipole_coords_mni}")

    # 2. Per-subject cross-validation
    all_results = {}
    for subj_dir in subject_dirs:
        if verbose:
            print(f"\n{'#'*60}")
            print(f"  Processing {subj_dir.name}")
            print(f"{'#'*60}")
        results = cross_validate_subject(
            subj_dir,
            dipole_coords_mni=dipole_coords_mni,
            stimuli=stimuli,
            n_runs=n_runs,
            fmt=fmt,
            n_train_epochs=n_train_epochs,
            device=device,
            output_dir=output_dir / subj_dir.name,
            verbose=verbose,
        )
        all_results[subj_dir.name] = results

    # 3. Group summary
    _print_group_summary(all_results)

    # 4. Save dipoles
    np.save(str(output_dir / "dipole_locations.npy"), dipole_coords_mni)

    return all_results


def run_eegonly_pipeline(
    dataset_dir: Path,
    output_dir: Path,
    dipole_coords_mni: np.ndarray,
    n_subjects: int = 19,
    n_runs: int = N_RUNS,
    fmt: str = "eeglab",
    device: Optional[str] = None,
    n_train_epochs: int = TRAIN_EPOCHS,
    verbose: bool = True,
) -> Dict:
    """Run the EEG-only pipeline using pre-computed (group-average) dipole locations.

    Parameters
    ----------
    dataset_dir : Path
    output_dir : Path
    dipole_coords_mni : np.ndarray, shape (4, 3)
    n_subjects, n_runs, fmt, device, n_train_epochs, verbose

    Returns
    -------
    all_results : dict
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = list_subjects(dataset_dir)[:n_subjects]
    stimuli = load_audio_stimuli(dataset_dir, target_sr=AUDIO_SR)

    all_results = {}
    for subj_dir in subject_dirs:
        if verbose:
            print(f"\n{'#'*60}")
            print(f"  Processing {subj_dir.name} (EEG-only)")
            print(f"{'#'*60}")
        results = cross_validate_subject(
            subj_dir,
            dipole_coords_mni=dipole_coords_mni,
            stimuli=stimuli,
            n_runs=n_runs,
            fmt=fmt,
            n_train_epochs=n_train_epochs,
            device=device,
            output_dir=output_dir / subj_dir.name,
            verbose=verbose,
        )
        all_results[subj_dir.name] = results

    _print_group_summary(all_results)
    return all_results


def _print_group_summary(all_results: Dict) -> None:
    """Print a summary table of mean metrics across all subjects."""
    metric_keys = [
        "r_time_mean", "r_freq_mean", "ssim_mean", "rank_acc", "rank_acc_p"
    ]
    rows = []
    for subj, res in all_results.items():
        if "mean" in res:
            row = {"subject": subj}
            row.update({k: res["mean"].get(k, float("nan")) for k in metric_keys})
            rows.append(row)

    if not rows:
        return

    import pandas as pd
    df = pd.DataFrame(rows).set_index("subject")
    print("\n" + "=" * 70)
    print("  Group summary (mean across subjects)")
    print("=" * 70)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n  Grand mean:")
    for k in metric_keys:
        vals = [r[k] for r in rows if not np.isnan(r[k])]
        if vals:
            print(f"    {k:20s} = {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print("=" * 70)

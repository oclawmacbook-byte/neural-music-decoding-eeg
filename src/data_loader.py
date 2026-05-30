"""
data_loader.py
==============
Functions to load EEG, fMRI, and audio data from the OpenNeuro dataset structure
used by Daly (2023).

Dataset structures
------------------
ds002691 (EEG-fMRI):
    sub-<XX>/
        eeg/  sub-<XX>_task-music_run-<N>_eeg.vhdr  (BrainVision)
        func/ sub-<XX>_task-music_run-<N>_bold.nii.gz
        func/ sub-<XX>_task-music_run-<N>_events.tsv
    stimuli/  <piece_name>.wav

ds002137 (EEG-only):
    sub-<XX>/
        eeg/  sub-<XX>_task-music_run-<N>_eeg.set  (EEGLAB)
    stimuli/  <piece_name>.wav
"""

from __future__ import annotations

import os
import glob
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
import librosa


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _subject_id(index: int) -> str:
    """Return zero-padded subject identifier, e.g. 1 → 'sub-01'."""
    return f"sub-{index:02d}"


def _find_file(directory: Path, pattern: str) -> Optional[Path]:
    """Return the first file matching *pattern* inside *directory*, or None."""
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        warnings.warn(f"Multiple files match {pattern!r} in {directory}; using first.")
    return matches[0]


# ---------------------------------------------------------------------------
# EEG loading
# ---------------------------------------------------------------------------

def load_eeg_brainvision(
    subject_dir: Path,
    run: int,
    preload: bool = True,
) -> mne.io.Raw:
    """Load BrainVision EEG recording for a given subject and run.

    Parameters
    ----------
    subject_dir : Path
        Root directory for the subject (e.g. ``data/ds002691/sub-01``).
    run : int
        Run index (1-based).
    preload : bool
        If True, load data into memory immediately.

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object.
    """
    eeg_dir = subject_dir / "eeg"
    vhdr_pattern = f"*task-music_run-{run:02d}_eeg.vhdr"
    vhdr_file = _find_file(eeg_dir, vhdr_pattern)
    if vhdr_file is None:
        # Try without zero-padding
        vhdr_file = _find_file(eeg_dir, f"*task-music_run-{run}_eeg.vhdr")
    if vhdr_file is None:
        raise FileNotFoundError(
            f"No BrainVision header file found for run {run} in {eeg_dir}"
        )
    raw = mne.io.read_raw_brainvision(str(vhdr_file), preload=preload, verbose=False)
    return raw


def load_eeg_eeglab(
    subject_dir: Path,
    run: int,
    preload: bool = True,
) -> mne.io.Raw:
    """Load EEGLAB (.set/.fdt) EEG recording.

    Parameters
    ----------
    subject_dir : Path
        Root directory for the subject.
    run : int
        Run index (1-based).
    preload : bool
        If True, load data into memory immediately.

    Returns
    -------
    raw : mne.io.Raw
    """
    eeg_dir = subject_dir / "eeg"
    set_pattern = f"*task-music_run-{run:02d}_eeg.set"
    set_file = _find_file(eeg_dir, set_pattern)
    if set_file is None:
        set_file = _find_file(eeg_dir, f"*task-music_run-{run}_eeg.set")
    if set_file is None:
        raise FileNotFoundError(
            f"No EEGLAB .set file found for run {run} in {eeg_dir}"
        )
    raw = mne.io.read_raw_eeglab(str(set_file), preload=preload, verbose=False)
    return raw


def load_eeg(
    subject_dir: Path,
    run: int,
    fmt: str = "auto",
    preload: bool = True,
) -> mne.io.Raw:
    """Load EEG, automatically detecting BrainVision or EEGLAB format.

    Parameters
    ----------
    subject_dir : Path
    run : int
    fmt : str
        One of ``'brainvision'``, ``'eeglab'``, or ``'auto'``.
    preload : bool

    Returns
    -------
    raw : mne.io.Raw
    """
    subject_dir = Path(subject_dir)
    if fmt == "brainvision":
        return load_eeg_brainvision(subject_dir, run, preload=preload)
    if fmt == "eeglab":
        return load_eeg_eeglab(subject_dir, run, preload=preload)

    # Auto-detect
    eeg_dir = subject_dir / "eeg"
    if list(eeg_dir.glob(f"*run-{run:02d}_eeg.vhdr")) or list(
        eeg_dir.glob(f"*run-{run}_eeg.vhdr")
    ):
        return load_eeg_brainvision(subject_dir, run, preload=preload)
    if list(eeg_dir.glob(f"*run-{run:02d}_eeg.set")) or list(
        eeg_dir.glob(f"*run-{run}_eeg.set")
    ):
        return load_eeg_eeglab(subject_dir, run, preload=preload)
    raise FileNotFoundError(
        f"Could not find EEG file for run {run} in {eeg_dir}. "
        "Supported formats: BrainVision (.vhdr) and EEGLAB (.set)."
    )


# ---------------------------------------------------------------------------
# fMRI loading
# ---------------------------------------------------------------------------

def load_fmri(
    subject_dir: Path,
    run: int,
) -> Tuple[object, np.ndarray]:
    """Load BOLD fMRI NIfTI image.

    Parameters
    ----------
    subject_dir : Path
    run : int

    Returns
    -------
    img : nibabel.Nifti1Image
        4-D BOLD image.
    events : np.ndarray, shape (N_events, 3)
        Events array [onset_sec, duration_sec, trial_type_code].
    """
    import nibabel as nib

    func_dir = subject_dir / "func"
    nii_pattern = f"*task-music_run-{run:02d}_bold.nii.gz"
    nii_file = _find_file(func_dir, nii_pattern)
    if nii_file is None:
        nii_file = _find_file(func_dir, f"*task-music_run-{run}_bold.nii.gz")
    if nii_file is None:
        raise FileNotFoundError(f"No BOLD file found for run {run} in {func_dir}")

    img = nib.load(str(nii_file))

    # Load events TSV
    tsv_pattern = f"*task-music_run-{run:02d}_events.tsv"
    tsv_file = _find_file(func_dir, tsv_pattern)
    if tsv_file is None:
        tsv_file = _find_file(func_dir, f"*task-music_run-{run}_events.tsv")

    events = np.empty((0, 3))
    if tsv_file is not None:
        df = pd.read_csv(str(tsv_file), sep="\t")
        # Map trial_type to numeric code: music=1, rest=0
        type_map = {"music": 1, "rest": 0}
        codes = df["trial_type"].map(type_map).fillna(0).astype(int)
        events = np.column_stack([df["onset"].values, df["duration"].values, codes.values])

    return img, events


# ---------------------------------------------------------------------------
# Audio (stimulus) loading
# ---------------------------------------------------------------------------

def load_audio_stimuli(
    dataset_dir: Path,
    target_sr: int = 1000,
) -> Dict[str, np.ndarray]:
    """Load all audio stimulus files from the dataset stimuli directory.

    Audio is resampled to *target_sr* Hz (default 1000 Hz as in the paper).

    Parameters
    ----------
    dataset_dir : Path
        Root of the OpenNeuro dataset (contains ``stimuli/`` subdirectory).
    target_sr : int
        Target sample rate.

    Returns
    -------
    stimuli : dict
        Mapping ``{filename_stem: waveform_array}``.
        Waveforms are 1-D float32 arrays at *target_sr* Hz.
    """
    dataset_dir = Path(dataset_dir)
    stimuli_dir = dataset_dir / "stimuli"
    if not stimuli_dir.exists():
        warnings.warn(f"stimuli/ directory not found in {dataset_dir}")
        return {}

    stimuli: Dict[str, np.ndarray] = {}
    audio_extensions = (".wav", ".mp3", ".flac", ".ogg")
    for ext in audio_extensions:
        for audio_file in sorted(stimuli_dir.glob(f"*{ext}")):
            y, _ = librosa.load(str(audio_file), sr=target_sr, mono=True)
            stimuli[audio_file.stem] = y.astype(np.float32)

    return stimuli


def load_events_for_run(
    subject_dir: Path,
    run: int,
) -> pd.DataFrame:
    """Load the BIDS events TSV for a given EEG run.

    Returns a DataFrame with columns: onset, duration, trial_type, stim_file.
    """
    # EEG events TSV (shared with fMRI or in eeg/ folder)
    for subdir in ["eeg", "func", "."]:
        tsv_dir = subject_dir / subdir
        tsv_file = _find_file(tsv_dir, f"*task-music_run-{run:02d}_events.tsv")
        if tsv_file is None:
            tsv_file = _find_file(tsv_dir, f"*task-music_run-{run}_events.tsv")
        if tsv_file is not None:
            return pd.read_csv(str(tsv_file), sep="\t")

    warnings.warn(f"No events TSV found for subject {subject_dir.name} run {run}")
    return pd.DataFrame(columns=["onset", "duration", "trial_type", "stim_file"])


# ---------------------------------------------------------------------------
# Dataset-level loaders
# ---------------------------------------------------------------------------

def list_subjects(dataset_dir: Path) -> List[Path]:
    """Return sorted list of subject directories in a BIDS dataset."""
    dataset_dir = Path(dataset_dir)
    return sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])


def load_subject_runs(
    subject_dir: Path,
    n_runs: int = 3,
    fmt: str = "auto",
) -> Dict[int, mne.io.Raw]:
    """Load all EEG runs for a subject.

    Parameters
    ----------
    subject_dir : Path
    n_runs : int
        Number of runs (default 3 for ds002691/ds002137).
    fmt : str
        EEG format: ``'auto'``, ``'brainvision'``, or ``'eeglab'``.

    Returns
    -------
    raws : dict
        ``{run_index: mne.io.Raw}``
    """
    raws: Dict[int, mne.io.Raw] = {}
    for run in range(1, n_runs + 1):
        try:
            raws[run] = load_eeg(subject_dir, run, fmt=fmt)
        except FileNotFoundError as e:
            warnings.warn(str(e))
    return raws


def load_subject_fmri_runs(
    subject_dir: Path,
    n_runs: int = 3,
) -> Dict[int, Tuple]:
    """Load all fMRI runs for a subject.

    Returns
    -------
    fmri_runs : dict
        ``{run_index: (nibabel_img, events_array)}``
    """
    fmri_runs = {}
    for run in range(1, n_runs + 1):
        try:
            fmri_runs[run] = load_fmri(subject_dir, run)
        except FileNotFoundError as e:
            warnings.warn(str(e))
    return fmri_runs


def epoch_eeg_by_stimulus(
    raw: mne.io.Raw,
    events_df: pd.DataFrame,
    tmin: float = 0.0,
    tmax: float = 40.0,
    baseline: Optional[Tuple[float, float]] = None,
) -> mne.Epochs:
    """Epoch the raw EEG around each music stimulus onset.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed (continuous) EEG.
    events_df : pd.DataFrame
        Events DataFrame with columns ``onset`` (seconds), ``duration``, ``trial_type``.
    tmin, tmax : float
        Epoch window relative to onset (seconds).
    baseline : tuple or None
        Baseline correction window.

    Returns
    -------
    epochs : mne.Epochs
    """
    sfreq = raw.info["sfreq"]

    music_rows = events_df[events_df["trial_type"] == "music"]
    if music_rows.empty:
        raise ValueError("No 'music' events found in events_df.")

    onsets_samples = (music_rows["onset"].values * sfreq).astype(int)
    mne_events = np.column_stack([
        onsets_samples,
        np.zeros(len(onsets_samples), dtype=int),
        np.ones(len(onsets_samples), dtype=int),
    ])

    epochs = mne.Epochs(
        raw,
        mne_events,
        event_id=1,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False,
    )
    return epochs

#!/usr/bin/env python3
"""
run_analysis.py
Reproduces the neural decoding of music from EEG analysis described in:
  Daly (2023) "Neural decoding of music from the EEG" (Scientific Reports)

Pipeline per subject:
  1. fMRI GLM (nilearn) to find 4 subject-specific dipole locations
  2. EEG preprocessing: bandpass 1-40 Hz, average reference, ICA artifact removal
  3. Source feature construction: per-ICA-component eLoreta projection to 4 dipoles
     -> (4 dipoles x 20 ICA components = 80) feature matrix at 100 Hz
  4. 3-fold cross-validation (leave-one-run-out) with 4-layer biLSTM regression
  5. Evaluation: time/frequency Pearson r, SSIM, rank accuracy with bootstrap

Usage:
  python run_analysis.py --n-subjects 3 --output-dir results/
  python run_analysis.py --subjects sub-01 sub-02 --skip-fmri --epochs 50
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

import mne
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import scipy.signal
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
from mne.preprocessing import ICA
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm import compute_fixed_effects
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging / warnings setup
# ---------------------------------------------------------------------------

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz",
    "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6", "TP9", "TP10", "POz",
]
N_EEG_CH = len(EEG_CHANNELS)  # 31

# Frontal channels used for EOG artifact detection (per paper)
FRONTAL_CHANNELS = ["Fp1", "Fp2", "F7", "F8"]

# Group-average fallback dipole locations (Table 1 in paper, MNI mm)
GROUP_AVG_DIPOLES_MM = np.array([
    [54.0,  -12.0,   2.0],   # Right transverse temporal gyrus
    [-50.0, -20.0,   2.0],   # Left transverse temporal gyrus
    [-22.0, -26.0, -16.0],   # Left hippocampus
    [16.0,  -38.0, -10.0],   # Right cerebellum exterior
])
N_DIPOLES = 4

ORIG_SR   = 44100   # WAV sample rate
EEG_SR    = 1000    # EEG sample rate
TARGET_SR = 100     # Downsampled target for both audio and EEG features

TRIAL_DURATION_S      = 40.0
TRIAL_SAMPLES_EEG     = int(TRIAL_DURATION_S * EEG_SR)     # 40000
TRIAL_SAMPLES_TARGET  = int(TRIAL_DURATION_S * TARGET_SR)  # 4000

N_ICA_COMPONENTS = 20
N_RUNS           = 3
N_FEATURES       = N_DIPOLES * N_ICA_COMPONENTS  # 80

ICA_RANDOM_STATE = 42

# biLSTM hyperparameters (exact paper spec)
BILSTM_HIDDEN  = 250
BILSTM_LAYERS  = 4
BILSTM_BATCH   = 4       # trials per batch
TRAIN_LR       = 1e-3
TRAIN_WD       = 1e-5

# fMRI GLM
GLM_TR          = 2.0
GLM_DIPOLES     = 4
GLM_P_THRESH    = 0.001  # uncorrected
GLM_MIN_SPACING = 30.0   # mm — minimum distance between selected dipoles


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce Daly (2023) neural music decoding from EEG."
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/ds002725",
        help="Path to ds002725 dataset root (default: data/ds002725)",
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="List of subject IDs like sub-01 sub-02 ... or 'all'",
    )
    parser.add_argument(
        "--n-subjects", type=int, default=3,
        help="Number of subjects to process when --subjects is not set (default: 3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for output files (default: results/)",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device: cuda or cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs for biLSTM (default: 50)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=4000,
        help="Bootstrap shuffles for rank accuracy p-value (default: 4000)",
    )
    parser.add_argument(
        "--skip-fmri", action="store_true",
        help="Skip fMRI GLM and use group-average dipole locations directly",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_subject_list(data_dir: Path, subjects_arg, n_subjects: int):
    """Return sorted list of subject directory names to process."""
    all_subs = sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )
    if subjects_arg is not None:
        if len(subjects_arg) == 1 and subjects_arg[0].lower() == "all":
            return all_subs
        return [s for s in subjects_arg if s in all_subs]
    return all_subs[:n_subjects]


def resample_audio(audio: np.ndarray, orig_sr: int = ORIG_SR) -> np.ndarray:
    """
    Downsample audio from orig_sr to TARGET_SR (100 Hz).
    Uses resample_poly as specified in paper.
    Returns mono float32 array, length ~ TRIAL_SAMPLES_TARGET.
    """
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    # orig_sr (44100) -> TARGET_SR (100): ratio = 1/441
    audio_100 = scipy.signal.resample_poly(audio, up=1, down=orig_sr // TARGET_SR).astype(np.float32)
    return audio_100


def decode_music_code_strict(music_channel_v: np.ndarray, stimuli_dir: Path):
    """
    Decode music stimulus ID from the 'music' EDF channel.
    Per channels.tsv: "multiply the value by 20" (value in µV).
    code = round(max_abs_uV * 20)
    A = code // 100, B = (code % 100) // 10, C = code % 10
    Example: 14.1 µV * 20 = 282 -> 2-8_2.wav

    STRICT: only accept trials where:
      - A >= 1, A <= 9
      - B >= 1, B <= 9
      - A != B
      - C in {1, 2, 3}
      - The file {A}-{B}_{C}.wav exists

    Returns stim_id string (e.g. '2-8_2') or None if any constraint fails.
    """
    music_uv = music_channel_v * 1e6  # V -> µV
    max_val = float(np.max(np.abs(music_uv)))
    code = round(max_val * 20)  # channels.tsv spec: value_uV * 20 = 3-digit code
    if code == 0:
        return None

    A = code // 100
    B = (code % 100) // 10
    C = code % 10

    # Strict validity checks — no approximation
    if A < 1 or A > 9:
        return None
    if B < 1 or B > 9:
        return None
    if A == B:
        return None
    if C not in (1, 2, 3):
        return None

    stim_id = f"{A}-{B}_{C}"
    wav_path = stimuli_dir / f"{stim_id}.wav"
    if not wav_path.exists():
        return None

    return stim_id


def load_fmri_safe(bold_path: Path):
    """
    Load fMRI NIfTI, handling truncated gzip files by loading as many
    scans as are available.
    Returns (img_array, affine, n_scans_loaded) or raises on total failure.
    """
    img = nib.load(str(bold_path))
    total_declared = img.shape[-1]

    # Try full load first
    try:
        data = img.get_fdata()
        return data, img.affine, total_declared
    except (EOFError, Exception):
        pass

    # Binary-search for the number of loadable scans
    lo, hi = 1, total_declared - 1
    last_ok = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            _ = np.array(img.dataobj[..., :mid])
            last_ok = mid
            lo = mid + 1
        except Exception:
            hi = mid - 1

    if last_ok == 0:
        raise RuntimeError(f"Cannot load any scans from {bold_path}")

    data = np.array(img.dataobj[..., :last_ok])
    logger.warning(
        "LIMITATION: fMRI %s truncated — loaded %d/%d scans",
        bold_path.name, last_ok, total_declared
    )
    return data, img.affine, last_ok


# ---------------------------------------------------------------------------
# Step 1: fMRI GLM dipole localisation
# ---------------------------------------------------------------------------

def run_fmri_glm(subject_id: str, data_dir: Path) -> tuple:
    """
    Fit first-level GLM on each genMusic run for a subject, combine with
    compute_fixed_effects, extract music > baseline t-map, and select
    4 dipole locations using the greedy spacing algorithm (>3cm apart).

    Returns (dipole_coords_mm, source_str) where source_str is either
    'fmri_glm' or 'group_average'.
    """
    subj_func_dir = data_dir / subject_id / "func"

    contrast_imgs = []
    variance_imgs = []

    for run_i in range(1, N_RUNS + 1):
        run_str = f"{run_i:02d}"
        bold_path  = subj_func_dir / f"{subject_id}_task-genMusic{run_str}_bold.nii.gz"
        events_tsv = subj_func_dir / f"{subject_id}_task-genMusic{run_str}_events.tsv"

        if not bold_path.exists() or not events_tsv.exists():
            logger.warning(
                "fMRI: missing files for %s run %d — skipping run",
                subject_id, run_i
            )
            continue

        try:
            data, affine, n_scans = load_fmri_safe(bold_path)
        except Exception as exc:
            logger.warning("fMRI: failed to load %s: %s", bold_path.name, exc)
            continue

        total_time = n_scans * GLM_TR

        # Build events design matrix from 768 events (every other = trial start)
        events_df = pd.read_csv(str(events_tsv), sep="\t")
        ev768 = events_df[events_df["trial_type"] == 768]
        ev768_starts = ev768.iloc[::2]  # first of each pair

        events_glm = pd.DataFrame({
            "onset":      ev768_starts["onset"].values,
            "duration":   TRIAL_DURATION_S,
            "trial_type": "music",
        })
        # Keep only events whose music segment fits within available data
        events_glm = events_glm[
            events_glm["onset"] + TRIAL_DURATION_S <= total_time
        ].reset_index(drop=True)

        if len(events_glm) < 2:
            logger.warning(
                "fMRI: only %d usable events for %s run %d, skipping run",
                len(events_glm), subject_id, run_i
            )
            continue

        # Save truncated NIfTI to temp file so nilearn can load it
        tmp_img = nib.Nifti1Image(data.astype(np.float32), affine)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            tmp_path = f.name
        try:
            nib.save(tmp_img, tmp_path)
            glm = FirstLevelModel(
                t_r=GLM_TR,
                hrf_model="spm",
                noise_model="ar1",
                standardize=True,
                verbose=0,
            )
            glm.fit(tmp_path, events_glm)
            c_img = glm.compute_contrast("music", stat_type="t", output_type="stat")
            v_img = glm.compute_contrast("music", stat_type="t", output_type="effect_variance")
            contrast_imgs.append(c_img)
            variance_imgs.append(v_img)
            logger.info(
                "  fMRI GLM: %s run %d fitted (%d events, %d/%d scans)",
                subject_id, run_i, len(events_glm), n_scans, n_scans
            )
        except Exception as exc:
            logger.warning("fMRI GLM failed for %s run %d: %s", subject_id, run_i, exc)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if len(contrast_imgs) == 0:
        logger.warning(
            "LIMITATION: fMRI GLM produced no valid runs for %s; using group-average dipoles",
            subject_id
        )
        return GROUP_AVG_DIPOLES_MM.copy(), "group_average"

    # Combine runs with fixed effects if multiple
    if len(contrast_imgs) == 1:
        t_img = contrast_imgs[0]
    else:
        try:
            t_img, _, _ = compute_fixed_effects(contrast_imgs, variance_imgs)
        except Exception as exc:
            logger.warning(
                "fMRI fixed-effects combination failed (%s); using first run only", exc
            )
            t_img = contrast_imgs[0]

    # Extract t-statistics
    t_data   = t_img.get_fdata()
    affine   = t_img.affine
    n_dof    = max(30, sum(c.get_fdata().size for c in contrast_imgs) // t_data.size)
    t_thresh = scipy.stats.t.ppf(1.0 - GLM_P_THRESH, df=n_dof)

    sig_mask    = t_data > t_thresh
    n_sig       = int(np.sum(sig_mask))
    logger.info(
        "  fMRI: %s — %d significant voxels (t > %.2f)",
        subject_id, n_sig, t_thresh
    )

    if n_sig < GLM_DIPOLES:
        logger.warning(
            "LIMITATION: only %d significant voxels for %s (need %d); "
            "using group-average dipoles",
            n_sig, subject_id, GLM_DIPOLES
        )
        return GROUP_AVG_DIPOLES_MM.copy(), "group_average"

    # Greedy dipole selection: highest t-value first, >3cm apart
    sig_indices = np.argwhere(sig_mask)
    t_values    = t_data[sig_mask]
    order       = np.argsort(-t_values)
    sig_indices = sig_indices[order]
    # Convert voxel indices to MNI mm coords via affine
    sig_coords_mm = nib.affines.apply_affine(affine, sig_indices)

    selected_coords = []
    for coord in sig_coords_mm:
        too_close = any(
            np.linalg.norm(coord - sc) < GLM_MIN_SPACING
            for sc in selected_coords
        )
        if not too_close:
            selected_coords.append(coord)
        if len(selected_coords) == GLM_DIPOLES:
            break

    if len(selected_coords) < GLM_DIPOLES:
        logger.warning(
            "LIMITATION: found only %d dipoles for %s (spacing constraint); "
            "padding with group-average",
            len(selected_coords), subject_id
        )
        for ga_coord in GROUP_AVG_DIPOLES_MM:
            if len(selected_coords) >= GLM_DIPOLES:
                break
            too_close = any(
                np.linalg.norm(ga_coord - sc) < GLM_MIN_SPACING
                for sc in selected_coords
            )
            if not too_close:
                selected_coords.append(ga_coord)
        # If still short, just append from group average
        while len(selected_coords) < GLM_DIPOLES:
            selected_coords.append(GROUP_AVG_DIPOLES_MM[len(selected_coords) - 1])

    dipoles_mm = np.array(selected_coords[:GLM_DIPOLES])
    logger.info("  fMRI dipoles (MNI mm) for %s: %s", subject_id, dipoles_mm.tolist())
    return dipoles_mm, "fmri_glm"


# ---------------------------------------------------------------------------
# Step 2: EEG preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess_eeg(edf_path: Path):
    """
    Load EDF, pick 31 EEG channels, apply average reference, bandpass 1-40 Hz.
    Also returns the raw 'music' channel data (Volts) before picking.
    Returns (raw_filtered, music_ch_data_volts).
    """
    raw_full = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    # Extract music channel before picking EEG-only channels
    music_ch_data = None
    if "music" in raw_full.ch_names:
        music_idx = raw_full.ch_names.index("music")
        music_ch_data = raw_full.get_data()[music_idx]  # Volts

    # Pick and rename to canonical EEG channels
    eeg_ch_lower = [c.lower() for c in EEG_CHANNELS]
    ch_to_pick   = [c for c in raw_full.ch_names if c.lower() in eeg_ch_lower]
    if len(ch_to_pick) == 0:
        raise RuntimeError(f"No EEG channels found in {edf_path}")
    raw_full.pick_channels(ch_to_pick)
    rename_map = {c: EEG_CHANNELS[eeg_ch_lower.index(c.lower())] for c in ch_to_pick}
    raw_full.rename_channels(rename_map)
    raw_full.reorder_channels([c for c in EEG_CHANNELS if c in raw_full.ch_names])

    # Set standard montage
    montage = mne.channels.make_standard_montage("standard_1005")
    raw_full.set_montage(montage, on_missing="warn", verbose=False)

    # Average reference
    raw_full.set_eeg_reference(projection=True, verbose=False)
    raw_full.apply_proj(verbose=False)

    # Bandpass 1–40 Hz
    raw_full.filter(l_freq=1.0, h_freq=40.0, method="fir", fir_window="hamming",
                    verbose=False)

    return raw_full, music_ch_data


def fit_ica_and_exclude_eog(raw: mne.io.BaseRaw):
    """
    Fit ICA (FastICA, 20 components) on the run.
    Auto-exclude EOG-like components: those where max |activation| in frontal
    channels (Fp1, Fp2, F7, F8) > 3x median frontal activation across all
    components.

    Returns (ica, excluded_indices).
    """
    ica = ICA(
        n_components=N_ICA_COMPONENTS,
        method="fastica",
        random_state=ICA_RANDOM_STATE,
        max_iter=500,
        fit_params={"tol": 1e-4},
    )
    ica.fit(raw, verbose=False)

    # Sensor-space mixing matrix: shape (n_channels, n_components)
    A = ica.get_components()  # (n_eeg_ch, n_ica)

    frontal_idx = [
        i for i, c in enumerate(raw.ch_names) if c in FRONTAL_CHANNELS
    ]

    if frontal_idx:
        frontal_act    = np.abs(A[frontal_idx, :])       # (n_frontal, n_ica)
        max_frontal    = frontal_act.max(axis=0)          # (n_ica,)
        median_frontal = np.median(max_frontal)
        threshold      = 3.0 * median_frontal if median_frontal > 0 else np.inf
        exclude = [k for k in range(A.shape[1]) if max_frontal[k] > threshold]
    else:
        exclude = []

    ica.exclude = exclude
    if exclude:
        logger.debug("  ICA EOG-like components excluded: %s", exclude)

    return ica


# ---------------------------------------------------------------------------
# Step 3: Source feature construction
# ---------------------------------------------------------------------------

_source_pipeline_cache = {}  # keyed by tuple(dipoles_mm.tobytes())


def clamp_dipoles_to_sphere(dipoles_mm: np.ndarray,
                            r0_mm: np.ndarray = None,
                            inner_skull_mm: float = 80.0,
                            safety: float = 0.95) -> np.ndarray:
    """
    Project any dipole that lies outside the sphere's inner skull back
    to safety * inner_skull_mm from the sphere origin r0.
    This prevents MNE's make_forward_solution from silently dropping sources.

    r0_mm: sphere origin in mm (default [0, 0, 40])
    inner_skull_mm: inner skull radius in mm (default 80mm)
    safety: fraction of inner skull radius to project to (default 0.95)
    """
    if r0_mm is None:
        r0_mm = np.array([0.0, 0.0, 40.0])
    clamped = dipoles_mm.copy().astype(float)
    max_r = inner_skull_mm * safety
    for i, d in enumerate(clamped):
        vec  = d - r0_mm
        dist = np.linalg.norm(vec)
        if dist > max_r:
            clamped[i] = r0_mm + vec / dist * max_r
            logger.debug(
                "  Dipole %d clamped from %.1fmm to %.1fmm from sphere centre",
                i, dist, max_r
            )
    return clamped


def build_source_pipeline(info: mne.Info, dipoles_mm: np.ndarray):
    """
    Build sphere head model, discrete source space, forward solution,
    noise covariance, and eLoreta inverse operator for the given dipole
    locations. Caches per unique dipole set.

    Dipoles that fall outside the sphere's inner skull are projected inward
    so that all N_DIPOLES sources are retained in the forward solution.

    Returns (inv_op, info_used).
    """
    # Clamp dipoles inside the sphere before using as cache key, so that
    # different raw coords that clamp to the same location share the cache.
    dipoles_clamped = clamp_dipoles_to_sphere(dipoles_mm)
    cache_key = dipoles_clamped.tobytes()
    if cache_key in _source_pipeline_cache:
        return _source_pipeline_cache[cache_key]

    # Sphere head model (paper spec)
    sphere = mne.make_sphere_model(r0=(0, 0, 0.04), head_radius=0.09, verbose=False)

    # Discrete source space at dipole locations (MNI mm -> head m)
    dipole_rr = dipoles_clamped / 1000.0               # mm -> m
    dipole_nn = dipole_rr / (
        np.linalg.norm(dipole_rr, axis=1, keepdims=True) + 1e-10
    )
    src = mne.setup_volume_source_space(
        pos={"rr": dipole_rr, "nn": dipole_nn},
        verbose=False,
    )

    # Forward solution
    fwd = mne.make_forward_solution(
        info, trans=None, src=src, bem=sphere,
        eeg=True, meg=False, verbose=False,
    )

    # Ad-hoc noise covariance (identity scaled to data)
    noise_cov = mne.make_ad_hoc_cov(info, verbose=False)

    # eLoreta inverse operator
    inv_op = mne.minimum_norm.make_inverse_operator(
        info, fwd, noise_cov, loose=1.0, depth=0.8, verbose=False
    )

    _source_pipeline_cache[cache_key] = (inv_op, info)
    return inv_op, info


def compute_source_features(
    eeg_epoch: np.ndarray,    # (31, T_1000hz)
    info: mne.Info,
    ica: ICA,
    dipoles_mm: np.ndarray,   # (4, 3) MNI mm
) -> np.ndarray:
    """
    Paper's source feature construction:

    1. Compute ICA activations: W @ eeg_data -> (20, T_1000hz)
    2. For each ICA component k:
       a. Component sensor data = A[:, k:k+1] @ ica_act[k:k+1, :] -> (31, T)
       b. Create temp Raw, apply eLoreta -> source activity (4, T)
    3. Concatenate all 20 component source activities:
       features shape = (4*20=80, T_1000hz)
    4. Downsample to 100 Hz with resample_poly(features, 1, 10, axis=1)

    Falls back to direct ICA activations (20, T_100hz) if eLoreta fails.
    Returns features (n_feat, T_100hz).
    """
    n_ch, n_times = eeg_epoch.shape
    n_ica = ica.n_components_

    # ICA activations via get_sources on the epoch
    # Build temporary Raw for this epoch
    raw_epoch = mne.io.RawArray(eeg_epoch, info, verbose=False)
    raw_epoch.set_eeg_reference(projection=True, verbose=False)
    raw_epoch.apply_proj(verbose=False)

    # Sensor-space mixing matrix A: (n_ch, n_ica)
    A = ica.get_components()  # (n_ch, n_ica)

    # ICA activations using MNE's source extraction
    sources_raw = ica.get_sources(raw_epoch)
    ica_activations = sources_raw.get_data()  # (n_ica, n_times)

    try:
        inv_op, _ = build_source_pipeline(info, dipoles_mm)

        feature_parts = []
        for k in range(n_ica):
            # Component k's contribution to sensor space
            comp_sensor = A[:, k:k+1] @ ica_activations[k:k+1, :]  # (n_ch, T)

            raw_comp = mne.io.RawArray(comp_sensor, info, verbose=False)
            raw_comp.set_eeg_reference(projection=True, verbose=False)
            raw_comp.apply_proj(verbose=False)

            stc = mne.minimum_norm.apply_inverse_raw(
                raw_comp, inv_op,
                lambda2=1.0 / 9.0,  # SNR=3
                method="eLORETA",
                verbose=False,
            )
            # stc.data: (4, T_1000hz)
            feature_parts.append(stc.data)  # (4, T)

        # Concatenate: (n_ica * 4, T) = (80, T)
        features_1k = np.concatenate(feature_parts, axis=0).astype(np.float32)

        # Downsample to 100 Hz (paper spec: resample_poly(..., 1, 10, axis=1))
        features_100 = scipy.signal.resample_poly(
            features_1k, up=1, down=10, axis=1
        ).astype(np.float32)

        return features_100

    except Exception as exc:
        logger.warning(
            "LIMITATION: eLoreta source localisation failed (%s); "
            "falling back to ICA activations directly.",
            exc
        )
        # Fallback: shape (20, T_100hz) — fewer features but still usable
        ica_100 = scipy.signal.resample_poly(
            ica_activations, up=1, down=10, axis=1
        ).astype(np.float32)
        return ica_100


# ---------------------------------------------------------------------------
# Step 4: biLSTM model (paper specification: 4 layers, 250 hidden units)
# ---------------------------------------------------------------------------

class BiLSTMDecoder(nn.Module):
    """
    4-layer bidirectional LSTM for sequence regression.
    Input:  (batch, seq_len, n_features)
    Output: (batch, seq_len) — predicted audio waveform
    Paper spec: 4 layers, 250 hidden units, bidirectional, FC output Linear(500, 1).
    """

    def __init__(self, input_size: int,
                 hidden_size: int = BILSTM_HIDDEN,
                 num_layers: int = BILSTM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)  # 500 -> 1 (bidir doubles hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)             # (batch, seq_len, 2*hidden)
        pred = self.fc(out).squeeze(-1)   # (batch, seq_len)
        return pred


def zscore_trial_features(features: np.ndarray) -> np.ndarray:
    """
    Z-score normalize per trial, per feature dimension (across time axis).
    features: (n_features, n_times) -> normalized (n_features, n_times).
    Paper: z-score normalize features per-trial (not per-window).
    """
    mean = features.mean(axis=1, keepdims=True)
    std  = features.std(axis=1, keepdims=True)
    std  = np.where(std < 1e-12, 1.0, std)
    return (features - mean) / std


def zscore_trial_audio(audio: np.ndarray) -> np.ndarray:
    """Z-score normalize audio per trial."""
    std = audio.std()
    if std < 1e-12:
        return audio - audio.mean()
    return (audio - audio.mean()) / std


def train_bilstm(
    train_features: list,   # list of np arrays (n_features, TRIAL_SAMPLES_TARGET)
    train_audio: list,      # list of np arrays (TRIAL_SAMPLES_TARGET,)
    n_features: int,
    device: torch.device,
    n_epochs: int = 50,
) -> BiLSTMDecoder:
    """
    Train a BiLSTM on training trials.
    - Each trial is a full 4000-sample sequence (40s at 100Hz)
    - Features z-scored per-trial (preserving temporal context)
    - Audio z-scored per-trial
    - Batch size 4 trials, Adam lr=0.001, ReduceLROnPlateau, 50 epochs
    """
    model = BiLSTMDecoder(input_size=n_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    # Pre-process: z-score each trial individually (not per-window)
    X_list, Y_list = [], []
    for feat, aud in zip(train_features, train_audio):
        feat_norm = zscore_trial_features(feat)     # (n_feat, T)
        aud_norm  = zscore_trial_audio(aud)          # (T,)
        T = min(feat_norm.shape[1], len(aud_norm), TRIAL_SAMPLES_TARGET)
        X_list.append(feat_norm[:, :T].T)  # (T, n_feat)
        Y_list.append(aud_norm[:T])        # (T,)

    if not X_list:
        logger.warning("No training trials, returning untrained model.")
        return model

    # Stack into tensors — all trials are the same length (TRIAL_SAMPLES_TARGET)
    X_all = torch.from_numpy(np.stack(X_list)).float()  # (N, T, n_feat)
    Y_all = torch.from_numpy(np.stack(Y_list)).float()  # (N, T)

    n_trials = len(X_list)
    model.train()
    pbar = tqdm(range(n_epochs), desc="  biLSTM epochs", leave=False)
    for epoch in pbar:
        epoch_loss = 0.0
        perm = torch.randperm(n_trials)
        n_batches = 0
        for start in range(0, n_trials, BILSTM_BATCH):
            idx = perm[start: start + BILSTM_BATCH]
            x = X_all[idx].to(device)
            y = Y_all[idx].to(device)

            optimizer.zero_grad()
            pred = model(x)                         # (batch, T)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step(avg_loss)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}",
                          "lr": f"{optimizer.param_groups[0]['lr']:.5f}"})

    return model


def predict_bilstm(
    model: BiLSTMDecoder,
    features: np.ndarray,   # (n_features, TRIAL_SAMPLES_TARGET)
    device: torch.device,
) -> np.ndarray:
    """
    Run inference on a full trial sequence.
    Returns predicted waveform of shape (TRIAL_SAMPLES_TARGET,).
    """
    feat_norm = zscore_trial_features(features)
    T = feat_norm.shape[1]
    x = torch.from_numpy(feat_norm[:, :T].T).float().unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy()
    return pred


# ---------------------------------------------------------------------------
# Step 5: Evaluation metrics
# ---------------------------------------------------------------------------

def compute_pearson_time(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Time-domain Pearson r."""
    n = min(len(predicted), len(actual))
    if n < 2:
        return float("nan")
    r, _ = pearsonr(predicted[:n], actual[:n])
    return float(r) if np.isfinite(r) else float("nan")


def compute_pearson_freq(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Frequency-domain Pearson r between power spectra."""
    n = min(len(predicted), len(actual))
    if n < 2:
        return float("nan")
    p_pow = np.abs(np.fft.rfft(predicted[:n])) ** 2
    a_pow = np.abs(np.fft.rfft(actual[:n]))    ** 2
    if np.std(p_pow) < 1e-20 or np.std(a_pow) < 1e-20:
        return float("nan")
    r, _ = pearsonr(p_pow, a_pow)
    return float(r) if np.isfinite(r) else float("nan")


def compute_ssim(predicted: np.ndarray, actual: np.ndarray,
                 nperseg: int = 256, noverlap: int = 128) -> float:
    """SSIM of log-magnitude STFT spectrograms."""
    n = min(len(predicted), len(actual))
    if n < nperseg:
        return float("nan")
    _, _, Sp = scipy.signal.stft(predicted[:n], fs=TARGET_SR,
                                  nperseg=nperseg, noverlap=noverlap)
    _, _, Sa = scipy.signal.stft(actual[:n],    fs=TARGET_SR,
                                  nperseg=nperseg, noverlap=noverlap)
    Sp_mag = np.abs(Sp).astype(np.float64)
    Sa_mag = np.abs(Sa).astype(np.float64)

    # Log-magnitude (paper uses log-spectrogram)
    eps = 1e-10
    Sp_log = np.log(Sp_mag + eps)
    Sa_log = np.log(Sa_mag + eps)

    # Normalise to [0,1] for SSIM
    def norm01(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-20)

    return float(structural_similarity(norm01(Sp_log), norm01(Sa_log), data_range=1.0))


def compute_rank_accuracy(
    predicted_list: list,
    actual_list: list,
    n_bootstrap: int = 4000,
) -> dict:
    """
    For each test trial T playing music M:
      - Compute sim_correct = pearsonr(decoded_T, original_M)
      - For every other trial T' (with music M'): sim_other = pearsonr(decoded_T, original_M')
      - rank_T = fraction of other trials where sim_other < sim_correct
    Mean rank across all test trials = rank accuracy.
    Bootstrap p-value: shuffle trial identities 4000 times.

    Skip if fewer than 5 test trials (not enough for meaningful ranking).
    """
    n = len(predicted_list)
    if n < 5:
        logger.warning(
            "LIMITATION: only %d test trials — skipping rank accuracy (need >= 5)", n
        )
        return {"mean_rank": float("nan"), "p_value": float("nan"), "n_trials": n,
                "skipped": True}

    # Build similarity matrix: sim[i, j] = pearsonr(predicted_i, actual_j)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            nn_ = min(len(predicted_list[i]), len(actual_list[j]))
            if nn_ < 2:
                sim[i, j] = 0.0
            else:
                try:
                    r, _ = pearsonr(predicted_list[i][:nn_], actual_list[j][:nn_])
                    sim[i, j] = float(r) if np.isfinite(r) else 0.0
                except Exception:
                    sim[i, j] = 0.0

    # Observed rank accuracy
    ranks = np.array([
        float(np.mean(sim[i, :] < sim[i, i]))
        for i in range(n)
    ])
    mean_rank = float(np.mean(ranks))

    # Bootstrap null distribution: shuffle which actual corresponds to each decoded
    rng = np.random.default_rng(seed=0)
    null_ranks = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        perm = rng.permutation(n)
        null_r = np.mean([
            float(np.mean(sim[i, :] < sim[i, perm[i]]))
            for i in range(n)
        ])
        null_ranks[b] = null_r

    p_value = float(np.mean(null_ranks >= mean_rank))

    return {
        "mean_rank":       mean_rank,
        "p_value":         p_value,
        "n_trials":        n,
        "ranks_per_trial": ranks.tolist(),
    }


# ---------------------------------------------------------------------------
# Per-subject processing (main pipeline)
# ---------------------------------------------------------------------------

def process_subject(
    subject_id: str,
    data_dir: Path,
    stimuli_dir: Path,
    device: torch.device,
    n_epochs: int,
    n_bootstrap: int,
    skip_fmri: bool = False,
) -> dict:
    """
    Full Daly (2023) pipeline for one subject.

    Returns a dict with per-subject results in the paper's output format.
    """
    logger.info("=" * 60)
    logger.info("Processing subject: %s", subject_id)

    subj_dir = data_dir / subject_id
    eeg_dir  = subj_dir / "eeg"

    # ===========================================================
    # Step 1: fMRI GLM to find dipole locations
    # ===========================================================
    if skip_fmri:
        dipoles_mm     = GROUP_AVG_DIPOLES_MM.copy()
        dipole_source  = "group_average"
        logger.info("  --skip-fmri: using group-average dipole locations")
    else:
        logger.info("  Running fMRI GLM for %s ...", subject_id)
        try:
            dipoles_mm, dipole_source = run_fmri_glm(subject_id, data_dir)
        except Exception as exc:
            logger.warning(
                "  fMRI GLM raised exception (%s); using group-average dipoles", exc
            )
            dipoles_mm    = GROUP_AVG_DIPOLES_MM.copy()
            dipole_source = "group_average"

    logger.info("  Dipole source: %s", dipole_source)

    # ===========================================================
    # Step 2: EEG preprocessing — all 3 runs
    # ===========================================================
    runs = []   # list of (run_idx, raw_clean, ica, music_ch_data, events_df)

    for run_i in range(1, N_RUNS + 1):
        run_str   = f"{run_i:02d}"
        edf_path  = eeg_dir / f"{subject_id}_task-genMusic{run_str}_eeg.edf"
        evts_path = eeg_dir / f"{subject_id}_task-genMusic{run_str}_events.tsv"

        if not edf_path.exists():
            logger.warning(
                "LIMITATION: missing EDF %s — skipping run %d", edf_path.name, run_i
            )
            continue
        if not evts_path.exists():
            logger.warning(
                "LIMITATION: missing events TSV for %s run %d — skipping", subject_id, run_i
            )
            continue

        logger.info("  EEG run %d: loading and preprocessing ...", run_i)
        try:
            raw_clean, music_ch_data = load_and_preprocess_eeg(edf_path)
        except Exception as exc:
            logger.warning("  EEG load failed for run %d: %s", run_i, exc)
            continue

        logger.info("  EEG run %d: fitting ICA ...", run_i)
        try:
            ica = fit_ica_and_exclude_eog(raw_clean)
        except Exception as exc:
            logger.warning("  ICA failed for run %d: %s", run_i, exc)
            continue

        events_df = pd.read_csv(str(evts_path), sep="\t")
        runs.append((run_i - 1, raw_clean, ica, music_ch_data, events_df))

    if len(runs) < 2:
        msg = f"Only {len(runs)} valid EEG runs (need >= 2)"
        logger.error("LIMITATION: %s for %s — skipping subject", msg, subject_id)
        return {"subject": subject_id, "error": msg, "n_valid_runs": len(runs)}

    # ===========================================================
    # Step 2b: Extract trial epochs + load matching WAV files
    # ===========================================================
    all_trials = []   # list of dicts

    n_skipped_invalid = 0
    n_skipped_no_audio = 0

    for run_idx, raw_clean, ica, music_ch_data, events_df in runs:
        run_display = run_idx + 1
        sfreq       = raw_clean.info["sfreq"]  # 1000 Hz

        # 768 events come in pairs; take every other one (0, 2, 4 ...) = trial starts
        ev768 = events_df[events_df["trial_type"] == 768].reset_index(drop=True)
        trial_onsets_s = ev768.iloc[::2]["onset"].values
        logger.info(
            "  Run %d: %d 768 events -> %d trial starts",
            run_display, len(ev768), len(trial_onsets_s)
        )

        for t_idx, onset_s in enumerate(trial_onsets_s):
            onset_samp = int(onset_s * sfreq)
            end_samp   = onset_samp + TRIAL_SAMPLES_EEG

            if end_samp > raw_clean.n_times:
                logger.debug(
                    "  SKIP trial %d run %d: extends beyond recording (%d > %d)",
                    t_idx, run_display, end_samp, raw_clean.n_times
                )
                n_skipped_invalid += 1
                continue

            # ---- Strict stimulus code matching ----
            if music_ch_data is None:
                logger.debug(
                    "  SKIP trial %d run %d: no music channel data", t_idx, run_display
                )
                n_skipped_no_audio += 1
                continue

            music_window = music_ch_data[onset_samp:end_samp]
            stim_id = decode_music_code_strict(music_window, stimuli_dir)
            if stim_id is None:
                logger.debug(
                    "  SKIP trial %d run %d: invalid/missing stimulus code",
                    t_idx, run_display
                )
                n_skipped_invalid += 1
                continue

            # ---- Load WAV ----
            wav_path = stimuli_dir / f"{stim_id}.wav"
            try:
                sr_wav, wav_data = wavfile.read(str(wav_path))
            except Exception as exc:
                logger.warning(
                    "  SKIP trial %d run %d: cannot read WAV %s: %s",
                    t_idx, run_display, wav_path.name, exc
                )
                n_skipped_no_audio += 1
                continue

            audio_100 = resample_audio(wav_data, orig_sr=sr_wav)
            # Trim / pad to TRIAL_SAMPLES_TARGET
            if len(audio_100) >= TRIAL_SAMPLES_TARGET:
                audio_100 = audio_100[:TRIAL_SAMPLES_TARGET]
            else:
                audio_100 = np.pad(audio_100,
                                   (0, TRIAL_SAMPLES_TARGET - len(audio_100)))

            # ---- Extract EEG epoch ----
            eeg_epoch = raw_clean.get_data()[:, onset_samp:end_samp].astype(np.float32)

            all_trials.append({
                "eeg":       eeg_epoch,
                "audio":     audio_100.astype(np.float32),
                "stim_id":   stim_id,
                "run_idx":   run_idx,
                "trial_idx": t_idx,
                "info":      raw_clean.info,
                "ica":       ica,
            })

        logger.info(
            "  Run %d: %d valid trials loaded", run_display,
            sum(1 for t in all_trials if t["run_idx"] == run_idx)
        )

    logger.info(
        "  %s: %d valid trials total (%d skipped: %d invalid code, %d no audio)",
        subject_id, len(all_trials),
        n_skipped_invalid + n_skipped_no_audio,
        n_skipped_invalid, n_skipped_no_audio,
    )

    if len(all_trials) == 0:
        logger.error("LIMITATION: no valid trials for %s — skipping", subject_id)
        return {"subject": subject_id, "error": "no valid trials"}

    n_unique_stim = len(set(t["stim_id"] for t in all_trials))
    logger.info("  Unique stimuli: %d", n_unique_stim)

    # ===========================================================
    # Step 3: Source feature construction
    # ===========================================================
    logger.info("  Computing source features (eLoreta, %d trials) ...", len(all_trials))

    for trial in tqdm(all_trials, desc=f"  Source features ({subject_id})", leave=False):
        ica  = trial.pop("ica")
        info = trial.pop("info")

        features = compute_source_features(trial["eeg"], info, ica, dipoles_mm)

        # Trim / pad features to TRIAL_SAMPLES_TARGET
        n_f, n_t = features.shape
        if n_t > TRIAL_SAMPLES_TARGET:
            features = features[:, :TRIAL_SAMPLES_TARGET]
        elif n_t < TRIAL_SAMPLES_TARGET:
            features = np.pad(features, ((0, 0), (0, TRIAL_SAMPLES_TARGET - n_t)))

        trial["features"] = features

    n_features = all_trials[0]["features"].shape[0]
    logger.info("  Feature dimension: %d (paper expects 80)", n_features)
    if n_features != N_FEATURES:
        logger.warning(
            "LIMITATION: feature dim is %d, expected %d (using fallback features)",
            n_features, N_FEATURES
        )

    # ===========================================================
    # Step 4: 3-fold cross-validation (leave-one-run-out)
    # ===========================================================
    run_indices = sorted(set(t["run_idx"] for t in all_trials))
    all_predicted = []
    all_actual    = []
    cv_metrics    = []

    for test_run_idx in run_indices:
        train_trials = [t for t in all_trials if t["run_idx"] != test_run_idx]
        test_trials  = [t for t in all_trials if t["run_idx"] == test_run_idx]

        if len(train_trials) == 0 or len(test_trials) == 0:
            logger.warning(
                "  CV fold run %d: empty train or test set — skipping fold",
                test_run_idx + 1
            )
            continue

        logger.info(
            "  CV fold: test_run=%d | train=%d trials | test=%d trials",
            test_run_idx + 1, len(train_trials), len(test_trials)
        )

        # Train biLSTM
        model = train_bilstm(
            train_features=[t["features"] for t in train_trials],
            train_audio=[t["audio"]    for t in train_trials],
            n_features=n_features,
            device=device,
            n_epochs=n_epochs,
        )

        # Predict on test set
        fold_preds   = [predict_bilstm(model, t["features"], device) for t in test_trials]
        fold_actuals = [t["audio"] for t in test_trials]

        all_predicted.extend(fold_preds)
        all_actual.extend(fold_actuals)

        # Per-fold metrics
        fold_time_r = float(np.nanmean([
            compute_pearson_time(p, a) for p, a in zip(fold_preds, fold_actuals)
        ]))
        fold_freq_r = float(np.nanmean([
            compute_pearson_freq(p, a) for p, a in zip(fold_preds, fold_actuals)
        ]))
        fold_ssim = float(np.nanmean([
            compute_ssim(p, a) for p, a in zip(fold_preds, fold_actuals)
        ]))

        cv_metrics.append({
            "test_run":        test_run_idx + 1,
            "n_test_trials":   len(test_trials),
            "time_pearson_r":  fold_time_r,
            "freq_pearson_r":  fold_freq_r,
            "ssim":            fold_ssim,
        })
        logger.info(
            "  Fold run %d: time_r=%.4f  freq_r=%.4f  ssim=%.4f",
            test_run_idx + 1, fold_time_r, fold_freq_r, fold_ssim,
        )

    if len(all_predicted) == 0:
        logger.error("No predictions for %s", subject_id)
        return {"subject": subject_id, "error": "no predictions", "cv_folds": cv_metrics}

    # ===========================================================
    # Step 5: Overall evaluation
    # ===========================================================
    logger.info("  Computing overall metrics ...")

    mean_time_r = float(np.nanmean([
        compute_pearson_time(p, a) for p, a in zip(all_predicted, all_actual)
    ]))
    mean_freq_r = float(np.nanmean([
        compute_pearson_freq(p, a) for p, a in zip(all_predicted, all_actual)
    ]))
    mean_ssim = float(np.nanmean([
        compute_ssim(p, a) for p, a in zip(all_predicted, all_actual)
    ]))

    logger.info("  Computing rank accuracy (n_bootstrap=%d) ...", n_bootstrap)
    rank_res = compute_rank_accuracy(all_predicted, all_actual, n_bootstrap=n_bootstrap)

    result = {
        "subject":              subject_id,
        "n_valid_trials":       len(all_trials),
        "n_unique_stimuli":     n_unique_stim,
        "dipole_locations_mni": dipoles_mm.tolist(),
        "dipole_source":        dipole_source,
        "n_features":           n_features,
        "cv_folds":             cv_metrics,
        "mean_time_pearson_r":  mean_time_r,
        "mean_freq_pearson_r":  mean_freq_r,
        "mean_ssim":            mean_ssim,
        "rank_accuracy":        rank_res,
    }

    logger.info(
        "  %s RESULTS: time_r=%.4f | freq_r=%.4f | ssim=%.4f | "
        "rank=%.4f (p=%.4f)",
        subject_id,
        mean_time_r, mean_freq_r, mean_ssim,
        rank_res.get("mean_rank", float("nan")),
        rank_res.get("p_value",   float("nan")),
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir
    stimuli_dir = data_dir / "stimuli" / "generated"
    output_dir  = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger.info("Device:       %s", device)
    logger.info("Data dir:     %s", data_dir)
    logger.info("Stimuli dir:  %s", stimuli_dir)
    logger.info("Output dir:   %s", output_dir)
    logger.info("Skip fMRI:    %s", args.skip_fmri)

    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)
    if not stimuli_dir.exists():
        logger.error("Stimuli directory not found: %s", stimuli_dir)
        sys.exit(1)

    subjects = get_subject_list(data_dir, args.subjects, args.n_subjects)
    logger.info("Subjects (%d): %s", len(subjects), subjects)

    all_results = {}

    for subj_id in tqdm(subjects, desc="Subjects"):
        try:
            res = process_subject(
                subject_id=subj_id,
                data_dir=data_dir,
                stimuli_dir=stimuli_dir,
                device=device,
                n_epochs=args.epochs,
                n_bootstrap=args.n_bootstrap,
                skip_fmri=args.skip_fmri,
            )
        except Exception as exc:
            logger.exception("Unhandled error for %s: %s", subj_id, exc)
            res = {"subject": subj_id, "error": str(exc)}

        all_results[subj_id] = res

    # Save results in paper format
    out_path = output_dir / "subject_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # Summary table
    logger.info("\n%s", "=" * 72)
    logger.info("SUMMARY")
    logger.info("=" * 72)
    logger.info(
        "%-10s  %10s  %10s  %8s  %10s  %8s",
        "Subject", "time_r", "freq_r", "SSIM", "rank_acc", "p-value"
    )
    logger.info("-" * 72)
    for subj_id, res in all_results.items():
        if "error" in res:
            logger.info("%-10s  ERROR: %s", subj_id, res["error"])
        else:
            rk = res.get("rank_accuracy", {})
            logger.info(
                "%-10s  %10.4f  %10.4f  %8.4f  %10.4f  %8.4f",
                subj_id,
                res.get("mean_time_pearson_r", float("nan")),
                res.get("mean_freq_pearson_r", float("nan")),
                res.get("mean_ssim",           float("nan")),
                rk.get("mean_rank",            float("nan")),
                rk.get("p_value",              float("nan")),
            )
    logger.info("=" * 72)

    return all_results


if __name__ == "__main__":
    main()

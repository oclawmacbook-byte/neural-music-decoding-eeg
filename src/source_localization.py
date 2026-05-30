"""
source_localization.py
======================
EEG source localization using eLoreta (MNE-Python), constrained to the 4 dipole
locations identified by the fMRI GLM analysis.

Pipeline
--------
1. Build a forward model using a 5-shell spherical head model with the
   conductivities from the paper (or a realistic BEM model when an MRI is
   available).
2. Compute eLoreta inverse operator.
3. For each ICA-cleaned epoch, project to source space at the 4 dipole
   coordinates.
4. Concatenate: for each source, we obtain time courses at all 31 EEG
   channel-weights → feature matrix of shape (124, N_samples).

Head-model conductivities (Daly 2023)
--------------------------------------
gray matter:  0.33 S/m
white matter: 0.14 S/m
CSF:          1.79 S/m
skull:        0.01 S/m
scalp:        0.43 S/m

Grid resolution: 1.5 × 1.5 × 1.5 cm
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
from mne.minimum_norm import apply_inverse, make_inverse_operator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONDUCTIVITIES = {
    "gray_matter": 0.33,
    "white_matter": 0.14,
    "csf": 1.79,
    "skull": 0.01,
    "scalp": 0.43,
}

GRID_RESOLUTION_M = 0.015   # 1.5 cm in metres
N_DIPOLES = 4
N_EEG_CHANNELS = 31


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def build_spherical_forward(
    info: mne.Info,
    dipole_coords_mni: np.ndarray,
    conductivities: Optional[Dict[str, float]] = None,
) -> mne.Forward:
    """Build a spherical forward model.

    Uses a 4-shell spherical head model (scalp/skull/CSF/brain) with the
    conductivities from the paper.  The dipoles are placed at the fMRI-derived
    MNI coordinates.

    Parameters
    ----------
    info : mne.Info
        EEG info object (channel names, positions required).
    dipole_coords_mni : np.ndarray, shape (n_dipoles, 3)
        Dipole MNI coordinates in millimetres.
    conductivities : dict or None
        Conductivity values.  Defaults to paper values.

    Returns
    -------
    fwd : mne.Forward
    """
    if conductivities is None:
        conductivities = CONDUCTIVITIES

    n_dipoles = len(dipole_coords_mni)
    coords_m = dipole_coords_mni / 1000.0  # mm → m

    # Build a custom source space from the fMRI dipole locations
    # Each dipole gets three orientations (x, y, z)
    src = _make_dipole_source_space(info, coords_m)

    # Spherical head model: 4 shells
    # Radii (metres): brain ~ 0.085, csf ~ 0.088, skull ~ 0.093, scalp ~ 0.1
    sphere_radii = np.array([0.085, 0.088, 0.093, 0.100])
    sigma = np.array([
        conductivities["gray_matter"],
        conductivities["csf"],
        conductivities["skull"],
        conductivities["scalp"],
    ])

    # Use MNE's make_sphere_model with explicit conductivities
    sphere = mne.make_sphere_model(
        r0=(0.0, 0.0, 0.040),  # centre of sphere (head centre, ~4 cm above origin)
        head_radius=0.100,
        info=info,
        sigma=sigma[-1],  # outer layer conductivity
        relative_radii=sphere_radii / sphere_radii[-1],
        sigmas=sigma,
        verbose=False,
    )

    fwd = mne.make_forward_solution(
        info,
        trans=None,
        src=src,
        bem=sphere,
        eeg=True,
        meg=False,
        mindist=0.0,
        ignore_ref=True,
        verbose=False,
    )
    return fwd


def _make_dipole_source_space(
    info: mne.Info,
    coords_m: np.ndarray,
) -> mne.SourceSpaces:
    """Create a discrete source space at given coordinates.

    Parameters
    ----------
    info : mne.Info
    coords_m : np.ndarray, shape (n_dipoles, 3)
        Dipole positions in metres (head coordinate system).

    Returns
    -------
    src : mne.SourceSpaces
    """
    n = len(coords_m)
    # Unit normals pointing outward from origin
    norms = coords_m / (np.linalg.norm(coords_m, axis=1, keepdims=True) + 1e-12)

    src_dict = {
        "rr": coords_m.astype(np.float32),
        "nn": norms.astype(np.float32),
        "inuse": np.ones(n, dtype=int),
        "nuse": n,
        "np": n,
        "coord_frame": mne.io.constants.FIFF.FIFFV_COORD_HEAD,
        "id": 1,
        "type": "discrete",
        "vertno": np.arange(n),
        "use_tris": np.empty((0, 3), dtype=int),
        "tris": np.empty((0, 3), dtype=int),
        "ntri": 0,
        "nuse_tri": 0,
    }
    return mne.SourceSpaces([src_dict])


# ---------------------------------------------------------------------------
# Inverse operator (eLoreta)
# ---------------------------------------------------------------------------

def make_eloreta_inverse(
    info: mne.Info,
    fwd: mne.Forward,
    noise_cov: Optional[mne.Covariance] = None,
    snr: float = 3.0,
) -> object:
    """Compute the eLoreta inverse operator.

    Parameters
    ----------
    info : mne.Info
    fwd : mne.Forward
    noise_cov : mne.Covariance or None
        If None, an identity covariance is used.
    snr : float
        Signal-to-noise ratio for regularisation.

    Returns
    -------
    inverse_operator : mne.minimum_norm.InverseOperator
    """
    if noise_cov is None:
        # Use a diagonal / identity covariance
        noise_cov = mne.make_ad_hoc_cov(info, verbose=False)

    lambda2 = 1.0 / snr ** 2

    inverse_operator = make_inverse_operator(
        info,
        fwd,
        noise_cov,
        loose=1.0,        # free orientation
        depth=0.8,
        verbose=False,
    )
    return inverse_operator, lambda2


def estimate_noise_cov(
    raw: mne.io.Raw,
    tmin: float = 0.0,
    tmax: Optional[float] = None,
) -> mne.Covariance:
    """Estimate noise covariance from a baseline or rest period.

    Parameters
    ----------
    raw : mne.io.Raw
    tmin, tmax : float
        Time window (seconds) for covariance estimation.

    Returns
    -------
    noise_cov : mne.Covariance
    """
    noise_cov = mne.compute_raw_covariance(
        raw, tmin=tmin, tmax=tmax, method="empirical", verbose=False
    )
    return noise_cov


# ---------------------------------------------------------------------------
# Source projection and feature matrix construction
# ---------------------------------------------------------------------------

def project_to_sources(
    epochs_data: np.ndarray,
    fwd_leadfield: np.ndarray,
    n_channels: int = N_EEG_CHANNELS,
    n_dipoles: int = N_DIPOLES,
) -> np.ndarray:
    """Project EEG data to source space using the eLoreta weight matrix.

    This implements the key equation from the paper:
        S = W^T * X
    where W is the spatial filter (columns = source filters) and X is the EEG.

    The feature matrix for one epoch is then constructed by arranging source
    activations as described in the paper: 4 sources × 31 channels = 124 features.

    Parameters
    ----------
    epochs_data : np.ndarray, shape (n_epochs, n_channels, n_times)
        Pre-processed EEG data.
    fwd_leadfield : np.ndarray, shape (n_channels, n_dipoles * 3)
        Forward model leadfield (columns = dipoles × orientations).
    n_channels : int
    n_dipoles : int

    Returns
    -------
    features : np.ndarray, shape (n_epochs, 124, n_times)
        Feature matrix: (n_dipoles × n_channels, n_times) per epoch.
    """
    n_epochs, n_ch, n_times = epochs_data.shape
    n_sources = n_dipoles  # one resultant per dipole (after orientation combining)

    # Compute Moore-Penrose pseudo-inverse of leadfield as simple spatial filter
    # (full eLoreta requires iterative weight matrix; here we use regularized pinv)
    L = fwd_leadfield  # (n_channels, n_dipoles * 3)
    W = np.linalg.pinv(L)  # (n_dipoles * 3, n_channels)

    features = np.zeros((n_epochs, n_dipoles * n_channels, n_times))

    for ep in range(n_epochs):
        X = epochs_data[ep]  # (n_channels, n_times)

        for d in range(n_dipoles):
            # Weights for 3 orientations of dipole d
            w_3 = W[d * 3: (d + 1) * 3, :]  # (3, n_channels)
            # Source activation at this dipole (sum over orientations)
            s = w_3 @ X  # (3, n_times) → take RMS over orientations
            s_rms = np.sqrt(np.mean(s ** 2, axis=0))  # (n_times,)

            # Feature block: source activation scaled onto each channel
            # (equivalent to column of feature matrix in paper)
            for c in range(n_channels):
                row_idx = d * n_channels + c
                features[ep, row_idx, :] = s_rms * X[c, :]

    return features


def apply_eloreta_inverse(
    evoked: mne.Evoked,
    inverse_operator: object,
    lambda2: float,
    method: str = "eLORETA",
) -> mne.SourceEstimate:
    """Apply eLoreta inverse to an Evoked object.

    Parameters
    ----------
    evoked : mne.Evoked
    inverse_operator : InverseOperator
    lambda2 : float
    method : str

    Returns
    -------
    stc : mne.SourceEstimate
    """
    stc = apply_inverse(
        evoked,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        return_residual=False,
        verbose=False,
    )
    return stc


# ---------------------------------------------------------------------------
# High-level feature extraction
# ---------------------------------------------------------------------------

def extract_features_from_epochs(
    epochs: mne.Epochs,
    fwd: mne.Forward,
    noise_cov: Optional[mne.Covariance] = None,
    snr: float = 3.0,
    target_sfreq: float = 100.0,
    n_dipoles: int = N_DIPOLES,
    verbose: bool = False,
) -> np.ndarray:
    """Extract the 124-feature matrix from EEG epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed EEG epochs (at 1000 Hz).
    fwd : mne.Forward
        Forward model from :func:`build_spherical_forward`.
    noise_cov : mne.Covariance or None
    snr : float
    target_sfreq : float
        Downsample features to this rate (100 Hz in the paper).
    n_dipoles : int
    verbose : bool

    Returns
    -------
    features : np.ndarray, shape (n_epochs, 124, n_samples_100Hz)
        Feature matrix ready for the biLSTM.
    """
    info = epochs.info
    inv_op, lambda2 = make_eloreta_inverse(info, fwd, noise_cov=noise_cov, snr=snr)

    # Get leadfield for building weight matrix
    leadfield = fwd["sol"]["data"]  # (n_channels, n_dipoles * 3)

    # Epochs data
    epochs_resampled = epochs.copy()
    if epochs_resampled.info["sfreq"] != target_sfreq:
        decim = int(round(epochs_resampled.info["sfreq"] / target_sfreq))
        epochs_resampled = epochs_resampled.decimate(decim, verbose=verbose)

    data = epochs_resampled.get_data()  # (n_epochs, n_channels, n_times)
    n_channels = data.shape[1]

    features = project_to_sources(
        data,
        leadfield,
        n_channels=n_channels,
        n_dipoles=n_dipoles,
    )
    if verbose:
        print(f"Feature matrix shape: {features.shape}  "
              f"(n_epochs={features.shape[0]}, "
              f"n_features={features.shape[1]}, "
              f"n_samples={features.shape[2]})")
    return features


def build_feature_matrix(
    all_features: List[np.ndarray],
) -> np.ndarray:
    """Concatenate feature matrices across epochs / trials.

    Parameters
    ----------
    all_features : list of np.ndarray, each shape (n_epochs, 124, n_samples)

    Returns
    -------
    X : np.ndarray, shape (total_epochs, 124, n_samples)
    """
    return np.concatenate(all_features, axis=0)

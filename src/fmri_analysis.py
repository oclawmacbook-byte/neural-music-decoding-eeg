"""
fmri_analysis.py
================
fMRI GLM analysis following Daly (2023).

Steps
-----
1. First-level GLM per subject per run (music-on HRF regressor vs. rest).
2. Second-level one-sample t-test across participants.
3. Threshold at p < 0.001 (uncorrected).
4. Greedy dipole selection: pick the 4 most spatially distinct voxels.
   - Criterion: each new dipole must be > 3 cm from all already-selected dipoles
   - Ordered by descending t-score

Paper head-model conductivities (S/m)
--------------------------------------
gray matter:  0.33
white matter: 0.14
CSF:          1.79
skull:        0.01
scalp:        0.43

Grid resolution: 1.5 × 1.5 × 1.5 cm
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib

from nilearn import image, plotting
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn.reporting import get_clusters_table


# ---------------------------------------------------------------------------
# Conductivity constants (from paper)
# ---------------------------------------------------------------------------

CONDUCTIVITIES = {
    "gray_matter": 0.33,
    "white_matter": 0.14,
    "csf": 1.79,
    "skull": 0.01,
    "scalp": 0.43,
}

GRID_RESOLUTION_CM = 1.5   # voxel side length (cm)
MIN_DIPOLE_DISTANCE_CM = 3.0  # minimum distance between selected dipoles (cm)
N_DIPOLES = 4               # number of source dipoles


# ---------------------------------------------------------------------------
# First-level GLM
# ---------------------------------------------------------------------------

def run_first_level_glm(
    bold_img: nib.Nifti1Image,
    events_df: pd.DataFrame,
    t_r: float,
    hrf_model: str = "spm",
    drift_model: str = "cosine",
    high_pass: float = 0.01,
    smoothing_fwhm: float = 6.0,
    noise_model: str = "ar1",
    verbose: int = 0,
) -> FirstLevelModel:
    """Fit a first-level GLM for one run of one subject.

    Parameters
    ----------
    bold_img : nibabel.Nifti1Image
        4-D BOLD image.
    events_df : pd.DataFrame
        Events with columns ``onset``, ``duration``, ``trial_type``.
        Expected trial_types: ``'music'``, ``'rest'``.
    t_r : float
        Repetition time (seconds).
    hrf_model : str
        HRF model identifier (passed to nilearn).
    drift_model : str
    high_pass : float
        High-pass filter cutoff (Hz).
    smoothing_fwhm : float
        Spatial smoothing FWHM (mm).
    noise_model : str
        AR noise model.
    verbose : int

    Returns
    -------
    glm : FirstLevelModel
        Fitted GLM.
    """
    n_scans = bold_img.shape[-1]
    frame_times = np.arange(n_scans) * t_r

    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events_df,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
    )

    glm = FirstLevelModel(
        t_r=t_r,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
        smoothing_fwhm=smoothing_fwhm,
        noise_model=noise_model,
        standardize=False,
        verbose=verbose,
    )
    glm.fit(bold_img, design_matrices=design_matrix)
    return glm


def compute_music_contrast(
    glm: FirstLevelModel,
    contrast_name: str = "music",
) -> nib.Nifti1Image:
    """Compute the music > rest t-statistic map.

    Parameters
    ----------
    glm : FirstLevelModel
        Fitted first-level GLM.
    contrast_name : str
        Name of the music condition column in the design matrix.

    Returns
    -------
    t_map : nibabel.Nifti1Image
    """
    # Build contrast vector: +1 for 'music', 0 elsewhere
    design_matrix = glm.design_matrices_[0]
    contrast_vector = np.zeros(design_matrix.shape[1])
    if contrast_name in design_matrix.columns:
        contrast_vector[design_matrix.columns.tolist().index(contrast_name)] = 1.0
    else:
        # Try to find a column that contains the name
        for i, col in enumerate(design_matrix.columns):
            if contrast_name.lower() in col.lower():
                contrast_vector[i] = 1.0
                break

    t_map = glm.compute_contrast(contrast_vector, output_type="stat")
    return t_map


# ---------------------------------------------------------------------------
# Subject-level analysis
# ---------------------------------------------------------------------------

def subject_first_level(
    subject_dir: Path,
    n_runs: int = 3,
    t_r: float = 2.0,
    smoothing_fwhm: float = 6.0,
    verbose: int = 0,
) -> List[nib.Nifti1Image]:
    """Run first-level GLM for all runs of one subject.

    Parameters
    ----------
    subject_dir : Path
    n_runs : int
    t_r : float
    smoothing_fwhm : float
    verbose : int

    Returns
    -------
    t_maps : list of nibabel.Nifti1Image
        One t-map per run.
    """
    from .data_loader import load_fmri

    t_maps = []
    for run in range(1, n_runs + 1):
        try:
            bold_img, events_array = load_fmri(subject_dir, run)
        except FileNotFoundError as e:
            warnings.warn(str(e))
            continue

        # Convert events array to DataFrame
        events_df = pd.DataFrame(
            events_array, columns=["onset", "duration", "trial_type_code"]
        )
        events_df["trial_type"] = events_df["trial_type_code"].map(
            {1: "music", 0: "rest"}
        )
        events_df = events_df[["onset", "duration", "trial_type"]]

        if events_df.empty:
            warnings.warn(f"No events for subject {subject_dir.name} run {run}.")
            continue

        glm = run_first_level_glm(
            bold_img=bold_img,
            events_df=events_df,
            t_r=t_r,
            smoothing_fwhm=smoothing_fwhm,
            verbose=verbose,
        )
        t_map = compute_music_contrast(glm)
        t_maps.append(t_map)

    return t_maps


def subject_mean_t_map(t_maps: List[nib.Nifti1Image]) -> nib.Nifti1Image:
    """Average multiple t-maps (across runs) for one subject."""
    if len(t_maps) == 1:
        return t_maps[0]
    return image.mean_img(t_maps)


# ---------------------------------------------------------------------------
# Group-level (second-level) analysis
# ---------------------------------------------------------------------------

def run_group_analysis(
    subject_t_maps: List[nib.Nifti1Image],
    verbose: int = 0,
) -> nib.Nifti1Image:
    """Run second-level one-sample t-test across subjects.

    Parameters
    ----------
    subject_t_maps : list of nibabel.Nifti1Image
        One (mean) t-map per subject; must all be in the same space.
    verbose : int

    Returns
    -------
    group_t_map : nibabel.Nifti1Image
    """
    # Resample to a common space if needed
    ref = subject_t_maps[0]
    resampled = [image.resample_to_img(m, ref, interpolation="linear")
                 for m in subject_t_maps]

    design_matrix = pd.DataFrame(
        {"intercept": np.ones(len(resampled))}
    )

    model = SecondLevelModel(verbose=verbose)
    model.fit(resampled, design_matrix=design_matrix)
    group_t_map = model.compute_contrast("intercept", output_type="stat")
    return group_t_map


# ---------------------------------------------------------------------------
# Dipole selection
# ---------------------------------------------------------------------------

def threshold_t_map(
    t_map: nib.Nifti1Image,
    p_threshold: float = 0.001,
    two_sided: bool = False,
) -> Tuple[nib.Nifti1Image, float]:
    """Threshold a t-statistic map at a given (uncorrected) p-value.

    For large-sample approximation we use df=1000.

    Parameters
    ----------
    t_map : nibabel.Nifti1Image
    p_threshold : float
    two_sided : bool

    Returns
    -------
    thresholded_map : nibabel.Nifti1Image
    t_threshold : float
        The t-value corresponding to *p_threshold*.
    """
    from scipy.stats import t as t_dist

    df = 1000  # approximation; use actual df if available
    t_threshold = t_dist.ppf(1.0 - p_threshold, df=df)
    if two_sided:
        t_threshold = abs(t_dist.ppf(p_threshold / 2.0, df=df))

    t_data = t_map.get_fdata()
    thresholded = np.where(t_data > t_threshold, t_data, 0.0)
    return nib.Nifti1Image(thresholded, t_map.affine, t_map.header), t_threshold


def voxel_mni_coordinates(
    t_map: nib.Nifti1Image,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract voxel t-scores and their MNI coordinates.

    Parameters
    ----------
    t_map : nibabel.Nifti1Image

    Returns
    -------
    coords_mni : np.ndarray, shape (N, 3)
        MNI coordinates (mm) of all non-zero voxels.
    t_scores : np.ndarray, shape (N,)
        Corresponding t-scores.
    """
    t_data = t_map.get_fdata()
    affine = t_map.affine

    # Indices of non-zero voxels
    vox_indices = np.array(np.where(t_data > 0)).T  # (N, 3)
    if len(vox_indices) == 0:
        return np.empty((0, 3)), np.empty(0)

    # Convert to MNI
    vox_homogeneous = np.column_stack([vox_indices, np.ones(len(vox_indices))])
    mni_homogeneous = (affine @ vox_homogeneous.T).T
    coords_mni = mni_homogeneous[:, :3]

    t_scores = t_data[vox_indices[:, 0], vox_indices[:, 1], vox_indices[:, 2]]
    return coords_mni, t_scores


def select_dipole_locations(
    t_map: nib.Nifti1Image,
    n_dipoles: int = N_DIPOLES,
    min_distance_mm: float = MIN_DIPOLE_DISTANCE_CM * 10.0,
    p_threshold: float = 0.001,
    verbose: bool = True,
) -> np.ndarray:
    """Greedily select *n_dipoles* spatially distinct peak voxels.

    Algorithm (from the paper):
      1. Threshold the group t-map at p < *p_threshold* (uncorrected).
      2. Sort surviving voxels by descending t-score.
      3. Pick the top voxel as dipole 1.
      4. Iterate: pick the next voxel whose distance from ALL already-selected
         voxels is > *min_distance_mm*.
      5. Repeat until *n_dipoles* are selected.

    Parameters
    ----------
    t_map : nibabel.Nifti1Image
        Group-level t-statistic map (unthresholded).
    n_dipoles : int
    min_distance_mm : float
        Minimum Euclidean distance (mm) between selected dipoles.
    p_threshold : float
        Significance threshold for initial masking.
    verbose : bool

    Returns
    -------
    dipole_coords : np.ndarray, shape (n_dipoles, 3)
        MNI coordinates (mm) of the selected dipoles.
    """
    thresholded_map, t_thresh = threshold_t_map(t_map, p_threshold=p_threshold)
    if verbose:
        print(f"T-threshold at p < {p_threshold}: t > {t_thresh:.2f}")

    coords_mni, t_scores = voxel_mni_coordinates(thresholded_map)
    if len(coords_mni) == 0:
        raise ValueError(
            f"No voxels survived thresholding at p < {p_threshold}. "
            "Try relaxing the threshold."
        )

    # Sort by descending t-score
    order = np.argsort(t_scores)[::-1]
    coords_sorted = coords_mni[order]
    scores_sorted = t_scores[order]

    selected: List[np.ndarray] = []
    for i, (coord, score) in enumerate(zip(coords_sorted, scores_sorted)):
        if len(selected) == 0:
            selected.append(coord)
            if verbose:
                print(f"Dipole 1: MNI {coord} mm  (t = {score:.2f})")
            continue

        distances = np.linalg.norm(
            np.array(selected) - coord[np.newaxis, :], axis=1
        )
        if np.all(distances > min_distance_mm):
            selected.append(coord)
            if verbose:
                print(f"Dipole {len(selected)}: MNI {coord} mm  (t = {score:.2f})")

        if len(selected) == n_dipoles:
            break

    if len(selected) < n_dipoles:
        warnings.warn(
            f"Only {len(selected)} dipoles found (requested {n_dipoles}). "
            "Consider reducing min_distance_mm or p_threshold."
        )

    return np.array(selected)


# ---------------------------------------------------------------------------
# Full pipeline helper
# ---------------------------------------------------------------------------

def run_full_fmri_pipeline(
    dataset_dir: Path,
    subject_dirs: List[Path],
    n_runs: int = 3,
    t_r: float = 2.0,
    smoothing_fwhm: float = 6.0,
    n_dipoles: int = N_DIPOLES,
    min_distance_mm: float = MIN_DIPOLE_DISTANCE_CM * 10.0,
    p_threshold: float = 0.001,
    verbose: bool = True,
) -> Tuple[np.ndarray, nib.Nifti1Image, List[nib.Nifti1Image]]:
    """Run the complete fMRI analysis and return dipole locations.

    Parameters
    ----------
    dataset_dir : Path
        Root OpenNeuro dataset directory (unused, kept for API symmetry).
    subject_dirs : list of Path
        One Path per subject directory.
    n_runs, t_r, smoothing_fwhm, n_dipoles, min_distance_mm, p_threshold
    verbose : bool

    Returns
    -------
    dipole_coords : np.ndarray, shape (n_dipoles, 3)
    group_t_map : nibabel.Nifti1Image
    subject_t_maps : list of nibabel.Nifti1Image
    """
    subject_t_maps = []
    for subj_dir in subject_dirs:
        if verbose:
            print(f"First-level GLM: {subj_dir.name}")
        run_t_maps = subject_first_level(
            subj_dir,
            n_runs=n_runs,
            t_r=t_r,
            smoothing_fwhm=smoothing_fwhm,
            verbose=0,
        )
        if run_t_maps:
            subject_t_maps.append(subject_mean_t_map(run_t_maps))

    if not subject_t_maps:
        raise RuntimeError("No subject t-maps computed. Check data paths.")

    if verbose:
        print(f"Group analysis over {len(subject_t_maps)} subjects …")
    group_t_map = run_group_analysis(subject_t_maps, verbose=0)

    if verbose:
        print("Selecting dipole locations …")
    dipole_coords = select_dipole_locations(
        group_t_map,
        n_dipoles=n_dipoles,
        min_distance_mm=min_distance_mm,
        p_threshold=p_threshold,
        verbose=verbose,
    )

    return dipole_coords, group_t_map, subject_t_maps

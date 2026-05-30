#!/usr/bin/env python3
"""
run_eegfmri.py
==============
Run the full EEG-fMRI analysis pipeline from Daly (2023).

Steps
-----
1. fMRI GLM → group analysis → 4 dipole locations (MNI mm)
2. EEG preprocessing (bandpass, ICA)
3. eLoreta source localisation → 124-feature matrix
4. 4-layer biLSTM training with 3-fold (leave-one-run-out) CV
5. Evaluation: r_time, r_freq, SSIM, rank accuracy

Usage
-----
    python scripts/run_eegfmri.py \\
        --data_dir data/ds002691 \\
        --output_dir results/eegfmri \\
        --n_subjects 18

    # Skip fMRI GLM (use saved dipoles)
    python scripts/run_eegfmri.py \\
        --data_dir data/ds002691 \\
        --output_dir results/eegfmri \\
        --dipoles_file results/eegfmri/dipole_locations.npy

    # Dry run with 2 subjects, 10 training epochs
    python scripts/run_eegfmri.py \\
        --data_dir data/ds002691 \\
        --output_dir results/eegfmri_dryrun \\
        --n_subjects 2 \\
        --n_train_epochs 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full EEG-fMRI music decoding pipeline (Daly 2023)."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to the ds002691 dataset root.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/eegfmri"),
        help="Directory to write results, models, and dipole locations.",
    )
    parser.add_argument(
        "--n_subjects",
        type=int,
        default=18,
        help="Number of subjects to process (default: 18).",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=3,
        help="Number of EEG/fMRI runs per subject.",
    )
    parser.add_argument(
        "--dipoles_file",
        type=Path,
        default=None,
        help="Path to a .npy file with pre-computed dipole MNI coordinates. "
             "If provided, skips the fMRI GLM step.",
    )
    parser.add_argument(
        "--n_train_epochs",
        type=int,
        default=100,
        help="Number of training epochs for the biLSTM.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for biLSTM training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device ('cpu', 'cuda', 'mps').  Auto-detected if None.",
    )
    parser.add_argument(
        "--fmri_tr",
        type=float,
        default=1.5,
        help="fMRI repetition time in seconds.",
    )
    parser.add_argument(
        "--eeg_format",
        type=str,
        default="brainvision",
        choices=["brainvision", "eeglab", "auto"],
        help="EEG file format.",
    )
    parser.add_argument(
        "--no_fmri",
        action="store_true",
        help="Skip fMRI GLM entirely.  Requires --dipoles_file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    # Validate
    if args.no_fmri and args.dipoles_file is None:
        print("Error: --no_fmri requires --dipoles_file.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("  Neural decoding of music from EEG — EEG-fMRI pipeline")
        print("  Daly (2023), Scientific Reports")
        print("=" * 60)
        print(f"  Dataset  : {args.data_dir}")
        print(f"  Output   : {args.output_dir}")
        print(f"  Subjects : {args.n_subjects}")
        print(f"  Runs     : {args.n_runs}")
        print(f"  Epochs   : {args.n_train_epochs}")
        print()

    # Load pre-computed dipoles if provided
    dipole_coords = None
    if args.dipoles_file is not None:
        dipole_coords = np.load(str(args.dipoles_file))
        if verbose:
            print(f"Loaded dipole locations from {args.dipoles_file}:")
            for i, d in enumerate(dipole_coords):
                print(f"  Dipole {i+1}: MNI {d} mm")

    # Run pipeline
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.pipeline import run_eegfmri_pipeline

    all_results = run_eegfmri_pipeline(
        dataset_dir=args.data_dir,
        output_dir=args.output_dir,
        n_subjects=args.n_subjects,
        n_runs=args.n_runs,
        dipole_coords_mni=dipole_coords,
        fmt=args.eeg_format,
        device=args.device,
        n_train_epochs=args.n_train_epochs,
        verbose=verbose,
    )

    # Serialise results (excluding non-serialisable numpy arrays)
    results_json = _serialise_results(all_results)
    out_file = args.output_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(results_json, f, indent=2)
    if verbose:
        print(f"\nResults saved to {out_file}")


def _serialise_results(results: dict) -> dict:
    """Recursively convert numpy types for JSON serialisation."""
    out = {}
    for k, v in results.items():
        if isinstance(v, dict):
            out[k] = _serialise_results(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = float(v)
        elif isinstance(v, list):
            out[k] = v
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


if __name__ == "__main__":
    main()

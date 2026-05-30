#!/usr/bin/env python3
"""
run_eegonly.py
==============
Run the EEG-only analysis pipeline using group-averaged dipole locations
derived from the EEG-fMRI dataset (ds002691).

This corresponds to the second experiment in Daly (2023), applied to
ds002137 (19 EEG-only participants).

Usage
-----
    # Use dipole locations saved from the EEG-fMRI pipeline
    python scripts/run_eegonly.py \\
        --data_dir data/ds002137 \\
        --dipoles_file results/eegfmri/dipole_locations.npy \\
        --output_dir results/eegonly

    # Provide dipole coordinates directly (MNI mm, 4 dipoles)
    python scripts/run_eegonly.py \\
        --data_dir data/ds002137 \\
        --dipoles "-20,0,50  20,0,50  0,-20,50  0,20,50" \\
        --output_dir results/eegonly

    # Quick test with 2 subjects
    python scripts/run_eegonly.py \\
        --data_dir data/ds002137 \\
        --dipoles_file results/eegfmri/dipole_locations.npy \\
        --output_dir results/eegonly_test \\
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
        description="EEG-only music decoding pipeline (Daly 2023, ds002137)."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to the ds002137 (EEG-only) dataset root.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/eegonly"),
        help="Directory to write results and model checkpoints.",
    )

    dipole_group = parser.add_mutually_exclusive_group(required=True)
    dipole_group.add_argument(
        "--dipoles_file",
        type=Path,
        default=None,
        help="Path to .npy file with dipole MNI coordinates (shape 4×3, mm).",
    )
    dipole_group.add_argument(
        "--dipoles",
        type=str,
        default=None,
        help='Dipole MNI coordinates as a string, e.g. "-20,0,50  20,0,50  0,-20,50  0,20,50".',
    )

    parser.add_argument(
        "--n_subjects",
        type=int,
        default=19,
        help="Number of subjects to process.",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=3,
        help="Number of EEG runs per subject.",
    )
    parser.add_argument(
        "--n_train_epochs",
        type=int,
        default=100,
        help="Number of biLSTM training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device.  Auto-detected if None.",
    )
    parser.add_argument(
        "--eeg_format",
        type=str,
        default="eeglab",
        choices=["brainvision", "eeglab", "auto"],
        help="EEG file format (ds002137 uses EEGLAB .set).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dipole coordinates helper
# ---------------------------------------------------------------------------

def parse_dipole_string(s: str) -> np.ndarray:
    """Parse a string like '-20,0,50  20,0,50  0,-20,50  0,20,50' into (4,3)."""
    rows = []
    for token in s.strip().split():
        parts = token.split(",")
        if len(parts) != 3:
            raise ValueError(f"Expected 3 comma-separated values, got: {token!r}")
        rows.append([float(p) for p in parts])
    if len(rows) != 4:
        raise ValueError(f"Expected 4 dipole rows, got {len(rows)}.")
    return np.array(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    # Load dipole coordinates
    if args.dipoles_file is not None:
        dipole_coords = np.load(str(args.dipoles_file))
    else:
        dipole_coords = parse_dipole_string(args.dipoles)

    if dipole_coords.shape != (4, 3):
        print(f"Error: dipole array shape should be (4, 3), got {dipole_coords.shape}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("  Neural decoding of music from EEG — EEG-only pipeline")
        print("  Daly (2023), Scientific Reports")
        print("=" * 60)
        print(f"  Dataset  : {args.data_dir}")
        print(f"  Output   : {args.output_dir}")
        print(f"  Subjects : {args.n_subjects}")
        print(f"  Runs     : {args.n_runs}")
        print(f"  Epochs   : {args.n_train_epochs}")
        print(f"  Dipoles  :")
        for i, d in enumerate(dipole_coords):
            print(f"    {i+1}: MNI {d} mm")
        print()

    # Run pipeline
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.pipeline import run_eegonly_pipeline

    all_results = run_eegonly_pipeline(
        dataset_dir=args.data_dir,
        output_dir=args.output_dir,
        dipole_coords_mni=dipole_coords,
        n_subjects=args.n_subjects,
        n_runs=args.n_runs,
        fmt=args.eeg_format,
        device=args.device,
        n_train_epochs=args.n_train_epochs,
        verbose=verbose,
    )

    # Save results
    results_json = _serialise_results(all_results)
    out_file = args.output_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(results_json, f, indent=2)
    if verbose:
        print(f"\nResults saved to {out_file}")

    # Also save the dipole coordinates used
    np.save(str(args.output_dir / "dipole_locations.npy"), dipole_coords)


def _serialise_results(results: dict) -> dict:
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

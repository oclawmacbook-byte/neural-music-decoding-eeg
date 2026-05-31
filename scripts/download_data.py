#!/usr/bin/env python3
"""
download_data.py
================
Download OpenNeuro datasets used by Daly (2023).

Supports two methods:
1. datalad  (recommended — supports partial/lazy download)
2. openneuro-py  (direct HTTP download)

Usage
-----
    python scripts/download_data.py --dataset ds002691 --output data/ds002691
    python scripts/download_data.py --dataset ds002137 --output data/ds002137

    # Download only specific subjects
    python scripts/download_data.py --dataset ds002691 --output data/ds002691 \
        --subjects sub-01 sub-02 sub-03

    # Use openneuro-py instead of datalad
    python scripts/download_data.py --dataset ds002691 --output data/ds002691 \
        --method openneuro
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "ds002725": {
        "name": "EEG-fMRI music dataset (joint EEG-fMRI during affective music listening)",
        "n_subjects": 21,
        "n_runs": 3,
        "openneuro_url": "https://openneuro.org/datasets/ds002725",
        "snapshot": "1.0.0",
        "doi": "10.18112/openneuro.ds002725.v1.0.0",
    },
    "ds002721": {
        "name": "EEG-only music dataset (EEG during affective music listening)",
        "n_subjects": 20,
        "n_runs": 3,
        "openneuro_url": "https://openneuro.org/datasets/ds002721",
        "snapshot": "1.0.1",
        "doi": "10.18112/openneuro.ds002721.v1.0.1",
    },
}


# ---------------------------------------------------------------------------
# Download methods
# ---------------------------------------------------------------------------

def download_with_datalad(
    dataset_id: str,
    output_dir: Path,
    subjects: list[str] | None = None,
    snapshot: str = "1.0.0",
) -> None:
    """Install (clone) and optionally get data using datalad.

    Parameters
    ----------
    dataset_id : str  e.g. 'ds002691'
    output_dir : Path
    subjects : list of str or None  — if None, get all
    snapshot : str
    """
    try:
        import datalad.api as dl
    except ImportError:
        print("datalad not found.  Install with:  pip install datalad")
        sys.exit(1)

    openneuro_url = f"https://github.com/OpenNeuroDatasets/{dataset_id}.git"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cloning {dataset_id} from OpenNeuro …")
    ds = dl.install(path=str(output_dir), source=openneuro_url)

    if subjects is None:
        print("Downloading all subject data …")
        ds.get(path=str(output_dir))
    else:
        for sub in subjects:
            sub_dir = output_dir / sub
            if sub_dir.exists():
                print(f"Getting data for {sub} …")
                ds.get(path=str(sub_dir))
            else:
                print(f"Warning: {sub_dir} not found in cloned dataset.")

        # Always get stimuli
        stimuli_dir = output_dir / "stimuli"
        if stimuli_dir.exists():
            print("Getting stimuli …")
            ds.get(path=str(stimuli_dir))


def download_with_openneuro_py(
    dataset_id: str,
    output_dir: Path,
    subjects: list[str] | None = None,
    snapshot: str = "1.0.0",
) -> None:
    """Download using openneuro-py (direct HTTP)."""
    try:
        import openneuro
    except ImportError:
        print("openneuro not found.  Install with:  pip install openneuro-py")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if subjects is not None:
        include = subjects + ["stimuli", "dataset_description.json",
                              "participants.tsv", "README"]
    else:
        include = None

    openneuro.download(
        dataset=dataset_id,
        version=snapshot,
        target_dir=str(output_dir),
        include=include,
    )


def download_with_aws(
    dataset_id: str,
    output_dir: Path,
    subjects: list[str] | None = None,
) -> None:
    """Download from OpenNeuro S3 bucket using the AWS CLI.

    Requires `aws` CLI installed and configured (no credentials needed for
    public buckets).
    """
    s3_base = f"s3://openneuro.org/{dataset_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if subjects is None:
        cmd = ["aws", "s3", "sync", "--no-sign-request",
               s3_base, str(output_dir)]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        # Download each subject folder
        for sub in subjects:
            cmd = ["aws", "s3", "sync", "--no-sign-request",
                   f"{s3_base}/{sub}", str(output_dir / sub)]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        # Stimuli
        cmd = ["aws", "s3", "sync", "--no-sign-request",
               f"{s3_base}/stimuli", str(output_dir / "stimuli")]
        subprocess.run(cmd, check=True)

        # Top-level metadata
        for f in ["dataset_description.json", "participants.tsv",
                  "README", "task-music_bold.json"]:
            cmd = ["aws", "s3", "cp", "--no-sign-request",
                   f"{s3_base}/{f}", str(output_dir / f)]
            subprocess.run(cmd, check=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OpenNeuro datasets for Daly (2023) replication."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        required=True,
        help="Dataset identifier (ds002725 = EEG-fMRI, ds002721 = EEG-only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subject IDs to download (e.g. sub-01 sub-02). "
             "If omitted, all subjects are downloaded.",
    )
    parser.add_argument(
        "--method",
        choices=["datalad", "openneuro", "aws"],
        default="datalad",
        help="Download method.",
    )
    parser.add_argument(
        "--snapshot",
        default=None,
        help="Dataset snapshot/version (default: latest).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = DATASETS[args.dataset]
    snapshot = args.snapshot or info["snapshot"]

    print(f"\nDataset : {args.dataset}  —  {info['name']}")
    print(f"Method  : {args.method}")
    print(f"Output  : {args.output}")
    print(f"Snapshot: {snapshot}")
    if args.subjects:
        print(f"Subjects: {args.subjects}")
    else:
        print(f"Subjects: all ({info['n_subjects']})")
    print()

    if args.method == "datalad":
        download_with_datalad(
            args.dataset, args.output, args.subjects, snapshot
        )
    elif args.method == "openneuro":
        download_with_openneuro_py(
            args.dataset, args.output, args.subjects, snapshot
        )
    elif args.method == "aws":
        download_with_aws(args.dataset, args.output, args.subjects)

    print(f"\nDone.  Data saved to {args.output}")


if __name__ == "__main__":
    main()

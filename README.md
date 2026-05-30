# Neural Decoding of Music from EEG

Reproduction of Daly (2023) "Neural decoding of music from the EEG"  
*Scientific Reports*, DOI: [10.1038/s41598-022-27361-x](https://doi.org/10.1038/s41598-022-27361-x)

---

## Overview

This repository implements the full analysis pipeline described in Daly (2023), which demonstrates
that music being listened to by a participant can be reconstructed directly from EEG signals.
The approach fuses joint EEG-fMRI data to identify music-responsive brain regions, uses eLoreta
source localization to extract neural signals from those regions, and trains a bidirectional LSTM
to reconstruct the music waveform.

### Key contributions from the paper
- Joint EEG-fMRI recordings during music listening reveal music-responsive voxels via GLM.
- eLoreta projects scalp EEG onto 4 fMRI-identified source dipoles, yielding 124 features
  (4 sources × 31 channels).
- A 4-layer bidirectional LSTM (250 hidden units) reconstructs the 100 Hz audio envelope.
- Evaluation: Pearson correlation (time and frequency domains), SSIM of spectrograms,
  and rank accuracy with 4000-shuffle bootstrap significance test.

---

## Datasets

### Dataset 1 — EEG-fMRI (ds002691)
- 18 participants, 3 runs each
- 36 piano music pieces (~40 s each), 3 repetitions
- OpenNeuro: <https://openneuro.org/datasets/ds002691>

### Dataset 2 — EEG-only (ds002137)
- 19 participants
- Similar paradigm, no fMRI
- OpenNeuro: <https://openneuro.org/datasets/ds002137>

---

## Pipeline

```
Raw EEG + fMRI
     │
     ├─── fMRI GLM (music vs. rest) ──► top-4 cluster centres (dipole locations)
     │
     ├─── EEG preprocessing (bandpass, ICA artefact removal, resample to 1000 Hz)
     │
     ├─── eLoreta source localisation at 4 dipole locations
     │         → feature matrix  124 × N_samples  (4 sources × 31 channels)
     │         → downsample to 100 Hz
     │
     ├─── 4-layer biLSTM  (input 124, hidden 250)  ──► reconstructed audio
     │
     └─── Evaluation: r_time, r_freq, SSIM, rank-accuracy (bootstrap p)
```

### fMRI dipole selection algorithm
1. Fit first-level GLM (music-on vs. music-off HRF regressor) for every participant.
2. Run second-level one-sample t-test across participants.
3. Threshold at p < 0.001 (uncorrected); identify surviving voxels.
4. Greedily pick the voxel with the highest t-score; then iteratively add the voxel with the
   next-highest t-score whose MNI distance from all already-selected voxels is > 3 cm.
   Stop after 4 dipoles.

### Head model conductivities (from paper)
| Compartment | Conductivity (S/m) |
|-------------|-------------------|
| Gray matter | 0.33 |
| White matter | 0.14 |
| CSF | 1.79 |
| Skull | 0.01 |
| Scalp | 0.43 |

Grid resolution: 1.5 × 1.5 × 1.5 cm

---

## Installation

```bash
git clone <this-repo>
cd neural-music-decoding-eeg
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data Download

```bash
# Install datalad (recommended)
pip install datalad

# Download EEG-fMRI dataset
python scripts/download_data.py --dataset ds002691 --output data/ds002691

# Download EEG-only dataset
python scripts/download_data.py --dataset ds002137 --output data/ds002137
```

Alternatively, use the OpenNeuro CLI:
```bash
npm install -g @openneuro/cli
openneuro download --snapshot 1.0.0 ds002691 data/ds002691
```

---

## Running the Analysis

### Full EEG-fMRI pipeline
```bash
python scripts/run_eegfmri.py \
    --data_dir data/ds002691 \
    --output_dir results/eegfmri \
    --n_subjects 18
```

### EEG-only pipeline (uses group-averaged dipole locations)
```bash
python scripts/run_eegonly.py \
    --data_dir data/ds002137 \
    --dipoles_file results/eegfmri/dipole_locations.npy \
    --output_dir results/eegonly \
    --n_subjects 19
```

### Interactive demo
```bash
jupyter notebook notebooks/analysis_demo.ipynb
```

---

## Repository Structure

```
neural-music-decoding-eeg/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Load EEG, fMRI, and audio data
│   ├── eeg_preprocessing.py    # Bandpass filter, ICA artefact removal
│   ├── fmri_analysis.py        # GLM, group analysis, dipole selection
│   ├── source_localization.py  # eLoreta at fMRI dipoles → feature matrix
│   ├── models.py               # PyTorch 4-layer biLSTM
│   ├── evaluation.py           # r_time, r_freq, SSIM, rank accuracy
│   └── pipeline.py             # 3×3 cross-fold orchestration
├── scripts/
│   ├── download_data.py
│   ├── run_eegfmri.py
│   └── run_eegonly.py
└── notebooks/
    └── analysis_demo.ipynb
```

---

## Citation

```bibtex
@article{Daly2023,
  title   = {Neural decoding of music from the {EEG}},
  author  = {Daly, Ian},
  journal = {Scientific Reports},
  volume  = {13},
  pages   = {624},
  year    = {2023},
  doi     = {10.1038/s41598-022-27361-x}
}
```

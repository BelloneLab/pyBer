# Fiber Photometry Processing GUI

A desktop GUI for visualizing photometry recordings, cleaning artifacts, filtering/resampling, baseline estimation, motion-correction, and exporting processed traces for downstream analysis.

This project is designed for efficient exploratory QC (preview in the GUI) while keeping processing logic deterministic and scriptable (core functions live in `analysis_core.py`).

---

## Key Features

### Data IO
- Supports raw data in .doric, .h5 or .csv.
- Multi-channel support (e.g., AIN01, AIN02, DIO channels …).
- Optional alignment of analog traces to the DigitalIO timebase when a DIO is selected.

### Artifact Handling
- Artifact detection on raw 465 using derivative thresholding (`dx`) and MAD:
  - **Global MAD (dx)**: one threshold for the full trace
  - **Adaptive MAD (windowed)**: windowed thresholds for nonstationary noise
- Optional **padding** around detected artifacts to remove spillover.
- Manual artifact masking by user-defined time regions.
- Masked samples are replaced via **linear interpolation** to preserve time alignment.

### Signal Conditioning
- Zero-phase low-pass filtering (Butterworth SOS via `sosfiltfilt`).
- Joint decimation/resampling of 465 and 405 to a **target sampling rate** using `resample_poly`.

### Baseline Estimation
Baseline is computed after filtering and resampling using **pybaselines**:
- `asls`, `arpls`, `airpls`
- tunable smoothing parameters (lambda, diff order, iterations, tolerance)

### Output Modes (7)
The GUI exposes seven explicit output definitions:

1. **dFF (non motion corrected)**  
   `dFF = (signal_filtered - signal_baseline) / signal_baseline`

2. **zscore (non motion corrected)**  
   `zscore(dFF_nonMC)`

3. **dFF (motion corrected via subtraction)**  
   `dFF_mc = dFF_signal - dFF_ref`  
   where each dFF uses its own baseline.

4. **zscore (motion corrected via subtraction)**  
   `zscore(dFF_signal - dFF_ref)`

5. **zscore (subtractions)**  
   `zscore(dFF_signal) - zscore(dFF_ref)`

6. **dFF (motion corrected with fitted ref)**  
   Fit the isosbestic/reference channel to the signal:  
   `fitted_ref = a * ref_filtered + b`  
   then compute:  
   `dFF = (signal_filtered - fitted_ref) / fitted_ref`

7. **zscore (motion corrected with fitted ref)**  
   `zscore( (signal_filtered - fitted_ref) / fitted_ref )`

### Reference Fitting Methods (for “fitted ref” modes)
- **OLS (recommended)**: fast and stable
- **Lasso**: sparse regression (requires `scikit-learn`)
- **RLM (HuberT)**: robust linear model via IRLS + Huber weighting (no extra dependency)

### Export
- Export processed output to:
  - CSV (`time`, `output`, optional `dio`)
  - HDF5 with raw, baseline, and metadata fields
- Drag-and-drop support for preprocessing and post-processing files.

---

## Repository Structure (typical)

- `analysis_core.py`  
  Processing pipeline (loading, filtering, baselines, outputs, export helpers)
- `main.py` (or similar)  
  PySide6 GUI entry point and UI wiring
- `requirements.yml`  
  Conda environment definition

---

## Installation

1. Create the environment:
   ```bash
   conda env create -f requirements.yml
## Run
  python main.py

## Usage workflow

Open a Doric .h5 file

Choose a channel (e.g., AIN01).

Optionally select a DigitalIO line to overlay events.

### QC & artifact removal

Choose Global MAD (dx) or Adaptive MAD (windowed).

Tune mad_k, window size, and padding.

Add manual mask regions if needed.

### Filtering & resampling

Set low-pass cutoff (Hz) and filter order.

Set a target sampling rate (Hz) for consistent downstream analysis.

### Baseline estimation

Choose asls, arpls, or airpls.

Tune lambda and other parameters to avoid baseline leakage into fast transients.

### Select output

Pick one of the 7 output modes.

For “fitted ref” modes, choose the fit method (OLS/Lasso/RLM-HuberT).


### Export

Export CSV/H5 for analysis in Python/MATLAB/R.

---

## Preprocessing: Advanced Options

The preprocessing panel includes an **Advanced options** button with two features:

1) **Cut out regions (NaN)**  
Define start/end ranges to exclude parts of the trace from downstream analysis. Cutout regions are filled with NaN in the output and can be exported as-is.

2) **Sections (per-section processing)**  
Define multiple start/end sections and **assign per-section processing parameters**. Each section can be exported independently (one CSV/H5 per section).

### Time window
You can optionally set a start time, end time, or both:
- Start only: process from start → end of recording
- End only: process from 0 → end
- Start + end: process that window only
- Empty: process full trace

---

## Post-Processing

### Align sources
- **DIO**: choose DIO channel, polarity (0→1 or 1→0), and align to onset/offset
- **Behavior (CSV/XLSX)**: load a behavior file and select a column (binary 0/1)
  - Align to onset or offset
  - **Transitions**: align to A→B transitions with a max gap threshold

### Behavior file formats
- **CSV**: must include a time column and one or more binary behavior columns
- **Ethovision XLSX**: the loader preprocess and clean the file, the user has to select the sheet when loading

### PSTH/Heatmap
- Heatmap and PSTH refresh automatically when alignment settings change
- Event lines are overlaid on the trace preview
- Event duration histogram appears to the right of the heatmap
- Metrics bar plot (pre vs post) appears to the right of the PSTH

### Metrics
Choose AUC or mean z-score and define pre/post windows in seconds.

### Export results
Export any combination of:
- Heatmap matrix
- Average PSTH + SEM
- Event times
- Event durations
- Metrics table

---

## Group Mode (Multiple Animals)

Use the **Group** tab in Post-Processing to load multiple processed files (CSV/H5). Each file should represent a single animal, and matching behavior files should share the same base name. In group mode:
- Heatmap rows represent animals (not trials)
- PSTH is averaged across animals

---

## Build (PyInstaller)

Example command (run from repo root):

```bash
python -m PyInstaller --noconfirm --clean pyBer.spec
```

PowerShell helper:

```powershell
.\build_pyinstaller.ps1
```

This will create a `pyBer.exe` in the `dist/` folder.



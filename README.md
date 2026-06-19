<p align="center">
  <img src="assets/pyBer_logo_big.png" alt="pyBer logo" width="190">
</p>

<h1 align="center">pyBer</h1>

<p align="center">
  <b>Fiber photometry, from raw photons to publishable figures, without leaving the app.</b><br>
  Load it. Clean it. Align it. Model it. Export it. All in one polished desktop GUI.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/GUI-PySide6%20%2B%20pyqtgraph-41cd52?logo=qt&logoColor=white" alt="PySide6 + pyqtgraph">
  <img src="https://img.shields.io/badge/platform-Windows-0078D6?logo=windows&logoColor=white" alt="Windows">
  <img src="https://img.shields.io/badge/license-GPLv3-blue" alt="GPLv3">
  <img src="https://img.shields.io/badge/photometry-Doric%20%7C%20HDF5%20%7C%20CSV-purple" alt="Formats">
</p>

---

## Why pyBer?

Photometry analysis usually means duct-taping five scripts together and praying your
timestamps line up. pyBer puts the whole pipeline behind one interactive window:
artifact cleaning, motion correction, behavior and DIO alignment, sub-millisecond
camera-to-fiber synchronization, PSTHs, transient detection, and temporal modeling.
Interactive workflow on top, deterministic and testable processing code underneath.

> Built for neuroscientists who want to *see* every step, not just trust a black box.

---

## See it in action

### Preprocess raw traces into clean dFF
Raw signal, fitted-isosbestic baseline, and motion-corrected dFF, side by side, with
DIO markers laid right over the trace. Filtering, baseline, artifact handling, and
export all live in the rail on the left.

![Preprocessing panel](assets/screenshots/preprocessing.png)

### Sync the camera to the fiber with an LED barcode
Point an ROI at the sync LED, extract the on/off train, and auto-align it to the
photometry DIO column. pyBer reads the barcode flashes with a duty-cycle-aware
threshold (Otsu when balanced, Triangle when sparse) so even faint LEDs line up.

![LED barcode sync extraction](assets/screenshots/sync_extraction.png)

### Cross-correlate and verify the alignment
Reference and photometry sync signals stacked on one timeline, with matched edges,
correlation, and agreement metrics so you know the alignment is real before you trust it.

![Photometry sync overlay](assets/screenshots/sync_photometry.png)

---

## Quick install

Install [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda first, then:

```powershell
cd path\to\pyBer
conda env create -f environment.yml
conda activate pyBer
Rscript -e "install.packages('fastFMM', repos='https://cloud.r-project.org')"
python .\pyBer\main.py
```

The `fastFMM` step is only needed for the FLMM temporal modeling panel. Everything
else works without it.

## Launch from VS Code

1. Open the repository folder in VS Code.
2. Select the interpreter from the `pyBer` conda environment.
3. Open `pyBer/main.py`.
4. Press Run, or:

```powershell
conda activate pyBer
python .\pyBer\main.py
```

If VS Code grabs the wrong Python, run `Python: Select Interpreter` and pick the
environment created from `environment.yml`.

---

## What you can do

- 🧹 **Preprocess** raw traces: filtering, resampling, baseline correction, motion
  correction, and artifact handling.
- 🔍 **Hunt artifacts** with interpolation, cutout, local low-pass filtering, or no-op.
- 🎥 **Synchronize** photometry time to camera or behavior time from shared TTL/barcode
  columns, and export a `time_aligned` column for downstream work.
- 📊 **Align to behavior** (DIO, behavior states, onsets, or transitions) and build
  individual or group PSTHs, heatmaps, and event-duration plots.
- ⚡ **Detect transients** and compare amplitudes with baseline-prominence normalized
  metrics.
- 🧠 **Model** with a continuous GLM or trial-level FLMM, then rank feature contribution
  with leave-one-feature-out summaries.
- 💾 **Export** processed CSV or HDF5 with selectable fields and metadata, ready for
  Python, MATLAB, R, or Prism.

---

## Documentation

The full user guide lives in [docs/index.md](docs/index.md): installation, first launch,
preprocessing, postprocessing, transient detection, temporal modeling, group workflows,
export, and troubleshooting.

## Repository layout

| Path | What it is |
|------|------------|
| `pyBer/main.py` | Application entry point and preprocessing window. |
| `pyBer/analysis_core.py` | Preprocessing and signal-processing backend. |
| `pyBer/gui_preprocessing.py` | Preprocessing panels. |
| `pyBer/gui_postprocessing.py` | Postprocessing, PSTH, sync, metrics, and export. |
| `pyBer/led_extract.py` | LED / barcode sync-signal extraction. |
| `pyBer/time_sync.py` | Edge detection, pairing, and cross-correlation alignment. |
| `pyBer/temporal_modeling.py` | GLM and FLMM modeling panel. |
| `environment.yml` | Conda environment for development and user installs. |
| `pyBer.spec` | PyInstaller build configuration. |

## Build the executable

From an activated environment:

```powershell
conda activate pyBer
python -m PyInstaller --noconfirm --clean pyBer.spec
```

The executable is written to `dist/pyBer.exe`.

## Notes

pyBer sets `PYTHONNOUSERSITE=1` so stale packages from the user Python folder cannot
shadow the conda environment. This keeps Qt, pyqtgraph, numpy, and rpy2 stable on Windows.

---

<p align="center"><sub>Made with 🧠 and a lot of dFF at the Bellone Lab.</sub></p>

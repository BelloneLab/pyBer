# pyBer

pyBer is a desktop application for fiber photometry analysis. It helps you load
Doric, HDF5, or CSV recordings, clean artifacts, preprocess traces, align signals
to behavior or DIO events, inspect PSTHs and heatmaps, detect transients, and
export results for Python, MATLAB, R, or Prism.

The app is built for users who want an interactive workflow first, with
deterministic processing code underneath.

![pyBer logo](https://github.com/user-attachments/assets/e5acb000-17cd-451d-9f49-4218b41519aa)

## Quick Install

Install Miniforge or Anaconda first, then run:

```powershell
cd C:\Analysis\app_project\pyBer
conda env create -f environment.yml
conda activate pyBer
Rscript -e "install.packages('fastFMM', repos='https://cloud.r-project.org')"
python .\pyBer\main.py
```

The `fastFMM` step is only needed for the FLMM temporal modeling panel. The rest
of pyBer works without it.

## Launch From VS Code

1. Open the repository folder in VS Code.
2. Select the interpreter from the `pyBer` conda environment.
3. Open `pyBer/main.py`.
4. Press Run, or use:

```powershell
conda activate pyBer
python .\pyBer\main.py
```

If VS Code launches the wrong Python, run `Python: Select Interpreter` and choose
the environment created from `environment.yml`.

## What You Can Do

- Preprocess raw photometry traces with filtering, resampling, baseline
  correction, motion correction, and artifact handling.
- Detect and inspect artifacts with interpolation, cutout, local low-pass
  filtering, or no-op handling.
- Export processed CSV or HDF5 files with selectable fields and metadata.
- Align processed signals to DIO, behavior states, behavior onsets, or behavior
  transitions.
- Detect signal events and compare transient amplitude with baseline-prominence
  normalized metrics.
- Build individual or group PSTHs, heatmaps, event duration plots, and metrics.
- Fit temporal models with continuous GLM or trial-level FLMM.
- Rank GLM/FLMM feature contribution with leave-one-feature-out summaries.

## Documentation

The full user guide is here:

- [pyBer Documentation](docs/index.md)

It includes installation, first launch, preprocessing, postprocessing, transient
detection, temporal modeling, group workflows, export, and troubleshooting.

## Repository Layout

- `pyBer/main.py`: application entry point.
- `pyBer/analysis_core.py`: preprocessing and signal processing backend.
- `pyBer/gui_preprocessing.py`: preprocessing panels.
- `pyBer/gui_postprocessing.py`: postprocessing, PSTH, metrics, and export panels.
- `pyBer/temporal_modeling.py`: GLM and FLMM modeling panel.
- `environment.yml`: conda environment for development and user installs.
- `pyBer.spec`: PyInstaller build configuration.

## Build The Executable

From an activated environment:

```powershell
conda activate pyBer
python -m PyInstaller --noconfirm --clean pyBer.spec
```

The executable is written to `dist/pyBer.exe`.

## Notes

pyBer sets `PYTHONNOUSERSITE=1` in the environment so old packages from the user
Python folder do not interfere with the conda environment. This is important for
Qt, pyqtgraph, numpy, and rpy2 stability on Windows.

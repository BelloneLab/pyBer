# pyBer Documentation

This page is a practical guide for installing and using pyBer. It is written for
lab users who want to process recordings without reading the code first.

## 1. Install pyBer

### Recommended Windows install

Install Miniforge or Anaconda, open an Anaconda/Miniforge Prompt, then run:

```powershell
cd C:\Analysis\app_project\pyBer
conda env create -f environment.yml
conda activate pyBer
Rscript -e "install.packages('fastFMM', repos='https://cloud.r-project.org')"
python .\pyBer\main.py
```

The `fastFMM` command installs the R package used by the FLMM temporal modeling
panel. It can take a few minutes the first time because R downloads dependencies.

### Update an existing environment

If the environment already exists:

```powershell
conda activate pyBer
conda env update -f environment.yml --prune
Rscript -e "install.packages('fastFMM', repos='https://cloud.r-project.org')"
```

### Test the install

Run:

```powershell
conda activate pyBer
python .\pyBer\main.py
```

The app should open with Preprocessing and Postprocessing tabs.

## 2. Launch From VS Code

1. Open the pyBer repository folder.
2. Press `Ctrl+Shift+P`.
3. Choose `Python: Select Interpreter`.
4. Select the `pyBer` conda environment.
5. Open `pyBer/main.py`.
6. Press Run.

If the Run button still fails, use the terminal:

```powershell
conda activate pyBer
python .\pyBer\main.py
```

## 3. Preprocessing Workflow

Use Preprocessing when you want to clean and export photometry traces.

1. Load one or more raw files.
2. Select the signal channel and optional reference channel.
3. Choose artifact detection and handling.
4. Set filtering and resampling.
5. Choose baseline correction.
6. Choose the output signal definition.
7. Preview the result.
8. Export CSV or HDF5.

### Output definitions

pyBer exposes explicit output modes so exported traces are reproducible:

- dFF without motion correction.
- z-score without motion correction.
- dFF with motion correction by subtraction.
- z-score with motion correction by subtraction.
- z-score signal minus z-score reference.
- dFF with fitted reference.
- z-score with fitted reference.

For fitted-reference modes, pyBer fits the reference channel to the signal before
computing dFF. The usual choice is OLS. Lasso and robust Huber fitting are also
available.

## 4. Artifact Handling

Artifact settings let you choose how masked windows are handled:

- Interpolation: replace artifact samples by linear interpolation.
- Cut: keep artifact samples as NaN so downstream analysis ignores them.
- Strong local low-pass filtering: smooth only inside the artifact window.
- Do nothing: detect or mark artifacts without changing the trace.

Use interpolation when you need continuous traces. Use cut when the artifact
window should not contribute to statistics.

## 5. Postprocessing Workflow

Use Postprocessing when you want to align processed traces to events or behavior.

1. Load processed files from preprocessing.
2. Load behavior files if needed.
3. Choose the alignment source.
4. Click `Compute PSTH`.
5. Inspect the trace preview, heatmap, average PSTH, duration plot, and metrics.
6. Export matrices, event times, metrics, and figures.

### Alignment sources

pyBer can align to:

- DIO onset or offset.
- Behavior onset or offset from CSV or XLSX files.
- Binary behavior state columns.
- Behavior transitions.
- Signal events detected from the processed trace.

### Group mode

Use Group mode when each processed file represents one animal. pyBer keeps
per-file trial matrices for temporal modeling and can also display animal-level
group summaries.

For best GLM and FLMM results, load matching behavior files whose base names
match the processed files.

## 6. Signal Event Analyzer

The Signal Event Analyzer detects transients and reports metrics. Useful options:

- Auto MAD noise thresholding for transient detection.
- Min prominence, min height, min distance, and smoothing.
- Optional detected-peak overlay on the trace.
- Optional noise trace overlay.
- Baseline-prominence normalized amplitude for comparing recordings.

Baseline-prominence normalized amplitude is useful when recordings differ in
baseline level or noise scale. It normalizes each detected transient relative to
its local baseline/prominence context.

## 7. Temporal Modeling

The Temporal Modeling panel supports two approaches.

### Continuous GLM

Use GLM when you want to model the continuous photometry trace from event and
behavior predictors.

Typical predictors:

- DIO events.
- Behavior onsets.
- Behavior states.
- Numeric behavior columns.
- Signal event times.

The GLM output includes:

- R-squared.
- RMSE, MAE, MSE, residual SD, and actual/predicted correlation.
- Estimated kernels for each predictor.
- Actual vs predicted signal.
- Residual trace.
- Leave-one-predictor-out feature contribution.

The leave-one-predictor-out ranking refits the model after removing each
predictor. Larger `delta R^2` means that predictor explains more of the signal.

### Trial-level FLMM

Use FLMM when you want trial-level functional modeling with random effects.

Requirements:

- R installed through the conda environment or available on the system.
- Python package `rpy2`.
- R package `fastFMM`.
- Repeated rows per subject or animal.

The environment installs R and rpy2. Install fastFMM with:

```powershell
conda activate pyBer
Rscript -e "install.packages('fastFMM', repos='https://cloud.r-project.org')"
```

The FLMM output includes:

- Fixed-effect coefficient curves.
- Pointwise and joint confidence bands when available.
- AIC summary.
- Coefficient magnitude statistics.
- Leave-one-feature-out AIC contribution when the reduced models are estimable.

If a reduced FLMM cannot be estimated, pyBer still reports the coefficient-based
contribution so the feature ranking remains usable.

## 8. Export

Preprocessing can export processed traces as:

- CSV.
- HDF5.

Postprocessing can export:

- Heatmap matrix.
- Average PSTH and SEM.
- Event times.
- Event durations.
- Metrics tables.
- Group-level outputs.

Use HDF5 when you want metadata and multiple arrays in one file. Use CSV when you
want easy loading into spreadsheets or Prism.

## 9. Troubleshooting

### The app does not launch from VS Code

Make sure VS Code is using the conda environment:

```powershell
conda activate pyBer
python .\pyBer\main.py
```

If this works but the Run button fails, select the interpreter again in VS Code.

### Dark mode or Qt styling looks broken

This is usually a mixed Python environment. Recreate the environment and keep
`PYTHONNOUSERSITE=1` enabled:

```powershell
conda env remove -n pyBer
conda env create -f environment.yml
conda activate pyBer
```

### FLMM says fastFMM is unavailable

Run:

```powershell
conda activate pyBer
Rscript -e "install.packages('fastFMM', repos='https://cloud.r-project.org')"
```

Then restart pyBer.

### FLMM says random effects cannot be estimated

FLMM needs repeated rows per subject. In practice, each animal should have
multiple trials. If you only provide one animal-averaged row per animal, the GLM
panel is usually the better choice.

### The heatmap looks wrong after switching Individual and Group

Click `Compute PSTH` again after changing loaded files or behavior alignment.
pyBer stores both per-file trial matrices and group matrices, but recomputing is
the clearest way to refresh all derived views after a major setup change.

## 10. Build A Windows Executable

From the repository root:

```powershell
conda activate pyBer
python -m PyInstaller --noconfirm --clean pyBer.spec
```

The app is written to:

```text
dist\pyBer.exe
```

When building with FLMM support, make sure `fastFMM` is installed before running
PyInstaller.

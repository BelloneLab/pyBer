# pyBer Documentation

This page is a practical guide for installing and using pyBer. It is written for
lab users who want to process recordings without reading the code first.

## Headless CLI batch workflow

Version 0.45 adds `pyBer/cli.py` and a dedicated Windows executable named
`pyBer-cli-windows.exe`. Pass one or more files or folders. Folder traversal is
recursive unless `--no-recursive` is supplied.

```powershell
pyBer-cli-windows.exe "D:\data\experiment" --sensor jgcamp8f --channel AIN01
```

Useful options:

- `--sensor ID_OR_NAME`: selects the sensor used for kinetics-aware filtering,
  sampling, polarity checks, and export metadata.
- `--channel NAME`: processes only that raw channel. Repeat the option or use a
  comma-separated list. If omitted, every detected photometry channel is used.
- `--trigger NAME`: includes a matching DIO or analog output trigger.
- `--set NAME=VALUE`: overrides a recommended `ProcessingParams` field. Repeat
  it for any combination of artifact, filtering, resampling, baseline,
  reference-fit, polarity, smoothing, or output settings.
- `--params-file PATH`: loads the same overrides from a reusable JSON object.
  Repeatable `--set` values take precedence over the file.
- `--format both|csv|h5`: chooses processed data formats. The default is both.
- `--output-dir PATH`: places outputs under a chosen directory while retaining
  relative subfolder structure.

The automatic recommendation examines acquisition timing and trace statistics,
then combines those measurements with the selected sensor's expected kinetics.
Explicit `--set` values always take precedence. Each result includes a PNG and
JSON preprocessing report. Batch-level CSV and JSON summaries make processing
auditable, and `flagged_recordings.csv` is the review queue for failed, highly
artifactual, flat, incomplete, low-confidence, or sensor-inconsistent recordings.

Legacy Doric files using keys such as
`AIN01xAOUT02-LockIn/Values` are supported alongside current files using
`LockInAOUT02/AIN01`.

## 1. Install pyBer

### Recommended Windows install

Install Miniforge or Anaconda, open an Anaconda/Miniforge Prompt, then run:

```powershell
cd C:\Analysis\app_project\pyBer
powershell -ExecutionPolicy Bypass -File .\scripts\create_pyber_env.ps1
conda activate pyBer
python .\pyBer\main.py
```

The helper creates or updates the `pyBer` environment, installs R, and installs
the CRAN `fastFMM` binary used by the FLMM temporal modeling panel. It can take a
few minutes the first time because R downloads dependencies.

### Update an existing environment

If the environment already exists:

```powershell
conda activate pyBer
conda env update -f environment.yml --prune
Rscript scripts/install_fastfmm.R
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

### Recommended settings

When a raw preprocessing file is selected, pyBer analyzes the active channel and
time window, then shows **Recommended for your data** above the settings cards.
The recommendation estimates sampling rate, drift, artifact burden, 405/470
coupling, rolling short-timescale correlation, and full acquisition polarity. It
does not overwrite your settings automatically. Press **Apply recommended
settings** to accept the proposed parameters.

Each preprocessing card also receives a short explanation:

- Artifacts: detected outlier burden and why global or adaptive MAD is proposed.
- Filtering: raw sampling rate, target sampling rate, low-pass cutoff, smoothing,
  and whether full 465/405 polarity inversion appears necessary.
- Baseline: duration, drift score, baseline method, and lambda.
- Output: why signal-only dFF, fitted-reference dFF, inverted isobestic fit, or
  band-limited inverted correction was selected.

### Sensor library

Use **Sensor** beside **Plot style** to select the expressed indicator before
final preprocessing. The searchable sensor table includes calcium indicators
(GCaMP3, GCaMP6, jGCaMP7, jGCaMP8, jRGECO), dopamine sensors (dLight and
GRAB-DA variants), serotonin sensors (GRAB-5HT and sDarken), norepinephrine,
acetylcholine, glutamate, GABA, endocannabinoid, orexin, opioid, adenosine,
ATP, histamine, oxytocin, and GRAB neuropeptide sensors.

Each row records the sensor family, target, color, expected fluorescence
direction, excitation, isobestic or control wavelength, emission, rise and decay
values, a kinetics basis note, affinity, dynamic range, recommended sampling
rate, recommended low-pass cutoff, source, and a paper link. Click **Open paper**
from the dialog to open the source in your browser.

After a sensor is selected:

- The top raw plot title changes from `raw signal` to the selected sensor name.
- The recommendation engine caps target sampling rate and low-pass cutoff using
  the selected sensor kinetics.
- The auto-polarity check compares the raw trace direction with the expected
  sensor response direction, including darkening sensors such as sDarken.
- Export sidecars and embedded HDF5 metadata include the selected sensor, source
  link, optical wavelengths, kinetics, and trace-check result.

Brutal practical point: many sensor papers report kinetics under different
conditions (cell culture, slice, one-photon imaging, two-photon imaging, or
in vivo photometry). pyBer therefore uses conservative photometry-oriented
recommendations and writes qualitative fields when a single universal number
would be dishonest.

See [Fiber Photometry Sensor Literature Review](sensor_literature_review.md)
for the source-level review behind the registry.

### Output definitions

pyBer exposes explicit output modes so exported traces are reproducible. Each
mode maps to a stable **family column name** used in the exported files, plus a
short **variant** tag that distinguishes same-family modes. The exact mode you
picked is always recorded in the sidecar metadata (see [Export](#8-export)).

| Output mode | Column / dataset | Variant |
|-------------|------------------|---------|
| dFF without motion correction | `dFF` | `nomc` |
| z-score without motion correction | `z-score` | `nomc` |
| dFF with motion correction by subtraction | `dFF` | `sub` |
| z-score with motion correction by subtraction | `z-score` | `sub` |
| z-score signal minus z-score reference | `z-score` | `zdiff` |
| dFF with fitted reference | `dFF` | `fitref` |
| dFF with inverted isobestic fit | `dFF` | `invfitref` |
| dFF with band-limited inverted isobestic | `dFF` | `bandinvfitref` |
| z-score with fitted reference | `z-score` | `fitref` |
| prominence-normalized (fitted reference) | `prominence` | `fitref` |
| raw processed 465 signal | `signal_465` | `raw` |

The output you preview in **Output mode** is always the one exported (the
"primary" output). You can tick additional outputs under **Also export** to write
several at once; the primary keeps the bare family name (for example `dFF`) and
each extra same-family output gets a `family__variant` name (for example
`dFF__nomc`). There is never a generic `output` column.

For fitted-reference modes, pyBer fits the reference channel to the signal before
computing dFF. The fit slope is constrained to be nonnegative in the selected
polarity, so a normal 405 fit cannot silently become an inverted 405 fit by
learning a negative coefficient. The usual choice is OLS. Lasso and robust Huber
fitting are also available.

If the isobestic trace is inverted relative to the calcium signal, choose
**dFF (motion corrected with inverted isobestic fit)**. pyBer fits `-ref_f` onto
`sig_f`, then computes `(sig_f - fitted_ref) / fitted_ref` and records the
variant as `invfitref` in export metadata. If the un-inverted reference is
anti-correlated, the normal fitted-reference mode will not use a negative slope;
the inverted mode is the efficient correction. The preprocessing output context
reports the fitted slope and intercept so this is visible.

If slow 405/470 bleaching drift is positively shared but fast 405/470
fluctuations are anti-correlated, choose **dFF (motion corrected with
band-limited inverted isobestic)**. This mode subtracts a rolling median from
both dFF traces, fits only the short-timescale `-dFF_ref` component with
`beta >= 0`, and removes that component from `dFF_sig`. It is intended for mixed
polarity recordings where a single whole-session raw fit is dominated by slow
drift.

## 4. Artifact Handling

Artifact settings let you choose how masked windows are handled:

- Smart multi-evidence: combines local robust residuals, slope shocks,
  curvature, dropouts, level shifts, and shared 405/465 evidence. This is the
  recommended default for raw preprocessing.
- Adaptive MAD: uses a sliding local median/MAD envelope.
- Global MAD: uses one full-trace median/MAD envelope.
- Interpolation: replace artifact samples by linear interpolation.
- Cut: keep artifact samples as NaN so downstream analysis ignores them.
- Strong local low-pass filtering: smooth only inside the artifact window.
- Do nothing: detect or mark artifacts without changing the trace.

Use interpolation when you need continuous traces. Use cut when the artifact
window should not contribute to statistics.

Smart detection is intentionally conservative around possible neural events:
smooth positive transients that appear only in the 465 channel are not masked
unless they are abrupt, extreme, or supported by 405/reference evidence. The
Artifacts panel lists the evidence for each region, such as `465:amp+slope`,
`405:drop`, or `shared`, plus a quality score `Q`.

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

### Time synchronization

Use the `Sync` postprocessing panel when camera/behavior time and photometry
time are not already in the same clock.

1. Load processed photometry files.
2. Load behavior or EthoVision CSV/XLSX files that contain camera time and a
   sync column, such as a 1 Hz TTL or barcode/value column.
3. Open `Sync`.
4. Choose the behavior/camera file, or keep `Auto-match behavior file` for
   batch queues where each behavior file corresponds to one recording.
5. Choose the camera sync behavior/column and its extraction mode.
6. Choose the photometry sync source: embedded DIO from the processed file, or a
   raw A/D channel from the Doric file.
7. Choose `Linear regression` for one global clock-drift estimate, or
   `Interpolation` when you want pulse-to-pulse timing correction.
8. Click `Preview selected file` and inspect the event mapping and residual lag.
9. Click `Apply selected` or `Apply batch`.

The panel reports matched pulse count, median lag, residual RMS, maximum
residual, and clock drift in ppm. When `Use aligned time for postprocessing` is
enabled, PSTH, signal-event analysis, and spatial activity interpolation use the
new aligned timebase.

`Export aligned files` writes CSV/HDF5 files with both:

- `time`: original photometry time.
- `time_aligned`: camera/behavior-aligned time.

This makes the synchronization explicit and keeps the original trace unchanged.

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
Rscript scripts/install_fastfmm.R
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

### Processed-trace format (pyBer v1.0)

Preprocessing exports each processed recording as a matched pair plus a metadata
sidecar, all sharing one file stem:

```
M12_day1_3_AIN01.csv          # data table (clean, header on the first row)
M12_day1_3_AIN01.h5           # same data as HDF5 datasets (self-contained)
M12_day1_3_AIN01.pyber.json   # metadata sidecar (shared by the CSV and H5)
```

**Design principles**

- **Simple, stable headers.** Columns are named for downstream scripts, Prism,
  MATLAB, and R. There are no comment lines in the CSV; the first row is the
  header. The processed output uses its family name (`dFF`, `z-score`,
  `prominence`, or `signal_465`), never a generic `output`.
- **What you select is what you get.** The primary output (the one shown in
  *Output mode*) is always written and is flagged `primary` in the sidecar.
  Extra outputs are additive and are written under `family__variant` names.
- **Metadata lives in the sidecar**, not in the data file. HDF5 also embeds the
  same JSON (attribute `pyber_meta_json`) so a single `.h5` is self-contained.

**Columns / datasets** (present only when selected):

| Name | Meaning |
|------|---------|
| `time` | Recording time in seconds (photometry clock). Always present. |
| `time_aligned` | Camera/behavior-aligned time in seconds (only after Sync). |
| `raw_465` | Raw processed 465 signal. |
| `raw_405` | Isosbestic (405) reference. |
| `dFF` / `z-score` / `prominence` / `signal_465` | The primary processed output. |
| `dFF__nomc`, `z-score__sub`, ... | Any additional selected outputs. |
| `baseline_465`, `baseline_405` | Estimated baselines. |
| `DIO01`, `DIO02`, ... | Digital trigger channels, under their real names. |

In HDF5 these are datasets under the `/data` group; `/data` carries attributes
(`primary_output`, `output_label`, `fs_used`, `dio_name`, ...) and each output
dataset carries `label`, `family`, `variant`, and `units`.

**Sidecar (`<stem>.pyber.json`)** records everything needed to interpret the
file: `primary_output`, an `outputs` map (label, family, variant, units,
reference fit, motion-correction method), a `columns` role map, the `subject`
metadata, the full `processing` parameters, sampling rates, and the `sync`
report. Load the sidecar (or the embedded HDF5 JSON) to know exactly which
output definition a `dFF` column represents.

> Backward compatibility: pyBer still reads older exports that carried `#`
> comment metadata lines and a generic `output` column/dataset.

Postprocessing can export:

- Heatmap matrix.
- Average PSTH and SEM.
- Event times.
- Event durations.
- Metrics tables.
- Group-level outputs.

The **Export aligned files** action in postprocessing writes the same v1.0
processed-trace format (with a `time_aligned` column) so aligned files load back
exactly like preprocessing exports.

Use HDF5 when you want a single self-contained file with all arrays and metadata.
Use CSV plus its sidecar when you want easy loading into spreadsheets or Prism.

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
Rscript scripts/install_fastfmm.R
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

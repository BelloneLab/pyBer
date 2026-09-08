# Behavior and trajectory imports

In Postprocessing, choose **Setup > Behavior / Events > Load behavior CSV/XLSX**.
For Pykaboo metadata, use **Binary states (time + 0/1 columns)** and load the
original `_metadata.csv` file. No column renaming is required.

## Automatic selections

- **Time column: Auto** recognizes `time`, Ethovision trial/recording time,
  `timestamp_software`, and `timestamp_camera`. Software time takes precedence
  over camera time when neither a time nor trial/recording time column exists.
- The **Time column** selector can switch between the recognized clocks without
  rereading the file. Original timestamp offsets are preserved. Selecting a
  clock absent from one file falls back to that file's automatic clock.
- Pykaboo binary behavior columns and event flags become available for behavior
  analysis and PSTH alignment. Acquisition settings, frame IDs, confidence
  values and other metadata are excluded from automatic behavior selection.
- Spatial maps select a matching X/Y pair, preferring `mouse_1_center_x` and
  `mouse_1_center_y`. Other tracked animals and keypoints remain selectable.
  Missing detections are excluded using their sentinel/validity fields.
- `_metadata` filename suffixes are recognized when matching recordings to
  behavior files, before the existing positional fallback. Inspect file
  associations when filenames do not correspond.

Hover over **Loaded files** to see each file's selected clock and inferred
behavior/coordinate counts. Clock candidates and import details are retained
in saved postprocessing projects. Changing clocks updates derived threshold
events as well as binary behavior timing.

Automatic clock selection does not synchronize independent acquisition systems.
Use the Sync tools when photometry and behavior clocks require an offset or
drift correction.

## Other tables

Comma-, semicolon- and tab-delimited CSV files and the existing Ethovision XLSX
loader use the same column inference. Common X/Y names, numeric 0/1 behavior
states, text boolean states and elapsed `hh:mm:ss` or `mm:ss` clocks are supported.
Generic continuous numeric channels remain available for threshold alignment.
Files without a recognized measured clock retain the existing FPS-based time
option. Event-list tables should use **Timestamps** mode instead of binary mode.

## Figure export

Image and PDF exports render the entire plot widget. Export Results selects a
destination folder and constructs bundle names from the active recording and
alignment. In Individual view, select the intended recording before exporting.

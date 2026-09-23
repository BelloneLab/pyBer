# Behavior and trajectory imports

In Postprocessing, choose **Setup > Behavior / Events > Load behavior CSV/XLSX**.
This familiar action now detects MAMIR behavior/zone CSVs and EthoVision
workbooks within the existing postprocessing dashboard. For Pykaboo
metadata, use **Binary states (time + 0/1 columns)** and load the original
`_metadata.csv` file. No column renaming is required. **Timestamps per behavior**
keeps the established timestamp-column importer.

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

## Zone and behavior analysis

In Postprocessing, load the processed fiber recording and use the existing
Setup **Load behavior CSV/XLSX** action. The toolbar's **Zone / Behavior**
selector chooses the analysis. **Zone** retains the established trace,
PSTH heatmap, alignment controls and pre/post metrics. **Behavior** adds a
before/during/after comparison for scored social or individual behaviors above
those same plots. Click anywhere on a behavior row to add or remove it;
the plots and table update automatically. The search box filters behavior names.
Time-window, recording and Individual / Group changes also update automatically.
Rapid clicks are combined into a single update without rereading source files.
Behavior choices are large, whole-row check targets with keyboard Space support.
**Select visible** adds the behaviors matching the search without clearing hidden
selections; **Clear selection** removes every comparison selection.

Drag the divider between the comparison and PSTH to change their heights; this
desktop size preference is remembered. The selector/results divider and the
comparison plot/table divider can also be dragged. **Hide comparison** gives the
existing plots more space without clearing the selections. The comparison
scrolls when space is limited, so its controls remain reachable.

In Behavior mode, a dedicated **PSTH / HEATMAP** bar above the original plots
shows the single plotted behavior, alignment, Pre/Post windows, row meaning,
and color units. These controls mirror the existing PSTH drawer, not a second
set of settings. **Show heatmap** and **Show mean PSTH** navigate to those plots;
the filters/normalization button opens the original PSTH drawer. Behavior plot
titles name the selected behavior, and the heatmap has its own time ticks.
Switching to Zone removes these additions and restores the original plot labels.
It uses the existing **Individual / Group** and recording selectors.
As in main pyBer, **Individual** shows the selected animal/file's events;
**Group** first averages events within each animal/file, then combines those
animal means. The existing **keep trials** option retains individual event rows
in the group PSTH instead. Minimum-event exclusions remain part of the existing
PSTH workflow, and an excluded animal can still be inspected in Individual view.
Imported behaviors are also available in **PSTH > Alignment > Behavior name**,
using the same onset, offset, transition, window and event-filter controls as
zone events. Choosing a single behavior there selects it for comparison;
an existing multi-behavior comparison selection is retained and the chosen
behavior is added to it. The most recently checked behavior also becomes the
PSTH selection.
pyBer detects MAMIR bout/frame/zone CSV exports and
EthoVision XLSX workbooks automatically. Generic binary CSV and spreadsheet
tables remain supported. The Setup load button and file drop use the same
automatic import path in binary mode; timestamp-column mode remains available.

For MAMIR, select `behavior_bouts.csv`, `behavior_frames.csv`,
`behavior_frames_long.csv`, or
`zone_bouts.csv`. A frame export with another filename is accepted when it has
`frame`, `time_s`, and `identity` columns. For zones, keep `summary.json` beside
`zone_bouts.csv`; pyBer reads its FPS to convert inclusive frame bounds to event
times. MAMIR imports select the corresponding analysis mode automatically.
Aggregate-only `behavior_summary.csv` has no bout times and cannot be aligned
to a fiber trace; use one of the time-resolved exports above.
When a file is selected explicitly, pyBer reads that file, not a neighboring
export. Frame labels are matched by column name regardless of CSV column order.
If a saved project warns about an older MAMIR import, re-import the original
frame file: an early development adapter could associate flags with the wrong
behavior names. Existing saved arrays and raw source files are not rewritten.
For EthoVision workbooks with multiple arena sheets, pyBer lists the detected
arenas and subject metadata in **Setup > Arena / subject** beside the file loader.
Choose the fiber-associated arena in the selector; pyBer loads it immediately,
as in the EthoVision sheet viewer. You can switch it later; pyBer replaces only
that workbook's active arena data and
keeps other imported events. The zone importer reads explicit
0/1 zone columns such as `In zone`. Coordinates alone cannot establish zone
membership without a defined boundary.
EthoVision cleaning retains main's existing numeric interpolation default,
including missing tracking samples; the original workbook is never rewritten.

Choose the animal that carries the fiber. For MAMIR bout files, you may restrict
events to one partner. Explicitly pair each imported file or workbook sheet with
its fiber recording. The source files are read only and imports are embedded
when a pyBer postprocessing project is saved.

In Behavior mode, tick one or more names and compare the processed signal
before, during, and after complete bouts. **Individual** uses the toolbar's
selected recording; **Group** averages recording means so a recording with many bouts does not
outweigh another recording. Only windows fully covered by finite signal are
included. The table reports accepted and rejected event counts, and can be
exported to CSV. Group comparison rejects mixed signal units. The
**Comparison-only offset** adds seconds to event times in this extra comparison;
it does not shift the PSTH/heatmap or estimate synchronization. Verify the fiber/video clock relation
with pyBer's Sync tools or a shared hardware event before interpreting the
comparison. Switching back to Zone hides the additional comparison and keeps
the existing pre/post settings and plots.
The separate before/during/after comparison uses complete bouts of its checked
labels; the PSTH alignment and event filters control the PSTH plots.
The comparison's group plot shows SEM across contributing animal/file means
when at least two contribute. Its N is computed separately for each behavior;
missing events or unusable signal never become zero-valued animal responses.
Explicit source pairings survive recording reorder/removal. Duplicate recording
names are rejected to prevent double counting. As in main's animal-level file
workflow, use one intended animal-level recording per independent animal;
repeated sessions are not automatically combined by mouse ID.

Comparison CSVs include contributing recording IDs, counts, SEM, before/after
windows, offset, units, clock source, and averaging/normalization definitions.
Changing scope or synchronization clears stale comparison exports until the
automatic recomputation finishes. When aligned time is enabled but unavailable
for some recordings, the original-time fallback is reported explicitly.

### What do the time windows and heatmap mean?

For a behavior bout starting at **10 s** and ending at **13 s**, with Before
and After both set to 2 s:

| Comparison window | Recording time |
| --- | --- |
| Before | 8–10 s |
| During | 10–13 s (the whole bout) |
| After | 13–15 s |

The comparison plot shows three means, not three equally long time segments.
Before/after can include other bouts or behaviors; they are not automatically
quiet baseline periods. Endpoints are excluded from the preceding window.

The original PSTH **Pre / Post** settings are separate. With onset alignment,
Pre = 2 s and Post = 5 s show 8–15 s for this example, displayed as −2 to +5 s
relative to the start at zero. With offset alignment, zero is the bout end.
**Post does not mean after the behavior ends.**

Heatmap columns are relative time; individual rows are bouts. Group rows are
animal/recording averages, or individual trials when the existing keep-trials
option is enabled. Colors show the fiber signal using the selected PSTH
normalization and color scale, not behavior probability. Bouts are not stretched
to equal durations. The heatmap shows the single PSTH-selected label, while the
comparison can contain several checked labels. Removing the heatmap's label
switches to another checked label when available; removing all labels clears
the comparison but leaves the independent PSTH selection intact.

MAMIR's assay zone export determines membership from tracked animal centers.
Its head-direction outputs are separate orientation measurements. A body-area
overlap criterion should be exported and labeled separately if needed.

## Adjusting heatmap colors

The **Colors** controls above the PSTH heatmap work in both Zone and Behavior,
for Individual and Group views. The original automatic modes are unchanged:

- Full range uses the finite displayed minimum and maximum.
- Robust contrast uses the 2nd and 98th percentiles to reduce the visual effect
  of extreme values. Values outside these limits saturate in color; no data is removed.
- Symmetric limits put zero in the middle of the color scale.

Drag the compact color-bar handles, or click **Adjust** for spatial-style
histogram cursors, exact **Min / Max** inputs and palette selection. Blue–white–red
with symmetric limits is useful for showing positive and negative signals.
**Auto** releases fixed limits and reapplies the selected automatic mode.

Adjustments are display-only: they do not change event selection, baseline,
normalization, signal values or statistics. Fixed limits persist when switching
behaviors or recordings; changing normalization resets them. Use identical fixed
limits and units when comparing colors across animals or groups: independently
autoscaled plots need not assign the same color to the same value.

Projects preserve the palette and limits. Publication figures use the same
palette and limit policy as the live heatmap. PSTH color changes do not alter
spatial-map colors or limits.

## PSTH baseline interpretation

The existing default uses a separate baseline for each event, from -1 to 0 s
relative to its alignment time. Baseline z-score subtracts the mean and divides
by the standard deviation of finite source signal samples in that window.
At least five finite samples and a nonzero standard deviation are required;
unusable rows are excluded rather than filled with zeros. Endpoints are included,
so an end of 0 s can include the alignment-time sample. Offset alignment places
zero at the bout end, not its start.

Other bouts and behaviors are not automatically removed from this reference.
A quiet heatmap color means signal near its baseline, not absence of behavior.
Choose a window and normalization appropriate to the experiment; the default
does not establish a neutral or behavior-free baseline. Group animal rows average
the event-normalized rows within each recording. The before/during/after
comparison above uses separate windows and the original processed signal units.

## Interactive behavior example

Run `python scripts/preview_behavior_example.py` to open an isolated example
with simulated fiber and behavior data. Add `--mamir /path/behavior_frames.csv`
to use real MAMIR event times with simulated fiber signals. The window labels
the simulation explicitly; it does not represent a measured neural response.
The example includes two recordings for Individual and Group comparisons and
uses temporary preferences without changing source files or normal settings.
Add `--zone-workbook /path/arenas.xlsx` to load arena sheets for Zone testing too.
Do not interpret unrelated behavior and zone files as the same experiment.
Add `--heatmap-colors` to open the heatmap color editor in the example.

## Figure export

Image and PDF exports render the entire plot widget. Export Results selects a
destination folder and constructs bundle names from the active recording and
alignment. In Individual view, select the intended recording before exporting.

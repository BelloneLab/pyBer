# Behavior extension validation — 2026-09-23

Reference: original main commit `5fc55570a5e9375c51edac2f7747945ce49cc785`.
This audit was performed on the v0.53 release candidate before publication.

## Critical correction: early frame-import examples are superseded

The all-label audit found a column-order bug in the early MAMIR frame adapter:
Pandas `usecols` selects columns but does not reorder them. Alphabetically sorted
behavior names were incorrectly associated with states in original CSV order.
The adapter now explicitly reindexes wide and long frame tables before reading
rows. Earlier frame-import demo results must not be interpreted by behavior name.
Re-import original frame files in projects created with that early adapter;
old embedded sources show a warning and are never silently rewritten.

The corrected import was checked against independently extracted runs from the
original downloaded frame CSV: all 24 labels for each of two animals matched,
including every onset and offset. There are 1,588 bouts for animal 1 and 1,305
for animal 2; this particular export has no active fighting or grooming bouts.
An additional real-widget audit passed 144 onset/offset and individual/group
checks across those labels. The corrected demo uses walking and running with
explicitly **simulated** fiber signals, not the unrelated downloaded fiber file.
The frame CSV SHA-256 remained unchanged:
`5a24ccf120b87f5c8486f3ecf48b17d3d8f43de99e3561c2d9b22f1b09cdae94`.

Regression tests also compare all labels across wide, long, and bout exports
for both identities; cover shuffled metadata/behavior columns, duplicate long
rows, nonzero clock origins, invalid clocks, and explicit file selection when
neighboring exports exist. Explicitly selected frame files are no longer
silently replaced by neighboring bout files. Comparison selections/windows,
including intentional empty selections, survive project saves; Recent Files
uses the same auto-detection path as the main loader.

## Legacy zone compatibility

The original postprocessing module was loaded directly from the reference Git
object and compared with the candidate in isolated Qt preferences. Exact array
equality passed in 54 combinations of:

- Onset, offset, and A-to-B transition alignment.
- Original units, baseline subtraction, and baseline z-score.
- Individual, group animal-average, and group trial displays.
- Optional filters disabled/enabled, including event-index and duration limits.

Compared values included event times, durations, PSTH time axes, displayed
heatmap matrices, per-file matrices, group matrices, and event records.
The existing PSTH calculation, filtering, normalization, and metric methods
were also verified unchanged by syntax-tree comparison with main.

Both sheets of a downloaded two-arena EthoVision workbook were checked against
the original loader: 15,001 samples per sheet, with five and seven inferred
binary columns respectively. Time arrays, binary states, trajectories, and
extracted bouts matched exactly. A SHA-256 comparison confirmed the workbook
was unchanged. This audit found and corrected the new import route's accidental
disabling of main's numeric interpolation default. A gapped-workbook regression
test now protects that behavior.

## New behavior checks

GUI tests cover automatic selection, rapid-click coalescing, keyboard toggles,
removal of the currently plotted label, intentionally empty selections, timing
help, mixed-unit group rejection, arena changes, identity pairing, and embedded
project round trips. Numerical tests cover before/start, whole-bout, after/end
windows, offsets, missing signal, incomplete recording boundaries, and equal
recording weights in groups.

The corrected real-MAMIR-event example passed individual and two-recording
group checks with explicitly simulated fiber signals. It is a software
demonstration, not an experimental pairing or neural result. The critical
correction above supersedes the earlier example's named-behavior results.

## Test-suite status

The final release-candidate run collected 376 tests: 372 passed, four skipped,
with no failures or errors (100.1 seconds). The source GUI startup smoke test
and CLI version check also passed; the CLI reports `pyBer 0.53.0`.

An earlier run had five errors also reproduced with original main: four from
the missing `pybaselines` dependency and one from Qt submenu lifetime. The
release check uses an isolated environment with that declared dependency
installed. View submenus now have explicit Qt parentage and retained wrappers;
the regression test exercises action lookup after garbage collection.
The Windows workflow separately gates publication on regression tests, both
executable builds and built-executable smoke checks.
The subsequent 70-test focused run passed importer/comparison, GUI, group,
metric, event, publication and widget-export checks. A separate 19-test run
passed all metric and behavior-summary panel checks after the annotation fixes.

## Plot label layout

Behavior comparison names now occupy a separate, horizontally scrollable
legend above the graph, never its data rectangle. Tests cover 24 long names,
literal text rendering and clearing stale entries. Metric and behavior-summary
annotations reserve headroom based on their actual pixel height, including
after resizing; tests check separation from the highest point/error bound in
compact and larger cards. Matplotlib metric-export notes sit outside the axes,
below the title. These changes affect presentation, not computed values.

## Behavior usability follow-up

The resizable layout and visible PSTH control mirrors passed a 64-test focused
run covering behavior import/GUI, the original compact dashboard and empty
states, MAMIR comparisons, event filters, PSTH behavior summaries, and metrics.
Additional interaction checks verify large whole-row targets, filtered bulk
selection, two-way PSTH controls, remembered divider sizes, collapse/restore,
plot navigation, and unchanged result matrices after resizing and switching
back to Zone. Screenshot checks use isolated simulated-fiber demos.

## Group / individual audit

An additional direct comparison with original main passed exact equality in
54 combinations of Individual / animal-level Group / pooled group trials,
onset / offset / transition alignment, all three baseline modes, and
minimum-event exclusion enabled/disabled. Checks included matrices, event
times, durations, group-trial matrices, labels, and excluded-file reports.

The focused group audit passed 94 tests spanning GUI/import, the numerical core,
PSTH exports, metric panels, and behavior summaries. Known-value tests verify
that one animal at signal 2 with one bout and another at signal 8 with three
bouts give an animal-level group mean of 5 (SEM 3), not the pooled-trial mean
of 6.5. Individual mode still displays all three events for the second animal.
Flat baselines are excluded from z-score results but remain usable with baseline
subtraction, matching main. Pairings now survive reorder/removal; duplicate
recording names cannot be counted twice. Comparison CSVs include analysis
settings and contributor IDs, and group SEM is displayed.
All ten dedicated group tests also pass, including per-behavior missing-data N,
scope-switch export invalidation, partial-sync warnings, and clock provenance.

## Interpretation limits

The optional comparison uses complete bouts in processed signal units, while
PSTH retains its own filters and normalization. Its explicitly labeled
comparison-only offset does not shift the heatmap. Before/after windows can
overlap other behaviors. The heatmap remains fixed-time event aligned; it does
not stretch each bout into equal before/during/after sections. Synchronization
and the correct fiber/animal association still require experimental validation.

# Selectable behavior panel

In **Postprocessing > PSTH > Behavior panel**, choose the parameter to display:

| Display | Meaning |
| --- | --- |
| Bout duration | Distribution of complete, observed selected-bout lengths. |
| Frequency over time | Selected bout onsets per observed minute in each time bin. |
| Inter-bout interval | Time from the previous selected bout's end to the next selected bout's onset. |
| Cumulative duration | Accumulated observed time spent in the selected bouts. |
| Occupancy over time | Percentage of observed time covered by the union of known selected bouts. |
| Cumulative bout count | Running count of observed selected-bout onsets, including point events. |
| Onset-to-onset interval | Distribution of time between adjacent selected onsets within one observed segment. |
| Bout duration over time | Complete-bout median duration and interquartile range (IQR), assigned to the bout onset's bin. |

Type a bin width in seconds and press Enter. Time plots always use that width;
distribution plots allow automatic bins or a manually entered width. Choices
update the cached plot immediately, support undo/redo, and persist with the
postprocessing settings.

## Scientific definitions

The panel follows the currently selected recording or included group and the
same accepted event rows as the PSTH. Consequently, event filters and baseline
validity affect the bouts being summarized. This is a summary of **selected PSTH
events**, not an independent estimate using every event in the original file.
Files without accepted PSTH events do not enter this panel's group summaries.

Time plots use seconds from the beginning of each recording. For each bin,
frequency is `60 × selected onsets / observed seconds`. NaN signal cuts and
missing recording periods do not add to observed time. Partial final bins use
their actual observed duration. A bin with no observations is missing, not zero.

IBIs are formed within each recording and observed segment only. They use
offset-to-onset gaps, not onset-to-onset intervals. Overlapping bouts are treated
as one occupied period when determining the next gap. An unknown end prevents
constructing an interval from that bout to its successor. Point-event files can
show frequency, cumulative counts and onset-to-onset intervals. Unknown durations
are not invented for duration, occupancy or offset-to-onset IBI metrics.

Cumulative duration integrates each bout across bin boundaries. Overlapping
bouts count once, and only the observed portions contribute. It carries the last
observed total through gaps and beyond the end of shorter recordings; a plateau
means no additional **observed** duration, not proof of no unobserved behavior.
The fixed recording cohort keeps the group cumulative curve monotonic.

Duration and IBI histograms pool selected bouts across recordings. Group time
curves give each contributing recording equal weight. Frequency uses the files
observed within each bin; cumulative duration uses a fixed cohort with known
duration information. Faint curves show individual files and the shaded band
shows SEM where at least two files contribute. Files are the averaging units,
which only correspond to animals if each file represents a different animal.

Occupancy divides the union of known occupied seconds by observed seconds in
each bin and multiplies by 100. Overlaps never produce occupancy above 100%.
When some bout ends are unknown, the reported known occupied time is a lower
bound on true occupancy. Point-only files have unavailable occupancy, not zero.
Cumulative counts retain the fixed recording cohort and carry the final observed
count through gaps and beyond each recording's end, just like cumulative duration.

**Bout duration over time is a descriptive median/IQR plot, not a mean/SEM plot.**
Only bouts entirely inside an observed segment enter it. Their full duration is
assigned to their onset bin, even when the bout spans several display bins.
Empty bins remain missing. One recording shows the median and IQR of its bouts;
groups show the median and IQR of recording-level medians in each bin, giving
each contributing recording equal weight. The shaded IQR is spread, not a
confidence interval. Per-recording quartiles are also exported.

## Median and quartile annotations

Histograms display a dashed median marker and a compact median/IQR annotation.
These statistics use the exact unbinned observations, so changing the histogram
bin width cannot move the median. Time plots report a median and IQR across
recordings of a clearly defined whole-recording quantity: total onsets per
observed minute, overall occupancy, final cumulative count/duration, or the
complete-bout duration median. They do not take the median of the visible bins.
The figure tooltip and exported statistics explicitly state the definition and
sample count. For a single recording the between-recording IQR is necessarily
zero; it must not be interpreted as no within-recording variability.

## Appearance and export

The panel uses the active plot palette, spaced slender histogram bars, restrained
per-file lines, and translucent cumulative/SEM shading. Axes and titles update
with the selected metric. Plot images export the figure without the controls.

In **Export Results**, **Event durations + selected behavior summary** retains
the original duration export and adds `_behavior_summary.csv` and/or `.h5`, plus
a JSON description of the metric, binning, file IDs and units. The table contains
the exact displayed bins, group values, SEM and each recording's values.
Median/IQR definitions, values and per-file statistics are in the JSON and H5
metadata. Duration-over-time exports add `lower` and `upper` IQR columns/datasets;
SEM is unavailable for this median display, and is exported as NaN rather than
being mislabeled as IQR.
**Heatmap + selected behavior panel** exports the currently chosen chart next to
the heatmap.

## Validation

The numerical regression tests cover known intervals, overlaps, missing coverage,
point events, unequal recording lengths, group SEM and CSV/H5 export parity.
GUI tests check keyboard entry, immediate redraw without PSTH computation,
individual/group scope, themes, settings, undo/redo and invalid bin recovery.

Run `python scripts/validate_behavior_summary.py` in the pyBer environment for
60 reproducible seed/bin combinations. An independent endpoint integration checks
the exact cumulative total; frequency times observed exposure recovers the event
count. Results and timings are written to
`_test/behavior_summary_validation/checks.csv`. The legacy histogram timing is a
reference for the previous simpler calculation, not a like-for-like speed claim.

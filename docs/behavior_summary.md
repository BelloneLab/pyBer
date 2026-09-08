# Selectable behavior panel

In **Postprocessing > PSTH > Behavior panel**, choose the parameter to display:

| Display | Meaning |
| --- | --- |
| Bout duration | Distribution of complete, observed selected-bout lengths. |
| Frequency over time | Selected bout onsets per observed minute in each time bin. |
| Inter-bout interval | Time from the previous selected bout's end to the next selected bout's onset. |
| Cumulative duration | Accumulated observed time spent in the selected bouts. |

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
show frequency, but unknown durations are not invented for the other metrics.

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

## Appearance and export

The panel uses the active plot palette, spaced slender histogram bars, restrained
per-file lines, and translucent cumulative/SEM shading. Axes and titles update
with the selected metric. Plot images export the figure without the controls.

In **Export Results**, **Event durations + selected behavior summary** retains
the original duration export and adds `_behavior_summary.csv` and/or `.h5`, plus
a JSON description of the metric, binning, file IDs and units. The table contains
the exact displayed bins, group values, SEM and each recording's values.
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

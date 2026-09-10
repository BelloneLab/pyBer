# Compact PSTH dashboard

The Standard view keeps related plots in two shared cards beneath the trace preview:

- Left: heatmap above average PSTH in one continuous card. Both use identical
  plot widths and left-axis margins. The time axis is labeled once below PSTH;
  zooming or panning either plot updates both. The color-scale column is reserved
  beside both plots, including when the detailed histogram editor is open.
- Upper right: both selected behavior plots in one shared card, separated by
  just 2 pixels rather than two padded frames.
- Lower right: the compact pre/post comparison and selected global summary.

The left and right sections each receive half the available width. Optional
additional comparisons remain available through **More metrics**, with three
compact cards per additional row. Heatmap image export includes the paired PSTH;
PSTH/dashboard export includes the shared cards and comparison summaries.

## View menu and saved layout

Both workflows use a single 44-pixel toolbar above their plots. Postprocessing
keeps File, PSTH, Export, undo/redo, View, Individual/Group, the recording selector,
status and the drawer toggle together. Reset is in File; plot styling and help
are in View. Full recording names remain available on hover.

Preprocessing keeps File, QC, Export, undo/redo, Selection, View and recording
context on the same line. Selection contains Add from selector, Box select and
Clear manual regions. View contains thresholds, plot styling and sensor settings.
These menu entries use the existing editing actions and keyboard shortcuts.

The toolbar **View** menu contains layout, theme, heatmap contrast, automatic
scaling, plot fitting and the detailed scale editor. This replaces the separate
controls row. **Save current view** writes preferences immediately; normal edits
and divider drags save automatically.

Drag dividers to resize the trace versus the dashboard, the time plots versus
summaries, the behavior row versus comparisons, and each pair of side plots.
**Reset panel sizes** returns to the default proportions. Hidden sections retain
their proportions. Sizes, selected view settings and scale-editor visibility are
included in project files and app preferences. An explicitly selected Paper
theme remains Paper when the app restarts with a dark application theme.

The trace has a 190-pixel plot minimum and a 240-pixel card minimum, so its time
axis remains inside the panel even with the signal-detection caption visible.
Smaller windows scroll the results rather than cutting off plot labels.

Projects restore their embedded behavior, trajectory and time-column data when
opened through Project Open, the processed-file button, or drag-and-drop. Linked
CSV files are not needed and cannot silently replace the saved snapshot.

Choose the two behavior plots independently under **Behavior panel**. Their
shared bin width and automatic distribution-bin option apply consistently to
both plots. Both numerical summaries are exported, with `_second` identifying
the second selection. Existing median and IQR annotations remain visible.

**Global metrics > Display** selects exactly one of ten summaries:

| Choice | Definition |
| --- | --- |
| Mean peak amplitude | Mean processed value at detected peaks |
| Transient frequency | Peak count divided by observed duration |
| Transient count | Number of detected peaks |
| Median signal | Median finite sample |
| Signal interquartile range | 75th minus 25th percentile |
| Signal variability | Sample standard deviation |
| Root mean square | Square root of mean squared samples, including offset |
| Robust signal range | 95th minus 5th percentile |
| Integrated signal | Signed trapezoidal area over observed segments |
| Median inter-peak interval | Median successive-peak interval within segments |

These summaries use original processed units, independently of PSTH
normalization. Individual view uses the selected recording; Group view displays
one point per recording, a median line, and mean with descriptive SEM. Selecting
a metric or changing its global time range does not recompute PSTH trials.

The historical global peak detector is retained for compatibility. Its threshold
is three times the median after excluding high local maxima. It is independent
of Signal Events settings and is not a universal noise threshold. Distribution
summaries and integration do not depend on this detector. Missing samples and
large timestamp gaps break peak neighborhoods, integration and inter-peak
intervals; they contribute no observed duration to frequency.

CSV retains existing global columns and appends the new summaries. HDF5 includes
the expanded global group. Group scalar summaries are recording-level means;
the legacy `global_peaks` field remains the total count across recordings.

Validation covers numerical reference formulas, legacy peak parity, cuts and
gaps, selected-file scope, group aggregation, persistence, independent behavior
choices, and placement. A local benchmark on 100,000 synthetic samples measured
about 11 ms for all ten summaries versus 4.9 ms for the previous two. The minimum
height of the PSTH plus primary-comparison area falls from 460 to 220 pixels.


The shared-card alignment regression checks time coordinates to within one pixel
at 1000- and 1400-pixel dashboard widths, with both compact and detailed color
scales, and after zooming from either time plot. Plot ranges are synchronized
explicitly so a temporary resize mismatch cannot change the shared time limits.

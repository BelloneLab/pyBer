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

# Compact PSTH dashboard

The Standard view has two aligned rows beneath the trace preview:

| Main figure | Compact figure | Compact figure |
| --- | --- | --- |
| Heatmap and color scale | First behavior choice | Second behavior choice |
| Average PSTH | Selected pre/post comparison | Selected global metric |

The main column receives half the width and each side column one quarter.
The primary comparison shares the PSTH row instead of adding a full-width row.
Optional additional comparisons remain available through **More metrics**, with
three compact cards per additional row.

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

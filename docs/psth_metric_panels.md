# Selectable PSTH metrics

Choose the primary metric in **PSTH metrics**, then select additional panels in
**More metrics**. The default remains AUC. Each selected metric has its own scale
and units. Colored horizontal lines show medians, points show paired rows,
and diamonds show means with descriptive SEM. Missing pairs remain missing;
available unpaired observations are still visible in their own distribution.

Available metrics are mean signal, integrated AUC, median signal, maximum signal,
minimum signal, sample standard deviation, and peak delay. Peak delay measures
time from the start of each comparison window to its first maximum. It is not
an independently detected onset latency; flat traces have no defined peak delay.

Mean retains the existing available-bin calculation. AUC uses trapezoidal
integration including exact requested boundaries and requires complete coverage.
The new distribution and extremum metrics also require complete observed window
coverage. Sample SD uses N-1 and describes variation within a waveform window,
not variation across trials. Peak and minimum values use the available grid bins.

Each panel summarizes the currently displayed rows. In a grouped file-average
view, nonlinear metrics are calculated on each file's averaged waveform. A peak
of an averaged waveform is not the mean of individual trial peaks. A file is not
automatically an independent animal; users must establish independent sampling
units. Pooled group trials receive descriptive summaries without a p-value.

Finite paired sign-test p-values receive Holm correction across the selected
metric family. Raw and adjusted values, family membership, sample counts, units,
window boundaries, medians and means are exported. Unequal-length comparison
windows suppress inference for AUC, extrema and peak delay because their values
depend on duration. Holm adjustment does not correct repeatedly trying other
metrics, windows, events or normalization methods. Predefine confirmatory choices.

Result export preserves the legacy primary metric table and adds
`_metrics_selected.csv`, `_metrics_rows.csv`, and `_metrics_selected.h5` when their
formats are selected. Average plot export includes the complete panel layout and
separate `_plot_metric_<id>` PNG/PDF images. Publication figures give each selected
metric a separate column in PNG, PDF and SVG.

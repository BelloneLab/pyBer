# Recording groups, group PSTH, editable dashboard and vector export

This guide covers four postprocessing features: recording groups, the group
view of the PSTH, the editable results figure, and editable PDF/SVG export.
None of them changes trial matrices, events or statistics definitions; they
change which rows are shown together and how the figure is drawn.

## 1. Recordings and groups

**Setup > Behavior / Events > Recordings and groups** holds the processed
recordings list, the behaviors list and a row of group chips.

- **Open in window** moves the same lists into a separate resizable window,
  convenient for long lists. **Dock back** (or closing the window) returns them
  to the drawer. Drag-reordering, drop targets and removal keep working in both
  places, and the window size is remembered.
- **+ New group** asks for a label (WT, Het, Saline...). The selected
  recordings join the new group. Colors come from a validated colorblind-aware
  categorical palette, in a fixed order.
- **Attribute recordings**: select one or several recordings, then click a group
  chip. Alternatively right-click the selection and choose
  **Attribute to group**. **Ungroup** / **Remove from group** clears it.
  Right-clicking a row that is not selected targets that row only.
- **Edit a group**: right-click its chip to rename it, change its color,
  reorder it or delete it. Deleting a group never removes data; its recordings
  become ungrouped.
- Every attributed recording is outlined in its group color and tagged with
  the group name. Behavior files matched to a grouped recording show the same
  outline. The toolbar recording picker shows a colored dot.

### Managing the lists

- **+** above a list adds files (several can be selected at once); added
  recordings are appended, not replaced. **x** above a list removes the
  selected rows; the **x** at the end of a row removes that row only. Files on
  disk are never touched.
- Drag a row by its grip (the six dots on the left) to change the order.
- Right-click a list for: add files, remove selected, move up / down / to top /
  to bottom, auto match rows by name, open the containing folder, group
  attribution, and open/dock the window.
- Pairing is line-wise: a newly added behavior file is paired automatically
  with the first recording line that has no behavior yet, without asking. Only
  when every recording already has a behavior file does pyBer ask which
  recording to use. Multi-animal tables still ask which animal to read, but not
  which fiber. MAMIR zone and behavior files of the same animal from the same
  export folder join the same line.
- Reordering (drag, Move up/down, Auto match rows by name) re-pairs by line:
  recording 1 with behavior 1, and so on. Right-click a behavior and use
  **Pair with recording** to pin a specific pairing.
- The behaviors list shows behavior file names. Both lists keep identical row
  heights and scroll together so lines face each other. A behavior row not used
  by the recording on its own line is flagged (`pairs with #3`, `unpaired`), and
  a recording without behavior shows `no behavior`. A behavior file named after
  one recording is never reused for a second recording.

Groups are stored per recording file id (the processed file name), so they
survive reordering, reloading, project save/load and undo/redo.

## 2. Group view in the PSTH drawer

**PSTH > Groups**:

| Control | Effect |
|---|---|
| **Group view** | Turns group display on. It applies to the **Group** scope of the toolbar (it switches to it when several recordings are loaded). |
| **Show: All groups** | Heatmap rows are grouped into colored blocks (a color strip beside the heatmap, never over the data, and one label per block). The average PSTH draws one mean +/- SEM per group with a legend such as `WT (n=12)`. Pre/post and global metric points are tinted by group. |
| **Show: Only <group>** | Heatmap, average PSTH, pre/post metrics, global metrics and behavior summaries use that group's recordings only. |
| **Show: Pooled** | All recordings together, exactly as without groups. |

Statistics: the per-group curve is computed with the same `mean_sem` function
as the pooled view, on that group's rows. With recording rows (default), n is
the number of recordings and SEM is across recordings. With **Keep trial rows
in Group view**, n counts trials and the legend says so (`n=57 trials, 6 rec.`);
treat trial-level SEM as descriptive, because trials of one animal are not
independent. Recordings without a group appear as **Ungrouped** (grey) in the
All groups view, so nothing is dropped silently. Pre/post p-values remain the
pooled paired tests of the displayed rows; between-group tests are
described in "Metrics by group" below.

### Metrics by group (All groups)

With **Show: All groups** and at least two groups, the PSTH metric panels and
the global metric panel switch to a per-group layout: one column per group,
one point per animal, with the test written above the points.

Two selectors in **PSTH → Metrics** (also in the right-click menu of a metric
panel) choose what is drawn. Changing them redraws the panels without
recomputing the PSTH.

| Group comparison | Drawn per group | Statistics |
|---|---|---|
| Post − Pre (change) | change per animal | between groups, on the change |
| Post only | Post window value | between groups, on Post |
| Pre only | Pre window value | between groups, on Pre (baseline check) |
| Pre and Post side by side | Pre and Post columns, each animal joined by a line | within each group: paired Pre vs Post (paired t-test if the differences pass Shapiro-Wilk, otherwise Wilcoxon signed-rank; Holm across groups). Between groups: on the change |

**Summary style** draws the same numbers three ways: points with the median
line and mean ± SEM diamond, bars (mean ± SEM) with points, or box plots
(median, quartiles, 1.5 IQR whiskers) with points. It also applies to the
global metric panel.

Annotations: the omnibus or two-group test and its effect size are written
above the data. Brackets mark the pairs with a Holm-adjusted p < 0.05 (at most
three, the most significant first). In the paired view the label above each
pair is the Holm-adjusted paired test of that group. Nothing is written for a
comparison that could not be tested (fewer than 3 animals in a group).

- **Unit of analysis = animal / recording.** Trial rows (Keep trial rows) are
  first averaged per recording, so n is always the number of animals.
- **PSTH metrics** compare the quantity chosen in **Group comparison** (default:
  **Post − Pre**, the selected metric in the post window minus the pre window).
  **Global metrics** compare one value per recording.
- **Test choice** (reported in the panel tooltip and the CSV):
  - 2 groups: Welch t-test if every group passes Shapiro-Wilk (p > 0.05),
    otherwise Mann-Whitney U. Effect size: Hedges g or rank-biserial r.
  - 3+ groups: one-way ANOVA if all groups are normal and Levene does not reject
    equal variances, otherwise Kruskal-Wallis (eta² or epsilon²), with pairwise
    Welch t / Mann-Whitney tests corrected by Holm.
  - A group with fewer than 3 animals gives a descriptive result only.
- **Ungrouped** recordings are shown but never tested.
- Exports: `<prefix>_metrics_selected_groups.csv` (n, mean, SEM, median, SD per
  group and the test), `_group_pairwise.csv` (3+ groups) and `_group_rows.csv`
  (per-animal pre, post and post − pre), plus `_group_paired.csv` (paired Pre vs
  Post test per group) when the paired view is selected; the same files with
  `_metrics_global` for global metrics. The `quantity` column names what was
  tested (`post_minus_pre`, `post` or `pre`). **Pooled** and **Only <group>** keep the original display.

With few animals per group the tests have little power; report effect sizes
and confidence in the per-animal values, not only p.

Exports add `<prefix>_avg_psth_groups.csv` (time, mean, SEM and n per group)
when All groups is shown, and `<prefix>_recording_groups.csv` (file id, group,
color, source path) whenever groups exist.

### Individual traces behind the mean

Right-click the mean PSTH and open **Individual traces**. The same controls
are available under **Hide/show traces** in Individual scope and **Hide/show
groups > Individual traces** in Group scope. Each line represents one trial,
or one animal when Group scope displays animal averages.

- Choose **Average and individual traces** to see the bold mean and its
  outlined, filled ±1 SEM band above the trial lines. SEM uses the finite
  displayed rows at each time sample and requires at least two. **Average
  only** is the default. **Individual traces only** shows the responses on
  their own.
- **Color by** defaults to trial order. Choose response amplitude to rank
  each trace by its largest absolute excursion in the **Post** metric
  window relative to its mean in **Pre**. This gives equal weight to upward
  and downward responses of the same magnitude. Both windows include their
  endpoints and use finite samples. A trace with no finite samples in
  either window appears in a neutral shade.
- **Trace colors** lets you choose a single hue or multiple colors. In a
  single hue, earlier trials or smaller responses have the most muted shade.
  Later trials or larger responses gain contrast against the plot background.
  Each group has its own single-hue scale. To compare amplitude ranks across
  groups, choose **One color per trace**, which uses a shared color scale
  within the plot. Tied amplitudes share a shade. Changing the ranking
  preserves the row order and trial labels.
- **Trace opacity** adjusts the individual lines independently of the mean
  and SEM. Automatic opacity makes dense overlays lighter. Choose a preset
  percentage or select **Custom** to enter an opacity from 1% to 100%.
- **Line style** offers solid, dashed, dotted and dash-dot lines. Enable
  **Trace legend** to label a selection of traces, or point at a line to
  identify it. In amplitude mode the hover label includes the response
  amplitude in the plotted signal units.

For more than 300 rows, the overlay draws a sample of up to 300 traces. The
mean and SEM still use all displayed rows. Trace preferences are saved with
the settings and project.

### Several behaviors side by side

Check two or more behaviors in the Events panel (with **Align to: Behavior**).
The single heatmap / PSTH row is replaced by one column per checked behavior,
in the order of the list:

- **Heatmap**: every column uses the same color limits, shown by one color bar,
  so intensities can be compared between behaviors. The limits follow the
  heatmap contrast setting and are computed from all shown behaviors together.
- **Mean PSTH ± SEM**: every column uses the same y range, in the color the
  behavior has in the comparison plot. The individual-trace options above apply
  to each column.
- **Metric**: the selected PSTH metric, Pre vs Post, for that behavior.
- **Trace preview**: the events of every checked behavior are drawn in its
  color (a line per event, a shaded span per bout), with a legend giving the
  number of events.

Each column is computed with the same function, filters, windows and
normalization as the single-behavior view, so a column shows exactly what that
behavior shows alone. Up to six behaviors are drawn.

Nothing is cut when space is short: a column is never narrower than 300 px;
columns that do not fit wrap onto another row and the results area scrolls.
The y-axis label is written once per row and the behavior name is shortened
with its full text in the tooltip. Unchecking down to one behavior restores
the standard view with its behavior and global panels. Figure exports of the
PSTH dashboard export the columns while they are shown.

## 3. One seamless figure with editable panels

The results area is drawn as a single figure: no card outlines between panels.
Every boundary is still a divider: hover to see it, drag to resize. The divider
between the heatmap and the average PSTH keeps both time axes aligned;
double-click it to restore equal heights. Sizes are saved with preferences and
projects.

Heatmap colors (contrast mode, auto scale, exact limits, palette, histogram
editor) are in the right-click menu of the heatmap color bar; there is no color
row above the heatmap.

Right-click any plot to get the panel actions above the usual plot menu:

- **Remove this panel**: the neighboring panel takes the space.
- **Change to**: show another analysis in this place, chosen from
  *Behavior summary* (bout duration, inter-bout interval, frequency...),
  *Pre/post comparison* (AUC, peak, mean...) or *Global signal* metrics.
  Choosing an analysis of the panel's own kind simply switches its selector;
  another kind is drawn by a substitute plot in the same place.
  **Original** brings the initial panel back.
- **Restore removed panels** and **Reset dashboard layout**.

All displayed pre/post comparisons, including panels changed to a metric, form
one Holm family, so adding a comparison panel correctly widens the multiple
comparison correction.

### Trial-response PCA and cluster separation

Under **Change to > State-space (PCA)**, choose **PCA trajectories**. In
**Individual** view, the recording picker controls the data used to fit the
trajectories, loadings, and separation panels. Switching recordings refits
the axes, so each recording has its own PC1 and PC2. Use **Group** view for
a shared coordinate system across recordings. If Group view uses
**Recordings as dimensions**, switching to Individual shows trial-response PCA
for that file. Returning to Group restores the multi-recording method.

Right-click a PCA panel and choose **PCA options > Method > Trial-response
PCA**. Each point represents one event-aligned response. **Show GMM clusters**
colors the trials by their fitted Gaussian mixture component. Each outline
encloses 95% probability under that component in the displayed PC plane.
Changing the displayed plane preserves the assignments.

The mixture uses up to three retained PCs. The Bayesian information criterion
(BIC) selects among eligible models with one to six components. The number
of complete trials limits how many components can compete. A split must have
at least five assigned trials per component, or seven in three dimensions,
and enough variation to fit a covariance matrix. For more than 2,000 complete
trials, a reproducible sample of 2,000 determines the model and all complete
trials receive an assignment. When several models qualify, the tooltip reports
the BIC difference from the next-best model.

To inspect overlap, right-click a second panel and choose **Change to >
State-space (PCA) > Cluster separation**. Each bar shows one trial's
silhouette score in the fitted PC space. Values near 1 indicate clear
separation, values near 0 indicate overlap, and negative values indicate a
trial lies closer on average to another cluster. The dashed line shows the
mean. For clusters larger than 256 trials, distances use a reproducible
sample of 256 reference trials per cluster. With one cluster, the panel
reports the model-selection outcome.
This is a descriptive check on the fitted groups. The separate **PCA
separation test** compares predefined experimental conditions.

## 4. Editable PDF and SVG export

Image export now writes **PNG + PDF + SVG**. The PDF and SVG are rebuilt from
the displayed plot items (not a screenshot):

- each curve is one path, SEM bands and bars are plain filled shapes;
- text (titles, tick labels, axis labels, legends) is real text with an
  embedded TrueType font;
- colors are plain RGB values, so restyling in Illustrator or Inkscape works;
- objects carry readable ids (e.g. `PSTH_..._curve`), shown in the SVG layer
  panel;
- only matrices (heatmaps and the color bar) remain images.

Measured on the trial_0027 recording dashboard (1554 x 906 px): the previous
Qt export embedded 126 raster images in the PDF and produced SVG curves as
thousands of 2-point polylines with `non-scaling-stroke` transforms; the new
export has 2 images (heatmap, color bar), 0 polylines and 0 non-scaling
strokes, and the PDF is smaller (230 KiB instead of 380 KiB). Writing
PNG + PDF + SVG takes 1.1 to 1.7 s (the first export also loads matplotlib)
instead of about 0.3 s for the previous PDF + SVG: the price of a full vector
rebuild.

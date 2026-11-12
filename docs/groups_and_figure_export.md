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
pooled paired tests of the displayed rows; a between-group test is not
computed.

Exports add `<prefix>_avg_psth_groups.csv` (time, mean, SEM and n per group)
when All groups is shown, and `<prefix>_recording_groups.csv` (file id, group,
color, source path) whenever groups exist.

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

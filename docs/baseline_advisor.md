# PSTH baseline advisor

In **Postprocessing > PSTH > Window & baseline**, choose **Recommend baseline**.
Review the suggested window and its diagnostics, then choose **Apply suggested
window**. Applying the suggestion is undoable and preserves the normalization
method and event selections. The existing PSTH validity rules still apply, so
usable trial counts can change after changing the baseline.

The tool recommends a reference for a specific recording and event definition.
It can return **No reliable common baseline window found**. It does not claim a
universally optimal baseline or validate neural-response significance.

## Controls and scope

- **Search before event:** maximum lookback, independent of the displayed PSTH
  window. A reference can start before the plotted epoch.
- **Pre-event guard:** time excluded before every known selected-source event.
- **Post-bout recovery:** time excluded after each known bout ends. Choose this
  from the sensor and experimental protocol. Autocorrelation is not a direct
  measurement of sensor decay or biological recovery.
- **Minimum duration:** shortest candidate to assess. Duration alone is not a
  guarantee of sufficient information.
- Individual view assesses the selected recording. Group view requires the same
  relative window to pass in every loaded recording, including files excluded
  from the displayed group PSTH.

Whole bouts are excluded, including events removed by the PSTH filters. For
offset alignment the current bout can occupy negative relative times and is
excluded too. Transition alignment also excludes its two component behaviors.
Unknown durations are represented as point events and reported. Other behavior
types that are not part of the selected event source are not automatically
excluded; their potential effects still require experimental judgment.

## How the recommendation is made

1. **Validate native measurements.** Timestamps must be finite and increasing.
   NaN cuts and timestamp gaps are preserved. Candidate windows need observed
   samples throughout; interpolation never supplies extra information.
2. **Separate events chronologically.** Earlier events form the search set.
   Later events form the checking set, with the entire search lookback purged
   from their boundary plus a training-derived temporal buffer. This reduces
   shared data and nearby dependence. It does not prove statistical independence.
3. **Estimate temporal information.** Estimate the autocorrelation of both the
   signal and its centered squared values within contiguous reference segments.
   Squared-value dependence matters for estimating a standard deviation.
   The approximate information timescale is
   `dt * (1 + 2 * sum(positive autocorrelations))`.
   Both the broad search context and each candidate are checked; the slower
   timescale is retained. Effective sample count is approximately reference
   duration divided by this timescale, capped by the observed sample count.
   Truncated or unsupported estimates fail the information check.
4. **Check distribution stability.** Compare the means and standard deviations
   of the first and second halves of each window, and the distribution of
   baseline SDs across trials. Skewness and tail frequency are reported without
   assuming Gaussian signals. A stationary skewed signal is not automatically
   rejected because it is skewed.
5. **Rank on the earlier references only.** Reward observed event-free coverage,
   adequate information, stable location/scale and proximity to the event.
   The absolute size of the SD, post-event response amplitude and statistical
   significance are never optimization targets. The search score is a heuristic
   ranking, not a confidence percentage or probability.
6. **Check the first choice once on later events.** All files must pass again,
   including a check for a change of SD between the earlier and later blocks.
   A failed check causes abstention. The tool does not try other candidates
   against the same later events until one passes.
7. **Audit every selected event.** Search work is capped using evenly spaced
   events for large batches. A proposed window is then checked against all
   selected events before being offered. Full coverage is reported explicitly.

The baseline quality checks do not modify the signal or independently remove
PSTH trials. Current-window diagnostics allow comparison with the proposal.

## Default policies

These are conservative software heuristics that have been tested on simulations,
not universal thresholds established by the literature. All are configurable in
`BaselineAdvisorConfig` at the beginning of `pyBer/baseline_advisor.py` and are
included in exported reports.

| Policy | Default |
| --- | --- |
| Search lookback / minimum duration | 30 s / 1 s |
| Pre-event guard / post-bout recovery | 0.25 s / 1 s |
| Minimum native samples per reference | 20 |
| Minimum estimated effective samples, lower decile | 20 |
| Minimum usable references in each recording | 80% |
| Minimum earlier / later events | 8 / 4 |
| Maximum half-window mean difference, in window SDs, upper decile | 1 |
| Maximum half-window SD ratio, upper decile | 3 |
| Maximum across-trial baseline SD ratio, 90th / 10th percentile | 4 |
| Maximum earlier / later median baseline SD ratio | 3 |
| Search event cap per block and recording | 100 |
| Maximum rate used for autocorrelation estimation | 100 Hz |

For high-rate recordings, autocorrelation is evaluated on bin means formed
inside contiguous segments only. It does not count the display resampling rate
as independent information. Information estimates are approximate; no exact
confidence interval for SD or false-positive probability is claimed.

## Validation and reproduction

Run `python scripts/validate_baseline_advisor.py --seeds 10` in the pyBer
environment, or run that script directly from an IDE. Experiment parameters,
random seeds and figure styles are at the top of the script. Outputs go to
`_test/baseline_advisor_validation`: per-run CSV, summaries, configuration,
invariance checks and figures in PNG, PDF and SVG.

The repeated scenarios include white and correlated noise, sparse asymmetric
transients, dense events, long preceding bouts, cuts, strong drift, flat signals
and later changes of scale. Comparisons use the existing one-second fixed
reference as a descriptive benchmark. An unrelated-event average looking
closer to zero is not used to select windows and is not treated as proof of
correct inference.

Additional regression checks cover offset alignment, filtered-out neighboring
events, mixed clean/unreliable groups, source preservation, invariance to changes
inside excluded responses, conversion of units, candidate-local dependence,
single-candidate validation, and GUI apply/undo and worker lifecycle behavior.

Validation snapshot with the default policy, seeds 5100 to 5109:

| Simulated reference | Recommendations / 10 runs |
| --- | --- |
| Stationary white noise | 10 |
| Stationary correlated noise | 10 |
| Sparse asymmetric transients | 2 |
| Dense events with overlapping protection intervals | 0 |
| Long preceding bouts with adequate intervening time | 10 |
| Repeated cuts | 0 |
| Strong drift | 0 |
| Flat signal | 0 |
| Later increase in variance | 0 |

All ten later-variance cases failed the later-event check. The final batch passed
nine declared invariance/abstention checks. Typical median runtime was 0.28 to
0.49 seconds per recording on the validation machine; heavily fragmented cuts
took about 1.96 seconds. The correlated-noise scenario had approximately 1.8
effective samples for the fixed one-second reference versus 29 for the advised
reference. These are descriptive engineering results, not a validation of
neural-response accuracy or a statistical false-positive rate.

Simulation performance does not establish biological validity for arbitrary
recordings. For confirmatory experiments, specify the baseline policy using the
protocol or independent pilot data, then lock it. Repeatedly changing settings
and rerunning on the same recordings does not create fresh validation data.

## Methodological basis

The advisor is a new engineering method informed by these principles:

- [PASTa methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC12224222/): stable
  reference periods and normalization-dependent interpretation.
- [Published photometry protocol](https://www.nature.com/articles/s41467-021-22260-7):
  reference selection justified by the inter-trial interval and preceding events.
- [Stan effective sample size documentation](https://mc-stan.org/docs/2_38/reference-manual/analysis.html):
  information loss under autocorrelation. Applying related diagnostics to
  baseline variance is an approximation, not an exact inferential procedure.
- [Roberts et al., 2017](https://doi.org/10.1111/ecog.02881): validation strategies
  must account for temporal and hierarchical dependence.
- [Kriegeskorte et al., 2009](https://www.nature.com/articles/nn.2303): dependent
  selection and analysis can bias conclusions.
- [NIST standard deviation confidence limits](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/sdconfli.htm):
  exact normal-theory SD intervals depend on assumptions that should not be
  silently applied to correlated photometry samples.

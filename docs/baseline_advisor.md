# Automatic PSTH baseline suggestions

**Postprocessing > PSTH > Window & baseline** offers up to three scored windows in a compact dropdown menu
directly below the baseline inputs. Loading a signal, choosing events, changing
the pre-event duration or switching recording scope refreshes them automatically.
Open the menu and select a choice to apply it. There is no separate dialog or Estimate step.

Every choice ends strictly before event time zero and lies inside the displayed
pre-event span, up to the baseline controls' 60-second limit. Applying a choice
changes only baseline start/end, preserves normalization and event selection,
and creates one undoable change. Usable PSTH trial counts may change under the
existing baseline validity rules.

## Reading the scores

The displayed percentage is **a fit score out of 100, not a confidence
probability, p-value or biological validation**. It ranks available candidates
using signal information, distribution stability, data coverage and known-event
timing. It does not establish a universally optimal baseline.

Best available windows remain visible when evidence is limited. Their scores
are capped and the inline status identifies the main limitation. Hover over each
choice for coverage, event-free coverage, estimated information, recording scope
and specific cautions.

Limitations include few events, slowly varying signals, cuts, preceding-bout
overlap, a short-window SD unlike the wider reference, and SD variation across
events. If every candidate is flat, missing, lacks enough native samples, or is
consistently dominated by a nearly deterministic slope, there is no choice. The
tool also abstains when no observed event-free reference can be assessed.

## Scope and event protection

Individual view uses the selected recording. Group view assesses every loaded
recording and offers one shared relative window. Weak files remain represented
in the score rather than disappearing from the assessment.

Every selected alignment event and every unfiltered selected-source bout is
checked, including bouts removed by PSTH event filters. Offset alignment protects
the complete preceding bout. Transition alignment additionally protects both
component behaviors, including bouts not forming a qualifying transition.
Unknown durations are point events. Other unselected behavior types are not
automatically treated as contamination.

Known bouts are expanded by 0.10 seconds before onset and 0.50 seconds after
offset. These are explicit engineering settings, not an inferred sensor decay
constant or a guarantee of biological recovery. Actual inter-bout spacing
determines which candidate windows overlap these protected intervals.

Signal statistics use only completely observed, event-free candidate windows.
Coverage and overlap still count **every selected event**. A choice with some
overlap can be shown as Limited with its overlap fraction and capped score.
Applying it does not silently mask those events. Review that limitation before
use, especially for confirmatory analysis.

## Calculation

1. Validate native clocks and contiguous finite segments. NaN cuts and timestamp
   gaps larger than three median native sample intervals break a segment.
   Interpolation never supplies baseline information.
2. Search a bounded grid of durations and offsets inside the current Pre span.
   Endpoints are whole seconds, evaluated exactly as displayed, and remain before
   the pre-event guard, including its boundary. Fractional Pre limits are rounded
   inward. If no whole-second interval fits, no suggestion is offered.
3. Check data and protected-bout coverage at every target event. Sample at most
   32 eligible events evenly through each recording for expensive statistics,
   retaining every loaded recording.
4. Estimate information from initial positive autocorrelations of the signal
   and its centered squared values within each candidate. Information is capped
   by the sampled native observations. This is an approximation, not a count of
   proven independent observations.
5. Check half-window mean/SD differences, SD variation across events, robust
   distribution shape and candidate SD relative to a wider event-free pre-event
   reference. Minimizing SD is never rewarded; an accidentally small denominator
   can score poorly.
6. Combine information, stability, representative scale, coverage and recency.
   File scores use 60% of their equal-file mean and 40% of their minimum, so weak
   recordings cannot be hidden by many strong ones. Explicit caps reflect
   overlap, missing coverage, few events and weak information. Display the three
   highest-ranking sufficiently distinct windows.

Signal diagnostics never inspect samples at or after the target event. Known
preceding-bout responses do not supply candidate statistics. Post-event amplitude,
pre/post effect size and significance are not optimization objectives.

## Defaults and performance

`BaselineSuggestionConfig`, at the beginning of `pyBer/baseline_suggestions.py`,
exposes the policy:

| Setting | Default |
| --- | --- |
| Search span | Current Pre value, capped at 60 s |
| Guard / post-bout protection | 0.10 s / 0.50 s |
| Minimum candidate duration | 1 s on the whole-second grid |
| Minimum native observations per assessed window | 6 |
| Maximum sampled events per recording | 32 |
| Maximum native samples per quality segment | 256 |
| Information target for Supported choices | 20 estimated effective samples |

Native samples are selected evenly when a segment exceeds the computation cap.
Compatible-length segments use batched statistics and FFTs without padding or
interpolation of observations. The GUI debounces edits for 240 ms, uses a
background thread pool, retains at most one numerical job per panel, and discards
superseded results. Baseline or normalization edits reuse cached choices when
signal, events and Pre span are unchanged.

The earlier [strict advisor](baseline_advisor_strict.md) remains a separate API.
Its mandatory chronological holdout and abstention policy is not silently
represented as a percentage in this faster descriptive ranking. Repeatedly
searching the same data is not independent validation.

## Validation

Run `python scripts/validate_inline_baseline.py` in the pyBer environment for
repeatable synthetic and read-only example-data benchmarks against the strict
advisor. The script writes timing/choice tables and figures. Tests cover sparse
events, known-bout overlap, gaps, constants, deterministic drift, units and
excluded-response invariance, batched/scalar numerical parity, automatic refresh,
cache reuse, stale-result suppression and explicit apply/undo.

The September 2026 validation used 27 seeded scenarios plus the provided
social-contact recording. All pre-event geometry, bounded-score, unit-invariance,
excluded-response-invariance and input-integrity checks passed. Median numerical
runtime was 0.118 seconds versus 0.991 seconds for the previous strict advisor;
the provided recording took 0.125 seconds. These timings exclude the deliberate
240 ms edit debounce and vary by machine and recording size. The two methods
have different evidence requirements, so this is a workflow-cost comparison.

Literature supports several baseline-referenced transformations, including
[GuPPy's per-event subtraction](https://www.nature.com/articles/s41598-021-03626-9)
and [FiPhA's baseline normalization](https://pmc.ncbi.nlm.nih.gov/articles/PMC10885510/).
These do not establish a universal baseline-selection score. This ranking is an
explicit software heuristic that should be checked against the protocol, ideally
using independent pilot recordings before locking a policy.

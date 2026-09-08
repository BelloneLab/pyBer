"""Reproducible synthetic benchmark, not biological validation. Run in IDE or Python."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pyBer.signal_events import detect_peaks, estimate_noise

# All synthetic design and display parameters are intentionally collected here.
OUT = Path(__file__).resolve().parents[1] / '_test' / 'signal_events_validation'
SEEDS = range(20)
DT, DURATION, SIGMA, MULTIPLIER = .02, 120., .03, 5.
EVENT_TIMES = np.arange(10., 115., 10.)
MATCH_TOLERANCE = .35
BASELINE_WINDOW_SEC = 10.0
MIN_DISTANCE_SEC = 0.5
DRIFT_AMPLITUDE, DRIFT_PERIOD_SEC = 0.8, 100.0
EVENT_AMPLITUDE, EVENT_WIDTH_SEC = 0.4, 0.15
DENSE_EVENT_START_SEC, DENSE_EVENT_END_SEC, DENSE_EVENT_STEP_SEC = 3.0, 118.0, 1.5
NOISE_CHANGE_SEC, LATE_NOISE_FACTOR = 60.0, 3.0
CUT_START_SEC, CUT_END_SEC = 54.0, 56.0
SCENARIOS = ['stationary', 'slow_drift', 'cut_drift', 'changing_noise', 'dense_activity']
METHODS = ['Previous whole-trace MAD', 'Residual MAD', 'Residual MAD + height gate']
COLORS = ['#7e8b9c', '#b5aadd', '#6858d6']
FIGURE_SIZE, TRACE_FIGURE_SIZE, DPI = (14, 3.2), (9, 3), 180
STYLE = {'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False}



def old_sigma(y):
    """Reproduce the preceding whole-trace MAD estimator exactly."""
    y = y[np.isfinite(y)]
    center = np.median(y)
    deviation = np.abs(y - center)
    sigma = 1.4826 * np.median(deviation)
    core = y[deviation <= 3 * sigma]
    if len(core) >= max(5, int(.1 * len(y))):
        sigma = 1.4826 * np.median(np.abs(core - np.median(core)))
    return sigma


def scores(detected, truth):
    """Match detections one-to-one to known synthetic peak times."""
    remaining = list(truth)
    tp = 0
    for detected_time in detected:
        if remaining:
            index = int(np.argmin(np.abs(np.asarray(remaining) - detected_time)))
            if abs(remaining[index] - detected_time) <= MATCH_TOLERANCE:
                tp += 1
                remaining.pop(index)
    return tp / len(detected) if len(detected) else 0., tp / len(truth)


def main():
    """Generate labeled synthetic fixtures, benchmark methods, and export evidence.

    This is a descriptive reproducibility check, not biological validation or
    a statistical significance test. Error bars summarize independent seeds.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    rows = []
    example = None
    for scenario in SCENARIOS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            t = np.arange(0, DURATION, DT)
            y = rng.normal(0, SIGMA, len(t))
            if scenario in ['slow_drift', 'cut_drift']:
                y += DRIFT_AMPLITUDE * np.sin(2 * np.pi * t / DRIFT_PERIOD_SEC)
            truth = np.arange(DENSE_EVENT_START_SEC, DENSE_EVENT_END_SEC, DENSE_EVENT_STEP_SEC) if scenario == 'dense_activity' else EVENT_TIMES
            if scenario == 'changing_noise':
                y[t >= NOISE_CHANGE_SEC] *= LATE_NOISE_FACTOR
            for event in truth:
                y += EVENT_AMPLITUDE * np.exp(-.5 * ((t - event) / EVENT_WIDTH_SEC) ** 2)
            if scenario == 'cut_drift':
                y[(t >= CUT_START_SEC) & (t <= CUT_END_SEC)] = np.nan
            for method in METHODS:
                started = time.perf_counter()
                if method.startswith('Previous'):
                    valid = np.isfinite(y)
                    sigma = old_sigma(y)
                    indices = find_peaks(y[valid], prominence=MULTIPLIER * sigma, distance=max(1, int(round(MIN_DISTANCE_SEC / DT))))[0]
                    detected = t[valid][indices]
                else:
                    stats = estimate_noise(t, y, baseline_window_sec=BASELINE_WINDOW_SEC)
                    sigma = stats['noise_sigma']
                    gate = stats['baseline'] + MULTIPLIER * sigma if method.endswith('height gate') else 0
                    result = detect_peaks(t, y, MULTIPLIER * sigma, min_height=gate, min_distance_sec=MIN_DISTANCE_SEC)
                    detected = t[result['indices']]
                elapsed = time.perf_counter() - started
                precision, recall = scores(detected, truth)
                rows.append(dict(dataset='SYNTHETIC', scenario=scenario, seed=seed, method=method,
                                 precision=precision, recall=recall, f1=2*precision*recall/(precision+recall) if precision+recall else 0,
                                 noise_sigma=sigma, base_noise_sigma=SIGMA, late_noise_sigma=SIGMA * (LATE_NOISE_FACTOR if scenario == 'changing_noise' else 1), seconds=elapsed, detections=len(detected)))
                if scenario == 'cut_drift' and seed == 0 and method == 'Residual MAD + height gate':
                    example = (t, y, stats, detected)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'synthetic_benchmark.csv', index=False)
    summary = frame.groupby(['scenario', 'method'])[['precision', 'recall', 'f1', 'noise_sigma', 'seconds']].agg(['mean','std'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary.to_csv(OUT / 'synthetic_summary.csv')
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=FIGURE_SIZE, constrained_layout=True)
    for ax, scenario in zip(axes, SCENARIOS):
        subset = frame[frame.scenario == scenario]
        for index, method in enumerate(frame.method.unique()):
            group = subset[subset.method == method]
            ax.bar(np.arange(2) + (index - 1) * .23, [group.precision.mean(), group.recall.mean()], width=.23,
                   yerr=[group.precision.std(),group.recall.std()], color=COLORS[index], label=method, capsize=2)
        ax.set(xticks=[0,1], xticklabels=['Precision','Recall'], ylim=(0,1.15), title=scenario.replace('_',' '))
    axes[0].set_ylabel('Synthetic detection score')
    axes[-1].legend(fontsize=6, loc='upper center', bbox_to_anchor=(.5, -.22))
    fig.suptitle(f'Synthetic validation: {len(SEEDS)} seeds per scenario; error bars = SD', fontsize=10)
    for extension in ['png','pdf','svg']:
        fig.savefig(OUT / f'synthetic_benchmark.{extension}', dpi=DPI)
    plt.close(fig)
    t,y,stats,detected = example
    fig, ax = plt.subplots(figsize=TRACE_FIGURE_SIZE, constrained_layout=True)
    ax.plot(t,y,color='#4196ba',lw=.6,label='Synthetic signal')
    ax.fill_between(t,stats['baseline']-stats['noise_sigma'],stats['baseline']+stats['noise_sigma'],color='#8b7bd4',alpha=.25,label='Residual noise: +/- 1 sigma')
    ax.plot(t,stats['baseline'] + MULTIPLIER*stats['noise_sigma'],color='#e59e47',lw=.8,ls='--',label=f'Baseline + {MULTIPLIER:g} sigma (height gate)')
    ax.scatter(detected,np.interp(detected,t,y),s=12,color='#dc694d',label='Detected peaks',zorder=5)
    ax.set(xlabel='Time (s)',ylabel='Synthetic amplitude',title='Synthetic drift and preserved cut: residual-noise estimate')
    ax.legend(fontsize=7, ncol=2)
    for extension in ['png','pdf','svg']:
        fig.savefig(OUT / f'synthetic_trace.{extension}', dpi=DPI)
    print(summary.to_string())
    plt.close('all')
    print(f'Synthetic validation outputs: {OUT}')


if __name__ == '__main__':
    main()

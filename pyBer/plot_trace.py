"""Display-only trace preparation that preserves missing recording intervals."""

from typing import Tuple

import numpy as np


def with_time_gap_breaks(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Insert NaN separators at timestamp gaps without changing input samples.

    Cut artifacts retain the acquisition clock but remove observations. A line
    plot needs explicit separators to avoid drawing across those absent samples.
    Match the preprocessing convention of three times the median sample interval.
    Returned arrays are for display only, never for analysis or stored data.
    """
    x, values = np.asarray(t, float), np.asarray(y, float)
    n = min(x.size, values.size)
    x, values = x[:n], values[:n]
    if n < 3:
        return x, values
    differences = np.diff(x)
    positive = differences[np.isfinite(differences) & (differences > 0)]
    if positive.size == 0:
        return x, values
    interval = float(np.median(positive))
    gaps = np.flatnonzero(differences > max(3.0 * interval, interval + 1e-9))
    if gaps.size == 0:
        return x, values

    # Keep both observed endpoints and add a missing display point between them.
    midpoints = x[gaps] + differences[gaps] * 0.5
    return np.insert(x, gaps + 1, midpoints), np.insert(values, gaps + 1, np.nan)

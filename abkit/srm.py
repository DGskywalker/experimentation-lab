"""Sample ratio mismatch (SRM) detection: is the traffic split what we designed?"""

from __future__ import annotations

import numpy as np
from scipy import stats


def srm_test(observed, expected_ratios=None, threshold: float = 0.001) -> dict:
    """Chi-square goodness-of-fit of observed arm counts against the designed split.

    A p-value below `threshold` (0.001 is the industry convention) means the
    assignment or logging is broken and the experiment's results can't be trusted.
    """
    obs = np.asarray(observed, float)
    ratios = np.ones_like(obs) if expected_ratios is None else np.asarray(expected_ratios, float)
    expected = obs.sum() * ratios / ratios.sum()
    chi2, p = stats.chisquare(obs, expected)
    return {
        "observed": obs.astype(int).tolist(),
        "expected": expected.round(1).tolist(),
        "observed_share": (obs / obs.sum()).round(5).tolist(),
        "chi2": float(chi2),
        "p_value": float(p),
        "srm_detected": bool(p < threshold),
    }

"""Variance reduction (CUPED) and the delta method for ratio metrics."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from abkit.tests import TestResult


def cuped(y, x, theta: float | None = None) -> tuple[np.ndarray, float]:
    """CUPED adjustment: y - theta * (x - mean(x)), x being a pre-experiment covariate.

    theta should be estimated on pooled data from both arms (pass it in), so the
    adjustment is identical across arms and the estimate stays unbiased.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    if theta is None:
        theta = np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)
    return y - theta * (x - x.mean()), float(theta)


def delta_method_ratio(num, den) -> tuple[float, float]:
    """Mean and variance of sum(num)/sum(den) when the unit of randomisation is the user.

    Example: revenue per session where users have many sessions. Treating sessions
    as independent understates variance; the delta method does not.
    """
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    mx, my = den.mean(), num.mean()
    vx, vy = den.var(ddof=1), num.var(ddof=1)
    cxy = np.cov(num, den, ddof=1)[0, 1]
    ratio = my / mx
    var = (vy / mx**2 - 2 * my * cxy / mx**3 + my**2 * vx / mx**4) / n
    return float(ratio), float(var)


def ratio_metric_test(num_c, den_c, num_t, den_t, alpha: float = 0.05, metric: str = "ratio") -> TestResult:
    """z-test on a ratio metric using delta-method variances."""
    r_c, v_c = delta_method_ratio(num_c, den_c)
    r_t, v_t = delta_method_ratio(num_t, den_t)
    diff, se = r_t - r_c, math.sqrt(v_c + v_t)
    z = diff / se
    zc = stats.norm.ppf(1 - alpha / 2)
    return TestResult(
        metric, r_c, r_t, diff, diff / r_c, z, float(2 * stats.norm.sf(abs(z))),
        diff - zc * se, diff + zc * se, "delta-method z-test",
    )

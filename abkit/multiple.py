"""Multiple-comparison corrections."""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def holm(p_values, alpha: float = 0.05) -> dict:
    """Holm–Bonferroni step-down: controls family-wise error rate, uniformly beats Bonferroni."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    adj[order] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    return {"p_adjusted": adj.tolist(), "reject": (adj < alpha).tolist()}


def benjamini_hochberg(p_values, alpha: float = 0.05) -> dict:
    """Benjamini–Hochberg step-up: controls the false discovery rate (use for many segments)."""
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    adj = np.empty(m)
    adj[order] = adj_sorted
    return {"p_adjusted": adj.tolist(), "reject": (adj < alpha).tolist()}


def dunnett_critical_z(n_treatments: int, alpha: float = 0.05, ratio: float = 1.0) -> float:
    """Two-sided critical |z| for comparing k treatments against one shared control.

    With equal arm sizes the test statistics are correlated (rho = 0.5), so this is
    less conservative than Bonferroni. `ratio` = n_treatment / n_control.
    """
    k = n_treatments
    if k == 1:
        return float(stats.norm.ppf(1 - alpha / 2))
    rho = ratio / (1 + ratio)
    cov = np.full((k, k), rho)
    np.fill_diagonal(cov, 1.0)
    mvn = stats.multivariate_normal(mean=np.zeros(k), cov=cov)

    def coverage(c: float) -> float:
        # P(max |Z_i| <= c) via inclusion–exclusion on the orthant is awkward; use the
        # box probability directly from the multivariate normal CDF.
        lo, hi = np.full(k, -c), np.full(k, c)
        return mvn.cdf(hi, lower_limit=lo) - (1 - alpha)

    return float(optimize.brentq(coverage, 1.0, 6.0))

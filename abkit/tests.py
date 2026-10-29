"""Frequentist significance tests and confidence intervals for two-arm experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    metric: str
    control: float
    treatment: float
    diff: float
    rel_lift: float
    stat: float
    p_value: float
    ci_low: float
    ci_high: float
    method: str

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def to_dict(self) -> dict:
        return asdict(self)


def wilson_ci(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a single proportion."""
    z = stats.norm.ppf(1 - alpha / 2)
    p = x / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def two_proportion_ztest(
    x_control: int,
    n_control: int,
    x_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
    metric: str = "conversion",
) -> TestResult:
    """Pooled z-test for H0: p_t = p_c, with an unpooled Wald CI on p_t - p_c."""
    p1, p2 = x_control / n_control, x_treatment / n_treatment
    p_pool = (x_control + x_treatment) / (n_control + n_treatment)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    z = (p2 - p1) / se_pool
    p_value = 2 * stats.norm.sf(abs(z))
    se = math.sqrt(p1 * (1 - p1) / n_control + p2 * (1 - p2) / n_treatment)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p2 - p1
    return TestResult(
        metric, p1, p2, diff, diff / p1, z, p_value,
        diff - z_crit * se, diff + z_crit * se, "two-proportion z-test",
    )


def welch_ttest(control, treatment, alpha: float = 0.05, metric: str = "mean") -> TestResult:
    """Welch's unequal-variance t-test with a Welch–Satterthwaite CI on the difference."""
    a, b = np.asarray(control, float), np.asarray(treatment, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    diff = b.mean() - a.mean()
    t = diff / se
    p_value = 2 * stats.t.sf(abs(t), df)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return TestResult(
        metric, a.mean(), b.mean(), diff, diff / a.mean(), t, p_value,
        diff - t_crit * se, diff + t_crit * se, "Welch t-test",
    )


def mann_whitney(control, treatment, metric: str = "distribution") -> TestResult:
    """Mann–Whitney U test. Reports medians, with the rank-biserial correlation as `stat`."""
    a, b = np.asarray(control, float), np.asarray(treatment, float)
    res = stats.mannwhitneyu(b, a, alternative="two-sided")
    # rank-biserial correlation: P(B > A) - P(B < A)
    rbc = 2 * res.statistic / (len(a) * len(b)) - 1
    ma, mb = float(np.median(a)), float(np.median(b))
    return TestResult(
        metric, ma, mb, mb - ma, (mb - ma) / ma if ma else float("nan"),
        rbc, float(res.pvalue), float("nan"), float("nan"), "Mann-Whitney U (medians)",
    )


def bootstrap_diff_ci(
    control,
    treatment,
    stat: Callable[[np.ndarray], np.ndarray] | None = None,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    chunk: int = 250,
) -> dict:
    """Percentile bootstrap CI for stat(treatment) - stat(control).

    `stat` must reduce along axis=1 (default: mean). Resampling is done in chunks
    so 10k resamples of ~50k users fit comfortably in memory.
    Returns the CI plus the bootstrap distributions for plotting.
    """
    stat = stat or (lambda m: m.mean(axis=1))
    rng = np.random.default_rng(seed)
    a, b = np.asarray(control, float), np.asarray(treatment, float)
    boot_a, boot_b = np.empty(n_boot), np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        boot_a[start:start + size] = stat(a[rng.integers(0, len(a), (size, len(a)))])
        boot_b[start:start + size] = stat(b[rng.integers(0, len(b), (size, len(b)))])
    diffs = boot_b - boot_a
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_treatment_better": float((diffs > 0).mean()),
        "boot_control": boot_a,
        "boot_treatment": boot_b,
        "boot_diff": diffs,
    }

"""Sample size, power and minimum detectable effect (MDE) calculations."""

from __future__ import annotations

import math

from scipy import optimize, stats


def _z(alpha: float, two_sided: bool) -> float:
    return stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)


def sample_size_proportions(
    p_control: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    relative: bool = False,
    ratio: float = 1.0,
    two_sided: bool = True,
) -> tuple[int, int]:
    """Users needed per arm to detect a change in a conversion rate.

    `mde` is absolute (0.02 = +2pp) unless `relative=True` (0.05 = +5%).
    `ratio` is n_treatment / n_control. Returns (n_control, n_treatment).
    """
    p2 = p_control * (1 + mde) if relative else p_control + mde
    if not 0 < p2 < 1:
        raise ValueError("treatment rate must be in (0, 1)")
    d = abs(p2 - p_control)
    k = ratio
    p_bar = (p_control + k * p2) / (1 + k)
    z_a, z_b = _z(alpha, two_sided), stats.norm.ppf(power)
    num = (
        z_a * math.sqrt(p_bar * (1 - p_bar) * (1 + 1 / k))
        + z_b * math.sqrt(p_control * (1 - p_control) + p2 * (1 - p2) / k)
    ) ** 2
    n1 = math.ceil(num / d**2)
    return n1, math.ceil(n1 * k)


def sample_size_means(
    sd: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    two_sided: bool = True,
) -> tuple[int, int]:
    """Users needed per arm to detect an absolute shift `mde` in a mean."""
    z_a, z_b = _z(alpha, two_sided), stats.norm.ppf(power)
    n1 = math.ceil((z_a + z_b) ** 2 * sd**2 * (1 + 1 / ratio) / mde**2)
    return n1, math.ceil(n1 * ratio)


def power_proportions(
    p_control: float,
    p_treatment: float,
    n_control: int,
    n_treatment: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Power of a pooled two-proportion z-test for a given true effect."""
    d = abs(p_treatment - p_control)
    p_bar = (p_control * n_control + p_treatment * n_treatment) / (n_control + n_treatment)
    se0 = math.sqrt(p_bar * (1 - p_bar) * (1 / n_control + 1 / n_treatment))
    se1 = math.sqrt(
        p_control * (1 - p_control) / n_control + p_treatment * (1 - p_treatment) / n_treatment
    )
    z_a = _z(alpha, two_sided)
    pw = stats.norm.cdf((d - z_a * se0) / se1)
    if two_sided:
        pw += stats.norm.cdf((-d - z_a * se0) / se1)
    return float(pw)


def mde_proportions(
    p_control: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> float:
    """Smallest absolute lift detectable with the given power at n users per arm."""
    def gap(d: float) -> float:
        return power_proportions(p_control, p_control + d, n_per_arm, n_per_arm, alpha, two_sided) - power

    return float(optimize.brentq(gap, 1e-7, 1 - p_control - 1e-7))

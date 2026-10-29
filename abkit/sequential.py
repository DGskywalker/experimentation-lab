"""Sequential testing: valid ways to look at results before the experiment ends."""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def _spend_obf(t: np.ndarray, alpha: float) -> np.ndarray:
    """Cumulative two-sided alpha spent by information fraction t.

    Lan–DeMets O'Brien–Fleming-type function applied to each tail at alpha/2
    (the convention used by gsDesign/ldbounds/rpact), then summed over both tails.
    """
    one_side = alpha / 2
    # 2 - 2*cdf(x) written as 4*sf(x) so tiny early spends don't underflow to zero
    return 2 * 2 * stats.norm.sf(stats.norm.isf(one_side / 2) / np.sqrt(t))


def obrien_fleming_boundaries(
    n_looks: int,
    alpha: float = 0.05,
    info_fractions=None,
    grid_points: int = 4001,
) -> dict:
    """Two-sided |z| boundaries for K interim looks using O'Brien–Fleming alpha spending.

    Under H0 the score S(t) = Z(t)*sqrt(t) is a Brownian motion. We carry the
    density of S over the "still running" region from look to look by numerical
    convolution (Armitage–McPherson–Rowe) and solve for each boundary so the
    probability of first crossing at that look equals the alpha spent there.
    """
    t = np.linspace(1 / n_looks, 1, n_looks) if info_fractions is None else np.asarray(info_fractions, float)
    cum_alpha = _spend_obf(t, alpha)
    spend = np.diff(np.concatenate([[0.0], cum_alpha]))

    bounds: list[float] = []
    grid = dens = None  # density of S on the continuation region of the previous look
    for k, tk in enumerate(t):
        dt = tk - (t[k - 1] if k else 0.0)
        sd = np.sqrt(dt)

        def cross_prob(b: float) -> float:
            c = b * np.sqrt(tk)
            if k == 0:
                return 2 * stats.norm.sf(c / sd)
            tail = stats.norm.sf((c - grid) / sd) + stats.norm.cdf((-c - grid) / sd)
            return float(np.trapezoid(dens * tail, grid))

        # solve on the log scale: early-look spends can be ~1e-20
        with np.errstate(divide="ignore"):
            b = float(optimize.brentq(lambda x: np.log(cross_prob(x)) - np.log(spend[k]), 0.5, 30.0))
        bounds.append(b)
        c = b * np.sqrt(tk)
        new_grid = np.linspace(-c, c, grid_points)
        if k == 0:
            new_dens = stats.norm.pdf(new_grid / sd) / sd
        else:
            kernel = stats.norm.pdf((new_grid[:, None] - grid[None, :]) / sd) / sd
            new_dens = np.trapezoid(kernel * dens[None, :], grid, axis=1)
        grid, dens = new_grid, new_dens

    return {
        "info_fractions": t.tolist(),
        "z_boundaries": bounds,
        "nominal_p": [float(2 * stats.norm.sf(b)) for b in bounds],
        "cumulative_alpha": cum_alpha.tolist(),
    }


def msprt(estimate: float, variance: float, tau: float, alpha: float = 0.05) -> dict:
    """Mixture sequential probability ratio test (Johari et al., 2017).

    `estimate` is the current treatment effect estimate with sampling `variance`;
    `tau` is the std of the normal mixing prior over the effect (roughly, the
    effect size you care about). The likelihood ratio can be checked after every
    observation; rejecting when it exceeds 1/alpha keeps the type-I error <= alpha.
    The always-valid p-value is min(1, 1/max_so_far(LR)); callers track the running max.
    """
    v, t2 = variance, tau**2
    lr = np.sqrt(v / (v + t2)) * np.exp(estimate**2 * t2 / (2 * v * (v + t2)))
    return {"likelihood_ratio": float(lr), "reject": bool(lr >= 1 / alpha), "p_value": float(min(1.0, 1 / lr))}

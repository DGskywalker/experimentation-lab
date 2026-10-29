"""Bayesian analysis of conversion experiments with a Beta–Binomial model."""

from __future__ import annotations

import numpy as np


def beta_binomial_ab(
    x_control: int,
    n_control: int,
    x_treatment: int,
    n_treatment: int,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    draws: int = 200_000,
    cred: float = 0.95,
    seed: int = 0,
) -> dict:
    """Posterior comparison of two conversion rates.

    expected_loss_* is the conversion rate you expect to give up by shipping that
    arm if it is in fact the worse one — the quantity a decision-maker cares about.
    """
    rng = np.random.default_rng(seed)
    pc = rng.beta(prior_a + x_control, prior_b + n_control - x_control, draws)
    pt = rng.beta(prior_a + x_treatment, prior_b + n_treatment - x_treatment, draws)
    diff = pt - pc
    q = (1 - cred) / 2
    return {
        "prob_treatment_better": float((diff > 0).mean()),
        "expected_loss_treatment": float(np.maximum(pc - pt, 0).mean()),
        "expected_loss_control": float(np.maximum(pt - pc, 0).mean()),
        "diff_mean": float(diff.mean()),
        "diff_cred_low": float(np.quantile(diff, q)),
        "diff_cred_high": float(np.quantile(diff, 1 - q)),
        "rel_lift_mean": float((diff / pc).mean()),
        "posterior_control": pc,
        "posterior_treatment": pt,
    }

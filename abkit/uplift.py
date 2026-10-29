"""Uplift modelling: who is *persuaded* by the treatment, not just who converts.

Meta-learners (S, T, X) turn any scikit-learn classifier into an estimator of the
conditional average treatment effect tau(x) = P(y=1 | x, treated) - P(y=1 | x, control).
They need randomised data: treatment must be independent of x.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.base import clone


def _p1(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


class SLearner:
    """One model with the treatment flag as a feature; uplift = f(x, 1) - f(x, 0)."""

    def __init__(self, model):
        self.model = clone(model)

    def fit(self, X, t, y):
        self.model.fit(np.column_stack([X, t]), y)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X)
        ones, zeros = np.ones(len(X)), np.zeros(len(X))
        return _p1(self.model, np.column_stack([X, ones])) - _p1(self.model, np.column_stack([X, zeros]))


class TLearner:
    """Separate outcome models for treated and control; uplift = mu1(x) - mu0(x)."""

    def __init__(self, model):
        self.m0, self.m1 = clone(model), clone(model)

    def fit(self, X, t, y):
        X, t, y = np.asarray(X), np.asarray(t), np.asarray(y)
        self.m0.fit(X[t == 0], y[t == 0])
        self.m1.fit(X[t == 1], y[t == 1])
        return self

    def predict(self, X) -> np.ndarray:
        return _p1(self.m1, X) - _p1(self.m0, X)


class XLearner:
    """Künzel et al. (2019). Imputes individual effects, then regresses them.

    Stage 1: outcome models mu0, mu1 (as in the T-learner).
    Stage 2: D1 = y - mu0(x) on treated, D0 = mu1(x) - y on control; fit regressors tau1, tau0.
    Combine with the propensity g: tau(x) = g * tau0(x) + (1 - g) * tau1(x).
    Shines when arms are unbalanced (e.g. 85/15), because each arm borrows the other's model.
    """

    def __init__(self, classifier, regressor):
        self.m0, self.m1 = clone(classifier), clone(classifier)
        self.tau0, self.tau1 = clone(regressor), clone(regressor)

    def fit(self, X, t, y):
        X, t, y = np.asarray(X), np.asarray(t), np.asarray(y)
        self.m0.fit(X[t == 0], y[t == 0])
        self.m1.fit(X[t == 1], y[t == 1])
        d1 = y[t == 1] - _p1(self.m0, X[t == 1])
        d0 = _p1(self.m1, X[t == 0]) - y[t == 0]
        self.tau1.fit(X[t == 1], d1)
        self.tau0.fit(X[t == 0], d0)
        self.g = t.mean()  # randomised experiment: constant propensity
        return self

    def predict(self, X) -> np.ndarray:
        return self.g * self.tau0.predict(X) + (1 - self.g) * self.tau1.predict(X)


# ------------------------------------------------------------------ evaluation

def qini_curve(y, t, uplift, n_points: int = 101) -> dict:
    """Incremental outcomes captured when treating the top fraction of users ranked by `uplift`.

    Q(phi) = Y_t(phi) - Y_c(phi) * N_t(phi) / N_c(phi): treated conversions in the top
    phi, minus what those treated users would have done anyway (control rate in the same
    slice). Q(1) is the total incremental effect of treating everyone.
    """
    y, t, u = np.asarray(y, float), np.asarray(t), np.asarray(uplift)
    order = np.argsort(-u, kind="stable")
    y, t = y[order], t[order]
    yt, yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(nc > 0, yt - yc * nt / nc, 0.0)
    idx = np.unique(np.linspace(0, len(y) - 1, n_points).astype(int))
    frac = np.concatenate([[0.0], (idx + 1) / len(y)])
    q = np.concatenate([[0.0], q[idx]])
    return {"fraction": frac, "qini": q, "random": frac * q[-1]}


def qini_coefficient(y, t, uplift) -> float:
    """Area between the model's Qini curve and the random-targeting line, normalised by |Q(1)|.

    0 means no better than random; higher is better. It can be negative.
    """
    c = qini_curve(y, t, uplift)
    area = np.trapezoid(c["qini"] - c["random"], c["fraction"])
    return float(area / abs(c["qini"][-1])) if c["qini"][-1] else float("nan")


def uplift_at_k(y, t, uplift, k: float = 0.3) -> float:
    """Treated rate minus control rate among the top-k fraction ranked by predicted uplift."""
    y, t, u = np.asarray(y, float), np.asarray(t), np.asarray(uplift)
    top = np.argsort(-u, kind="stable")[: max(1, int(round(k * len(u))))]
    yt, tt = y[top], t[top]
    return float(yt[tt == 1].mean() - yt[tt == 0].mean())


def share_of_incremental(y, t, uplift, k: float = 0.3) -> float:
    """Fraction of the total incremental effect captured by treating only the top-k users."""
    c = qini_curve(y, t, uplift, n_points=1001)
    return float(np.interp(k, c["fraction"], c["qini"]) / c["qini"][-1])


def bootstrap_metric(metric: Callable, y, t, uplift, n_boot: int = 300, seed: int = 0, **kw) -> tuple[float, float]:
    """95% percentile interval for an uplift metric, resampling users."""
    rng = np.random.default_rng(seed)
    y, t, u = map(np.asarray, (y, t, uplift))
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(metric(y[i], t[i], u[i], **kw))
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(lo), float(hi)

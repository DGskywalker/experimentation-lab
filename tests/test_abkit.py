"""abkit is checked against statsmodels/SciPy and published reference values."""

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

import abkit


# ---------- power ----------

def test_sample_size_matches_fleiss_reference():
    # Fleiss (no continuity correction): p1=0.10, p2=0.12, alpha=0.05, power=0.8 -> 3841/arm
    n1, n2 = abkit.sample_size_proportions(0.10, 0.02)
    assert n1 == n2
    assert abs(n1 - 3841) <= 2


def test_sample_size_gives_target_power():
    n1, n2 = abkit.sample_size_proportions(0.35, 0.05, relative=True, power=0.9)
    pw = abkit.power_proportions(0.35, 0.35 * 1.05, n1, n2)
    assert pw == pytest.approx(0.9, abs=0.002)


def test_unequal_allocation_needs_more_users_in_total():
    a = sum(abkit.sample_size_proportions(0.2, 0.01))
    b = sum(abkit.sample_size_proportions(0.2, 0.01, ratio=3))
    assert b > a


def test_mde_round_trip():
    n, _ = abkit.sample_size_proportions(0.19, 0.01)
    assert abkit.mde_proportions(0.19, n) == pytest.approx(0.01, rel=0.01)


def test_sample_size_means():
    # (1.96 + 0.8416)^2 * 2 * 10^2 / 2^2 = 392.4 -> 393
    assert abkit.sample_size_means(sd=10, mde=2)[0] == 393


# ---------- tests ----------

def test_ztest_matches_statsmodels():
    r = abkit.two_proportion_ztest(200, 1000, 240, 1000)
    z, p = proportions_ztest([240, 200], [1000, 1000])
    assert r.stat == pytest.approx(z)
    assert r.p_value == pytest.approx(p)
    assert r.ci_low < r.diff < r.ci_high


def test_welch_matches_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 300), rng.normal(0.2, 2, 500)
    r = abkit.welch_ttest(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert r.stat == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)
    ci = ref.confidence_interval()
    assert (r.ci_low, r.ci_high) == pytest.approx((ci.low, ci.high))


def test_wilson_matches_statsmodels():
    assert abkit.wilson_ci(37, 120) == pytest.approx(proportion_confint(37, 120, method="wilson"))


def test_mann_whitney_detects_shift():
    rng = np.random.default_rng(2)
    r = abkit.mann_whitney(rng.exponential(1, 2000), rng.exponential(1.3, 2000))
    assert r.p_value < 0.001 and r.stat > 0


def test_bootstrap_ci_close_to_analytic():
    rng = np.random.default_rng(3)
    a, b = rng.binomial(1, 0.30, 5000), rng.binomial(1, 0.33, 5000)
    boot = abkit.bootstrap_diff_ci(a, b, n_boot=4000)
    z = abkit.two_proportion_ztest(a.sum(), len(a), b.sum(), len(b))
    assert boot["ci_low"] == pytest.approx(z.ci_low, abs=0.003)
    assert boot["ci_high"] == pytest.approx(z.ci_high, abs=0.003)


# ---------- SRM ----------

def test_srm():
    assert not abkit.srm_test([50_000, 50_100])["srm_detected"]
    assert abkit.srm_test([51_000, 49_000])["srm_detected"]
    assert not abkit.srm_test([90_000, 10_050], [0.9, 0.1])["srm_detected"]


# ---------- variance ----------

def test_cuped_reduces_variance_by_rho_squared():
    rng = np.random.default_rng(4)
    x = rng.normal(10, 3, 50_000)
    y = 0.8 * x + rng.normal(0, 2, 50_000)
    adj, _ = abkit.cuped(y, x)
    rho = np.corrcoef(x, y)[0, 1]
    assert adj.var() / y.var() == pytest.approx(1 - rho**2, rel=0.01)
    assert adj.mean() == pytest.approx(y.mean())


def test_delta_method_reduces_to_mean_when_denominator_constant():
    rng = np.random.default_rng(5)
    num = rng.normal(5, 2, 1000)
    ratio, var = abkit.delta_method_ratio(num, np.ones(1000))
    assert ratio == pytest.approx(num.mean())
    assert var == pytest.approx(num.var(ddof=1) / 1000)


def test_delta_method_matches_bootstrap():
    rng = np.random.default_rng(6)
    den = rng.poisson(3, 4000) + 1
    num = rng.binomial(den, 0.2) * rng.gamma(2, 10, 4000)
    _, var = abkit.delta_method_ratio(num, den)
    idx = rng.integers(0, 4000, (2000, 4000))
    boot = num[idx].sum(1) / den[idx].sum(1)
    assert np.sqrt(var) == pytest.approx(boot.std(), rel=0.06)


# ---------- multiple comparisons ----------

P = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]


def test_holm_matches_statsmodels():
    assert abkit.holm(P)["p_adjusted"] == pytest.approx(multipletests(P, method="holm")[1].tolist())


def test_bh_matches_statsmodels():
    assert abkit.benjamini_hochberg(P)["p_adjusted"] == pytest.approx(
        multipletests(P, method="fdr_bh")[1].tolist()
    )


def test_dunnett_critical_values():
    assert abkit.dunnett_critical_z(1) == pytest.approx(1.95996, abs=1e-4)
    # Dunnett two-sided, infinite df, k=2: 2.212 ; k=3: 2.349
    assert abkit.dunnett_critical_z(2) == pytest.approx(2.212, abs=0.005)
    assert abkit.dunnett_critical_z(3) == pytest.approx(2.349, abs=0.005)


# ---------- sequential ----------

def test_obf_boundaries_match_reference():
    # Lan–DeMets O'Brien–Fleming spending, 5 equally spaced looks, alpha=0.05 (two-sided)
    ref = [4.877, 3.357, 2.680, 2.290, 2.031]
    b = abkit.obrien_fleming_boundaries(5)["z_boundaries"]
    assert b == pytest.approx(ref, abs=0.002)


def test_msprt():
    assert abkit.msprt(0.0, 1e-4, 0.01)["likelihood_ratio"] < 1
    assert abkit.msprt(0.05, 1e-4, 0.01)["reject"]


# ---------- bayes ----------

def test_bayes_symmetric_and_directional():
    same = abkit.beta_binomial_ab(500, 5000, 500, 5000)
    assert same["prob_treatment_better"] == pytest.approx(0.5, abs=0.01)
    better = abkit.beta_binomial_ab(500, 5000, 600, 5000)
    assert better["prob_treatment_better"] > 0.99
    assert better["expected_loss_treatment"] < better["expected_loss_control"]


# ---------- assignment ----------

def test_assignment_is_sticky_balanced_and_salted():
    arms = ("A", "B", "C")
    ids = [f"user-{i}" for i in range(30_000)]
    first = [abkit.assign(u, "booking_form_v1", arms) for u in ids]
    assert first == [abkit.assign(u, "booking_form_v1", arms) for u in ids]
    counts = [first.count(a) for a in arms]
    assert not abkit.srm_test(counts)["srm_detected"]
    other = [abkit.assign(u, "another_test", arms) for u in ids]
    agreement = np.mean([a == b for a, b in zip(first, other)])
    assert agreement == pytest.approx(1 / 3, abs=0.02)


def test_assignment_respects_weights():
    got = [abkit.assign(str(i), "ramp", ("control", "treatment"), weights=[0.9, 0.1]) for i in range(20_000)]
    assert got.count("treatment") / len(got) == pytest.approx(0.1, abs=0.01)


# ---------- uplift ----------

def _uplift_data(n=40_000, seed=0):
    """Randomised data where only users with x0 > 0 respond to treatment."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    t = rng.integers(0, 2, n)
    tau = np.where(X[:, 0] > 0, 0.15, 0.0)
    base = 0.10 + 0.05 * (X[:, 1] > 0)
    y = (rng.random(n) < base + t * tau).astype(int)
    return X, t, y, tau


def test_qini_endpoint_is_total_incremental_effect():
    X, t, y, tau = _uplift_data()
    c = abkit.uplift.qini_curve(y, t, tau)
    expected = y[t == 1].sum() - y[t == 0].sum() * (t == 1).sum() / (t == 0).sum()
    assert c["qini"][-1] == pytest.approx(expected)


def test_qini_oracle_beats_random():
    X, t, y, tau = _uplift_data()
    rng = np.random.default_rng(1)
    assert abkit.uplift.qini_coefficient(y, t, tau) > 0.15
    assert abs(abkit.uplift.qini_coefficient(y, t, rng.random(len(y)))) < 0.08
    # treating the responsive half captures ~all of the incremental effect
    assert abkit.uplift.share_of_incremental(y, t, tau, k=0.5) == pytest.approx(1.0, abs=0.12)


@pytest.mark.parametrize("learner", ["s", "t", "x"])
def test_meta_learners_recover_heterogeneity(learner):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    X, t, y, tau = _uplift_data(seed=2)
    Xte, tte, yte, tau_te = _uplift_data(n=20_000, seed=3)
    clf = HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=0)
    model = {
        "s": lambda: abkit.uplift.SLearner(clf),
        "t": lambda: abkit.uplift.TLearner(clf),
        "x": lambda: abkit.uplift.XLearner(clf, HistGradientBoostingRegressor(max_iter=100, max_depth=3, random_state=0)),
    }[learner]().fit(X, t, y)
    pred = model.predict(Xte)
    assert pred[tau_te > 0].mean() - pred[tau_te == 0].mean() > 0.08
    assert abkit.uplift.uplift_at_k(yte, tte, pred, k=0.3) > 0.10

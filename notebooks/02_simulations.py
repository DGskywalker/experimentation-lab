# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 · Simulations: do our methods keep their promises?
#
# A test that claims α = 5% should be wrong 5% of the time *when there is no effect*. This
# notebook checks that claim for `abkit`, then shows three things every experimenter
# should know:
#
# 1. **A/A tests**: the pipeline's false positive rate is calibrated.
# 2. **Peeking**: checking results every day inflates false positives, and two ways to fix it.
# 3. **CUPED**: using pre-experiment data to cut variance, and with it the test's duration.

# %%
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import abkit
from abkit import viz

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG = ROOT / "reports" / "figures"
viz.use_style()
rng = np.random.default_rng(2026)
out = {}

# %% [markdown]
# ## 1 · A/A test: 10,000 experiments with no real difference

# %%
N_SIMS, N, P = 10_000, 5_000, 0.20
x_a, x_b = rng.binomial(N, P, N_SIMS), rng.binomial(N, P, N_SIMS)
pvals = np.array([abkit.two_proportion_ztest(a, N, b, N).p_value for a, b in zip(x_a, x_b)])
fpr = (pvals < 0.05).mean()
ks = stats.kstest(pvals, "uniform")
out["aa"] = {"n_sims": N_SIMS, "false_positive_rate": fpr, "ks_uniform_p": ks.pvalue}
print(f"False positive rate: {fpr:.2%}   (expected 5.00% ± {1.96 * np.sqrt(.05 * .95 / N_SIMS):.2%})")
print(f"KS test that p-values are Uniform(0,1): p = {ks.pvalue:.3f}")

# %%
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(pvals, bins=20, range=(0, 1), color=viz.CONTROL, alpha=0.85, edgecolor=viz.SURFACE, linewidth=2)
ax.axhline(N_SIMS / 20, color=viz.INK_2, lw=1, ls=(0, (4, 3)))
ax.text(0.99, N_SIMS / 20 * 1.04, "expected if calibrated", ha="right", va="bottom", color=viz.INK_2, fontsize=9)
ax.set(xlabel="p-value", ylabel="Experiments",
       title=f"A/A tests: p-values are uniform, false positive rate = {fpr:.1%}")
ax.grid(axis="x", visible=False)
viz.save(fig, FIG / "sim_aa_pvalues.png")

# %% [markdown]
# ![](../reports/figures/sim_aa_pvalues.png)
#
# ## 2 · The peeking problem
# A team runs a 20-day test and checks the dashboard every day, stopping the first time
# p < 0.05. Below are three ways to run it, each simulated under the null hypothesis (no
# effect) and under a real +5% relative lift:
#
# * **Naive peeking**: stop on the first daily p < 0.05.
# * **O'Brien–Fleming (group sequential)**: pre-planned looks with strict early boundaries.
# * **mSPRT (always-valid)**: look whenever you like; the likelihood-ratio threshold 1/α holds at every look.

# %%
DAYS, PER_DAY, BASE, SIMS = 20, 1_000, 0.20, 4_000
obf = abkit.obrien_fleming_boundaries(DAYS)
bounds = np.array(obf["z_boundaries"])
TAU = 0.01  # mixing std: the size of effect we care about (1pp)


def run(lift: float) -> dict:
    n = PER_DAY * np.arange(1, DAYS + 1)
    ca = rng.binomial(PER_DAY, BASE, (SIMS, DAYS)).cumsum(1)
    cb = rng.binomial(PER_DAY, BASE * (1 + lift), (SIMS, DAYS)).cumsum(1)
    pa, pb = ca / n, cb / n
    pool = (ca + cb) / (2 * n)
    z = (pb - pa) / np.sqrt(pool * (1 - pool) * 2 / n)
    var = (pa * (1 - pa) + pb * (1 - pb)) / n
    lr = np.sqrt(var / (var + TAU**2)) * np.exp((pb - pa) ** 2 * TAU**2 / (2 * var * (var + TAU**2)))

    def summarise(hit: np.ndarray) -> dict:
        any_hit = hit.any(1)
        first = np.where(any_hit, hit.argmax(1) + 1, DAYS)
        return {"reject_rate": float(any_hit.mean()), "avg_days": float(first.mean())}

    return {
        "Fixed horizon (look once, day 20)": {**summarise(np.abs(z[:, -1:]) > 1.96), "avg_days": float(DAYS)},
        "Naive daily peeking": summarise(np.abs(z) > 1.96),
        "O'Brien–Fleming, 20 looks": summarise(np.abs(z) > bounds),
        "mSPRT, look any time": summarise(lr >= 1 / 0.05),
    }


null, alt = run(0.0), run(0.05)
peek = pd.DataFrame({
    "False positive rate (no effect)": {k: v["reject_rate"] for k, v in null.items()},
    "Power (+5% lift)": {k: v["reject_rate"] for k, v in alt.items()},
    "Avg. days to decision (+5% lift)": {k: v["avg_days"] for k, v in alt.items()},
})
out["peeking"] = peek.to_dict()
peek.style.format({"False positive rate (no effect)": "{:.1%}", "Power (+5% lift)": "{:.1%}",
                   "Avg. days to decision (+5% lift)": "{:.1f}"})

# %%
# How false positives accumulate with each extra look
n = PER_DAY * np.arange(1, DAYS + 1)
ca = rng.binomial(PER_DAY, BASE, (SIMS, DAYS)).cumsum(1)
cb = rng.binomial(PER_DAY, BASE, (SIMS, DAYS)).cumsum(1)
pool = (ca + cb) / (2 * n)
z = (cb / n - ca / n) / np.sqrt(pool * (1 - pool) * 2 / n)
naive_cum = np.maximum.accumulate(np.abs(z) > 1.96, axis=1).mean(0)
obf_cum = np.maximum.accumulate(np.abs(z) > bounds, axis=1).mean(0)

fig, ax = plt.subplots(figsize=(7, 3.6))
days = np.arange(1, DAYS + 1)
ax.plot(days, naive_cum * 100, color=viz.TREATMENT, marker="o", ms=4)
ax.plot(days, obf_cum * 100, color=viz.CONTROL, marker="o", ms=4)
ax.axhline(5, color=viz.INK_2, lw=1, ls=(0, (4, 3)))
ax.text(DAYS + 0.3, naive_cum[-1] * 100, f"Naive peeking  {naive_cum[-1]:.0%}", va="center", color=viz.INK, fontsize=9)
ax.text(DAYS + 0.3, obf_cum[-1] * 100 - 1.2, f"O'Brien–Fleming  {obf_cum[-1]:.1%}", va="center", color=viz.INK, fontsize=9)
ax.text(6, 5.6, "promised α = 5%", color=viz.INK_2, fontsize=9)
ax.set(xlabel="Day of experiment (one look per day)", ylabel="Cumulative false positive rate (%)",
       xlim=(0.5, DAYS + 6), ylim=(0, max(30, naive_cum[-1] * 100 + 3)),
       title="Checking a null A/B test daily makes you 'win' ~5× too often")
ax.set_xticks([1, 5, 10, 15, 20])
viz.save(fig, FIG / "sim_peeking.png")

# %% [markdown]
# ![](../reports/figures/sim_peeking.png)
#
# **Takeaways**
# * **Naive daily peeking produces a ~24% false positive rate**, about 5× the promised 5%.
#   Its "extra power" is just those false positives.
# * **O'Brien–Fleming** keeps α ≤ 5% with almost the same power as a fixed horizon. Its
#   early boundaries are very strict (|z| > 9.9 on day 1, 2.12 on day 20), so only very
#   large effects stop early.
# * **mSPRT** lets you look at *any* moment and still keeps α ≤ 5%. Here it's very
#   conservative (≈1% false positives), and it gives up a lot of power for that
#   flexibility. It suits monitoring for harm more than detecting small wins.
# * **Rule for the booking-form test:** use pre-planned weekly O'Brien–Fleming looks for
#   the primary metric, and mSPRT for continuous guardrail monitoring.
#
# ## 3 · CUPED: variance reduction with pre-experiment data
# Users who were active before the experiment tend to be active during it. If you
# subtract the part of the metric that pre-period behaviour predicts, the noise drops
# and the treatment effect doesn't change.

# %%
def cuped_experiment(n: int, effect: float, rho_strength: float = 1.0):
    pre = rng.gamma(2.0, 2.0, 2 * n)                       # sessions in the 2 weeks before
    arm = np.repeat([0, 1], n)
    y = rho_strength * 0.9 * pre + rng.gamma(2.0, 1.5, 2 * n) + effect * arm  # sessions during test
    return pre, y, arm


pre, y, arm = cuped_experiment(20_000, effect=0.15)
theta = np.cov(pre, y)[0, 1] / pre.var(ddof=1)            # pooled across arms
y_adj, _ = abkit.cuped(y, pre, theta=theta)
raw = abkit.welch_ttest(y[arm == 0], y[arm == 1], metric="sessions")
adj = abkit.welch_ttest(y_adj[arm == 0], y_adj[arm == 1], metric="sessions (CUPED)")
rho = np.corrcoef(pre, y)[0, 1]
var_red = 1 - y_adj.var() / y.var()
cuped_tbl = pd.DataFrame([raw.to_dict(), adj.to_dict()]).set_index("metric")[["diff", "ci_low", "ci_high", "p_value"]]
cuped_tbl["ci_width"] = cuped_tbl.ci_high - cuped_tbl.ci_low
print(f"corr(pre, during) = {rho:.2f} -> variance reduction {var_red:.0%} (theory: rho^2 = {rho**2:.0%})")
cuped_tbl.round(4)

# %%
# Repeat 2,000 times: CUPED stays unbiased and is far more precise
est_raw, est_adj = [], []
for _ in range(2_000):
    pre, y, arm = cuped_experiment(2_000, effect=0.15)
    th = np.cov(pre, y)[0, 1] / pre.var(ddof=1)
    ya = y - th * (pre - pre.mean())
    est_raw.append(y[arm == 1].mean() - y[arm == 0].mean())
    est_adj.append(ya[arm == 1].mean() - ya[arm == 0].mean())
est_raw, est_adj = np.array(est_raw), np.array(est_adj)
sd_y = y.std()
n_raw = abkit.sample_size_means(sd=sd_y, mde=0.15)[0]
n_adj = abkit.sample_size_means(sd=sd_y * np.sqrt(1 - var_red), mde=0.15)[0]
out["cuped"] = {
    "rho": rho, "variance_reduction": var_red,
    "mean_estimate_raw": est_raw.mean(), "mean_estimate_cuped": est_adj.mean(),
    "sd_raw": est_raw.std(), "sd_cuped": est_adj.std(),
    "n_per_arm_raw": n_raw, "n_per_arm_cuped": n_adj,
}
pd.Series(out["cuped"]).round(4)

# %%
fig, ax = plt.subplots(figsize=(7, 3.4))
bins = np.linspace(min(est_raw.min(), est_adj.min()), max(est_raw.max(), est_adj.max()), 60)
ax.hist(est_raw, bins=bins, color=viz.NEUTRAL, alpha=0.8, label="Raw difference in means")
ax.hist(est_adj, bins=bins, color=viz.CONTROL, alpha=0.8, label="CUPED-adjusted")
ax.axvline(0.15, color=viz.INK, lw=1)
ax.text(0.153, ax.get_ylim()[1] * 0.92, "true effect", color=viz.INK, fontsize=9)
ax.set(xlabel="Estimated treatment effect (sessions/user) across 2,000 simulated experiments",
       ylabel="Experiments",
       title=f"CUPED: same answer, {1 - est_adj.std() / est_raw.std():.0%} less noise, "
             f"{1 - n_adj / n_raw:.0%} fewer users needed")
ax.legend(loc="upper left")
ax.grid(axis="x", visible=False)
viz.save(fig, FIG / "sim_cuped.png")

# %% [markdown]
# ![](../reports/figures/sim_cuped.png)

# %%
(ROOT / "reports" / "simulation_results.json").write_text(json.dumps(out, indent=2, default=float))

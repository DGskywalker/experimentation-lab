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
# # 01 · Cookie Cats: auditing a real A/B test
#
# **Experiment.** In the mobile puzzle game *Cookie Cats*, players hit a "gate" that forces
# them to wait or pay to keep going. The test moved that gate from **level 30** (control,
# `gate_30`) to **level 40** (treatment, `gate_40`). Players were randomised at install.
#
# **Question.** Should the gate move to level 40?
#
# | Role | Metric | Why |
# |---|---|---|
# | Primary | `retention_7`: played 7 days after install | long-run health of the game |
# | Secondary | `retention_1`: played 1 day after install | early signal |
# | Secondary | `sum_gamerounds`: rounds in the first 14 days | engagement |
#
# The analysis follows a fixed order: **trust the data → size the test → test → estimate → decide.**

# %%
from pathlib import Path
import json

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import abkit
from abkit import viz

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
viz.use_style()

con = duckdb.connect(str(ROOT / "data" / "lab.duckdb"), read_only=True)
df = con.sql("SELECT * FROM cookie_cats").df()
df.head()

# %% [markdown]
# ## 1 · Can we trust the data?

# %%
quality = con.sql("""
    SELECT
        count(*)                                   AS rows,
        count(DISTINCT userid)                     AS unique_users,
        count(*) FILTER (WHERE sum_gamerounds IS NULL OR retention_1 IS NULL OR retention_7 IS NULL) AS null_rows,
        count(*) FILTER (WHERE sum_gamerounds = 0) AS zero_round_players,
        max(sum_gamerounds)                        AS max_rounds,
        quantile_cont(sum_gamerounds, 0.999)       AS p999_rounds
    FROM cookie_cats
""").df()
quality

# %% [markdown]
# No duplicates and no nulls. One player logged **49,854 rounds** in 14 days (≈ 3,500/day)
# while the 99.9th percentile is under 1,000. That's a bot or a logging error, so it's
# excluded from the engagement analysis. It can't affect the 0/1 retention metrics.
#
# ### Sample ratio mismatch (SRM)
# The design is a 50/50 split. If the observed split is too far from that, assignment
# or logging is broken and any "effect" may just be selection bias.

# %%
counts = df.groupby("version").size()
srm = abkit.srm_test(counts.values)
srm

# %% [markdown]
# p ≈ 0.009: suspicious, but above the 0.001 alarm threshold the industry uses. (With
# 90k users, even small logging differences show up.) **Verdict: proceed, and note it as
# a caveat.** One more check: is an SRM consistent with the two arms having different
# *types* of player? Since gamerounds is measured *after* treatment it can't be used as a
# balance check, so this is as far as the data lets us go.

# %%
df_clean = df[df.sum_gamerounds < 10_000].copy()
ctrl, trt = df[df.version == "gate_30"], df[df.version == "gate_40"]
ctrl_c, trt_c = df_clean[df_clean.version == "gate_30"], df_clean[df_clean.version == "gate_40"]
len(df) - len(df_clean), "row removed for engagement metrics"

# %% [markdown]
# ## 2 · How sensitive was this test?
# Before testing, ask what effect this many users could actually detect. If the
# experiment is underpowered, "not significant" means nothing.

# %%
n_per_arm = int(counts.min())
base7, base1 = ctrl.retention_7.mean(), ctrl.retention_1.mean()
mde7 = abkit.mde_proportions(base7, n_per_arm)
mde1 = abkit.mde_proportions(base1, n_per_arm)
pd.DataFrame({
    "baseline": [base1, base7],
    "MDE (abs, 80% power)": [mde1, mde7],
    "MDE (relative)": [mde1 / base1, mde7 / base7],
}, index=["retention_1", "retention_7"]).round(4)

# %%
# Power curve: probability of detecting a true change of each size in retention_7
deltas = np.linspace(0.001, 0.015, 60)
fig, ax = plt.subplots(figsize=(7, 3.6))
for n, color, label in [(n_per_arm, viz.CONTROL, f"this test ({n_per_arm:,}/arm)"),
                        (n_per_arm // 4, viz.NEUTRAL, f"¼ the traffic ({n_per_arm // 4:,}/arm)")]:
    ax.plot(deltas * 100, [abkit.power_proportions(base7, base7 - d, n, n) for d in deltas], color=color, label=label)
ax.axhline(0.8, color=viz.INK_2, lw=1, ls=(0, (4, 3)))
ax.text(14.9 / 10, 0.815, "80% power", color=viz.INK_2, ha="right", va="bottom", fontsize=9)
ax.set(xlabel="True drop in 7-day retention (percentage points)", ylabel="Power", ylim=(0, 1.02),
       title="This test could reliably detect a 0.7pp change in 7-day retention")
ax.legend(loc="lower right")
viz.save(fig, FIG / "cc_power_curve.png")

# %% [markdown]
# ![](../reports/figures/cc_power_curve.png)
#
# With ~45k players per arm, the test has 80% power for a ≈0.7pp (≈4% relative) change in
# 7-day retention. It is well powered for effects that would matter to the business.
#
# ## 3 · Significance tests
# Three metrics means three chances of a false positive, so the family of p-values gets a
# Holm correction.

# %%
r1 = abkit.two_proportion_ztest(ctrl.retention_1.sum(), len(ctrl), trt.retention_1.sum(), len(trt), metric="retention_1")
r7 = abkit.two_proportion_ztest(ctrl.retention_7.sum(), len(ctrl), trt.retention_7.sum(), len(trt), metric="retention_7")
rg = abkit.welch_ttest(ctrl_c.sum_gamerounds, trt_c.sum_gamerounds, metric="sum_gamerounds")
mw = abkit.mann_whitney(ctrl_c.sum_gamerounds, trt_c.sum_gamerounds, metric="sum_gamerounds")

results = pd.DataFrame([r.to_dict() for r in (r7, r1, rg)]).set_index("metric")
results["p_holm"] = abkit.holm(results.p_value)["p_adjusted"]
results[["control", "treatment", "diff", "rel_lift", "ci_low", "ci_high", "p_value", "p_holm", "method"]].round(4)

# %%
pd.DataFrame([mw.to_dict()]).set_index("metric")[["control", "treatment", "stat", "p_value", "method"]]

# %% [markdown]
# * **7-day retention drops by 0.82pp** (19.02% → 18.20%, −4.3% relative). This is
#   significant after Holm correction (p ≈ 0.0016 raw).
# * **1-day retention** dips by 0.59pp, but that is not significant (p ≈ 0.07).
# * **Engagement**: the gate_40 group plays slightly fewer rounds. Game rounds are
#   extremely skewed, so the Mann–Whitney test is the trustworthy one here. Its result
#   is a marginal difference with a negligible effect size.
#
# ## 4 · Confidence intervals: analytic vs. bootstrap
# A Wald interval leans on the normal approximation. A 10,000-resample bootstrap makes
# no such assumption. If the two agree, the inference is robust.

# %%
boot7 = abkit.bootstrap_diff_ci(ctrl.retention_7.values, trt.retention_7.values, n_boot=10_000, seed=7)
boot1 = abkit.bootstrap_diff_ci(ctrl.retention_1.values, trt.retention_1.values, n_boot=10_000, seed=1)
ci_table = pd.DataFrame({
    "wald_low": [r1.ci_low, r7.ci_low], "wald_high": [r1.ci_high, r7.ci_high],
    "boot_low": [boot1["ci_low"], boot7["ci_low"]], "boot_high": [boot1["ci_high"], boot7["ci_high"]],
    "P(gate_40 better)": [boot1["p_treatment_better"], boot7["p_treatment_better"]],
}, index=["retention_1", "retention_7"])
(ci_table * 100).round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
for ax, boot, name in [(axes[0], boot1, "1-day retention"), (axes[1], boot7, "7-day retention")]:
    ax.hist(boot["boot_control"] * 100, bins=60, color=viz.CONTROL, alpha=0.75, label="gate_30 (control)")
    ax.hist(boot["boot_treatment"] * 100, bins=60, color=viz.TREATMENT, alpha=0.75, label="gate_40")
    ax.set(title=name, xlabel="Retention rate (%)", ylabel="Bootstrap resamples")
    ax.grid(axis="x", visible=False)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2, bbox_to_anchor=(1, 1.0))
fig.suptitle("Bootstrap distributions of the mean: 7-day retention barely overlaps", x=0.01, ha="left",
             fontweight="semibold", fontsize=12)
fig.tight_layout()
viz.save(fig, FIG / "cc_bootstrap.png")

# %% [markdown]
# ![](../reports/figures/cc_bootstrap.png)

# %%
fig, ax = plt.subplots(figsize=(7, 2.2))
rows = [("7-day retention", r7, boot7), ("1-day retention", r1, boot1)]
for i, (label, r, b) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([b["ci_low"] * 100, b["ci_high"] * 100], [y, y], color=viz.TREATMENT, lw=2, solid_capstyle="round")
    ax.plot(r.diff * 100, y, "o", ms=9, color=viz.TREATMENT, mec=viz.SURFACE, mew=2)
    ax.text(b["ci_high"] * 100 + 0.05, y, f"{r.diff * 100:+.2f}pp  [{b['ci_low'] * 100:+.2f}, {b['ci_high'] * 100:+.2f}]",
            va="center", color=viz.INK, fontsize=9)
ax.axvline(0, color=viz.INK_2, lw=1)
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
ax.set_xlim(-1.6, 0.9)
ax.set_ylim(-0.6, 1.6)
ax.set(xlabel="gate_40 − gate_30 (percentage points, 95% bootstrap CI)",
       title="Moving the gate to level 40 lowers 7-day retention")
ax.grid(axis="y", visible=False)
viz.save(fig, FIG / "cc_effect_ci.png")

# %% [markdown]
# ![](../reports/figures/cc_effect_ci.png)
#
# ## 5 · The Bayesian view: what a decision-maker actually asks
# "How likely is it that gate_40 is better, and how much do we lose if we pick wrong?"

# %%
bayes7 = abkit.beta_binomial_ab(ctrl.retention_7.sum(), len(ctrl), trt.retention_7.sum(), len(trt), seed=7)
bayes1 = abkit.beta_binomial_ab(ctrl.retention_1.sum(), len(ctrl), trt.retention_1.sum(), len(trt), seed=1)
pd.DataFrame({
    k: {m: b[m] for m in ("prob_treatment_better", "expected_loss_treatment", "expected_loss_control",
                          "diff_cred_low", "diff_cred_high")}
    for k, b in [("retention_1", bayes1), ("retention_7", bayes7)]
}).T.round(5)

# %%
fig, ax = plt.subplots(figsize=(7, 3.4))
d = (bayes7["posterior_treatment"] - bayes7["posterior_control"]) * 100
ax.hist(d, bins=80, color=viz.TREATMENT, alpha=0.85)
ax.axvline(0, color=viz.INK_2, lw=1)
ax.text(0.03, ax.get_ylim()[1] * 0.9, "no change", color=viz.INK_2, fontsize=9)
ax.set(xlabel="Posterior of the change in 7-day retention (pp)", ylabel="Posterior draws",
       title=f"P(gate_40 improves 7-day retention) = {bayes7['prob_treatment_better']:.2%}")
ax.grid(axis="x", visible=False)
viz.save(fig, FIG / "cc_posterior.png")

# %% [markdown]
# ![](../reports/figures/cc_posterior.png)
#
# ## 6 · Recommendation
#
# > **Keep the gate at level 30. Don't ship gate_40.**
# >
# > * Moving the gate to level 40 **lowers 7-day retention by 0.82pp** (95% CI roughly
# >   −1.3 to −0.3pp; −4.3% relative). It is significant after multiple-testing
# >   correction, and the bootstrap and Bayesian analyses agree.
# > * There is a **≈99.9% probability** that gate_40 is worse for 7-day retention. Shipping
# >   it has an expected loss of ~0.8pp of retention, versus ~0.0001pp for keeping gate_30.
# > * 1-day retention and engagement are flat to slightly negative. **No metric favours gate_40.**
# > * **Why (hypothesis):** an earlier gate makes players take a break sooner, which may
# >   keep the game feeling fresh (*hedonic adaptation*).
# > * **Caveats:** a borderline SRM (p ≈ 0.009) and one bot-like outlier. Neither changes
# >   the retention conclusion.
# > * **Next test:** try gate_25 vs. gate_30. If an earlier break helps, a still earlier
# >   one might help more. Use 7-day retention as the primary metric and revenue per user
# >   as a guardrail.

# %%
summary = {
    "srm": srm,
    "power": {"n_per_arm": n_per_arm, "mde_retention_7_abs": mde7, "mde_retention_1_abs": mde1},
    "tests": results.reset_index().to_dict(orient="records"),
    "mann_whitney_gamerounds": mw.to_dict(),
    "bootstrap": {k: {m: v[m] for m in ("ci_low", "ci_high", "p_treatment_better")} for k, v in
                  [("retention_1", boot1), ("retention_7", boot7)]},
    "bayes": {k: {m: v[m] for m in v if not m.startswith("posterior")} for k, v in
              [("retention_1", bayes1), ("retention_7", bayes7)]},
}
(ROOT / "reports" / "cookie_cats_results.json").write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(summary["tests"], indent=1, default=float)[:600])

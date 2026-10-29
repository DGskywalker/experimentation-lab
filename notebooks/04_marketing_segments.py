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
# # 04 · Marketing campaign: unequal splits, segments and the multiple-testing trap
#
# **Experiment.** 588,101 users were split into two groups. Most saw a company's **ad**;
# a small holdout saw a **public-service announcement (PSA)** in the same ad slots. The
# PSA group shows what would have happened without the campaign.
#
# **Questions.**
# 1. Did the campaign cause conversions, and how many?
# 2. Was the 96/4 split a good design?
# 3. Does the effect differ by day, by hour, or by how many ads someone saw? Which of
#    those "segment wins" survive a multiple-testing correction?

# %%
from pathlib import Path
import json

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable

import abkit
from abkit import viz

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG = ROOT / "reports" / "figures"
viz.use_style()
con = duckdb.connect(str(ROOT / "data" / "lab.duckdb"), read_only=True)
df = con.sql("SELECT * FROM marketing").df()
ad, psa = df[df.test_group == "ad"], df[df.test_group == "psa"]
out = {}
df.head()

# %% [markdown]
# ## 1 · Trust checks
# **The split.** The design deliberately put ~96% of users in the ad group. The documented
# ratio isn't available, so an SRM test against 50/50 would be meaningless (it would just
# "detect" the intended design). We report the observed split and check something we
# *can* test: whether the two groups saw ads in the same way.

# %%
split = df.test_group.value_counts(normalize=True)
print(f"Observed split: ad {split['ad']:.2%} / psa {split['psa']:.2%}  (n = {len(df):,}, all user_ids unique: {df.user_id.is_unique})")

day_mix = pd.crosstab(df.most_ads_day, df.test_group)
chi_day = stats.chi2_contingency(day_mix)
ks_ads = stats.ks_2samp(ad.total_ads, psa.total_ads)
balance = pd.DataFrame({
    "check": ["Peak-ad day mix (chi-square)", "Ads seen per user (KS test)"],
    "statistic": [chi_day.statistic, ks_ads.statistic],
    "p_value": [chi_day.pvalue, ks_ads.pvalue],
})
out["balance"] = balance.to_dict(orient="records")
balance

# %%
(day_mix / day_mix.sum()).round(3).rename(columns={"ad": "ad share", "psa": "psa share"})

# %% [markdown]
# **Neither check passes cleanly.**
# * **Day mix** (p ≈ 1e-47): PSA users are over-represented on Thursday and Friday and
#   under-represented on Saturday and Tuesday.
# * **Ads per user**: the distributions differ statistically (KS p ≈ 1e-31), but barely
#   in practice. The means are 24.8 vs 24.8 and the KS distance is 0.04. With 588k users,
#   tiny differences become "significant".
#
# Ad delivery evidently interacted with the group, for example through different
# campaign pacing.
#
# This matters because conversion rates differ a lot by day (Monday and Tuesday are high).
# If the arms are spread differently across days, the pooled comparison could be
# confounded. Section 3 checks this directly.
#
# ## 2 · Overall effect

# %%
overall = abkit.two_proportion_ztest(psa.converted.sum(), len(psa), ad.converted.sum(), len(ad), metric="conversion")
boot = abkit.bayes.beta_binomial_ab(psa.converted.sum(), len(psa), ad.converted.sum(), len(ad))
incremental = overall.diff * len(ad)
inc_ci = (overall.ci_low * len(ad), overall.ci_high * len(ad))
out["overall"] = {**overall.to_dict(), "incremental_conversions": incremental, "incremental_ci": inc_ci,
                  "prob_ads_better": boot["prob_treatment_better"]}
print(f"PSA {overall.control:.3%}  →  ad {overall.treatment:.3%}   lift {overall.diff * 100:+.3f}pp ({overall.rel_lift:+.0%} relative)")
print(f"95% CI {overall.ci_low * 100:+.3f} to {overall.ci_high * 100:+.3f}pp,  p = {overall.p_value:.1e}")
print(f"Incremental conversions caused by the campaign: {incremental:,.0f}  (95% CI {inc_ci[0]:,.0f}–{inc_ci[1]:,.0f})")

# %% [markdown]
# ### Was 96/4 a good split?
# An unbalanced split has two prices. A 50/50 split measures the effect most precisely,
# but it withholds the ad from half the audience. A small holdout protects revenue, but
# the estimate gets noisier.

# %%
def ci_halfwidth(share_ctrl: float, n: int = len(df), p_c: float = overall.control, p_t: float = overall.treatment) -> float:
    nc, nt = n * share_ctrl, n * (1 - share_ctrl)
    return 1.96 * np.sqrt(p_c * (1 - p_c) / nc + p_t * (1 - p_t) / nt)

shares = np.array([0.01, 0.02, 0.04, 0.10, 0.20, 0.50])
tradeoff = pd.DataFrame({
    "holdout share": shares,
    "CI half-width (pp)": [ci_halfwidth(s) * 100 for s in shares],
    "conversions forgone": shares * len(df) * overall.diff,
})
tradeoff["CI vs 50/50"] = tradeoff["CI half-width (pp)"] / ci_halfwidth(0.5) / 100
out["allocation"] = tradeoff.to_dict(orient="records")
tradeoff.round(3)

# %% [markdown]
# The 4% holdout gives up only ~180 conversions (vs ~2,260 at 50/50). Its confidence
# interval is ~2.3× wider, but the effect is so large that it's still unmistakable.
# **For a campaign expected to work, a small holdout is the right call.** For a subtle
# effect it wouldn't be.
#
# ## 3 · Is the pooled effect confounded by the day imbalance?
# The Cochran–Mantel–Haenszel test compares ad vs PSA *within* each day, then combines
# the results. The Breslow–Day test checks whether the effect is the same on every day.

# %%
tables = [pd.crosstab(g.test_group, g.converted).loc[["ad", "psa"], [1, 0]].values
          for _, g in df.groupby("most_ads_day")]
st = StratifiedTable(tables)
cmh = st.test_null_odds()
bd = st.test_equal_odds()
pooled_or = (ad.converted.mean() / (1 - ad.converted.mean())) / (psa.converted.mean() / (1 - psa.converted.mean()))
out["cmh"] = {"pooled_or": pooled_or, "mh_or": st.oddsratio_pooled, "mh_or_ci": st.oddsratio_pooled_confint(),
              "cmh_p": cmh.pvalue, "breslow_day_p": bd.pvalue}
pd.Series(out["cmh"], dtype=object)

# %% [markdown]
# The day-adjusted odds ratio is 1.44 (95% CI 1.30–1.59), essentially the same as the
# pooled 1.44. **The day
# imbalance doesn't explain the effect, and there is no Simpson's paradox here.** The
# Breslow–Day test (p ≈ 0.013), though, says the size of the effect differs across days. That leads
# to the segment analysis.
#
# ## 4 · Segments, and the multiple-testing trap
# Slicing by 7 days, 24 hours and 6 exposure buckets gives 37 tests. At α = 0.05, about 2
# would come up "significant" by luck even if nothing varied. Benjamini–Hochberg (BH)
# controls the false discovery rate across the whole family.

# %%
bins = [0, 2, 5, 13, 27, 57, np.inf]
labels = ["1–2", "3–5", "6–13", "14–27", "28–57", "58+"]
df["ads_bucket"] = pd.cut(df.total_ads, bins=bins, labels=labels)

rows = []
for dim in ["most_ads_day", "most_ads_hour", "ads_bucket"]:
    for level, g in df.groupby(dim, observed=True):
        a, p = g[g.test_group == "ad"].converted, g[g.test_group == "psa"].converted
        r = abkit.two_proportion_ztest(p.sum(), len(p), a.sum(), len(a))
        rows.append({"dimension": dim, "segment": str(level), "n_psa": len(p), "psa": r.control, "ad": r.treatment,
                     "lift_pp": r.diff * 100, "ci_low": r.ci_low * 100, "ci_high": r.ci_high * 100, "p_raw": r.p_value})
seg = pd.DataFrame(rows)
seg["p_bh"] = abkit.benjamini_hochberg(seg.p_raw)["p_adjusted"]
seg["p_holm"] = abkit.holm(seg.p_raw)["p_adjusted"]
seg["sig_raw"], seg["sig_bh"] = seg.p_raw < 0.05, seg.p_bh < 0.05
summary_sig = seg.groupby("dimension")[["sig_raw", "sig_bh"]].sum().assign(tests=seg.groupby("dimension").size())
out["segments_significant"] = summary_sig.to_dict()
summary_sig

# %%
seg[seg.dimension == "most_ads_day"].round(4)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
d = seg[seg.dimension == "most_ads_day"].set_index("segment").loc[day_order].reset_index()
ax = axes[0]
for i, r in d.iterrows():
    y = len(d) - 1 - i
    color = viz.CONTROL if r.sig_bh else viz.NEUTRAL
    ax.plot([r.ci_low, r.ci_high], [y, y], color=color, lw=2, solid_capstyle="round")
    ax.plot(r.lift_pp, y, "o", ms=8, color=color, mec=viz.SURFACE, mew=2)
ax.axvline(0, color=viz.INK_2, lw=1)
ax.axvline(overall.diff * 100, color=viz.INK_2, lw=1, ls=(0, (4, 3)))
ax.text(overall.diff * 100 + 0.03, len(d) - 0.45, "overall", color=viz.INK_2, fontsize=8.5)
ax.set_yticks(range(len(d)), d.segment[::-1])
ax.set(xlabel="Ad − PSA conversion (pp, 95% CI)", title="Lift by peak-ad day", ylim=(-0.6, len(d) - 0.2))
ax.grid(axis="y", visible=False)
ax.text(0.99, 0.02, "blue = significant after BH\ngray = not", transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.5, color=viz.INK_2)

ax = axes[1]
b = df.groupby(["ads_bucket", "test_group"], observed=True).converted.mean().unstack() * 100
x = np.arange(len(b))
ax.plot(x, b["ad"], color=viz.TREATMENT, marker="o", ms=7, mec=viz.SURFACE, mew=2)
ax.plot(x, b["psa"], color=viz.CONTROL, marker="o", ms=7, mec=viz.SURFACE, mew=2)
ax.text(x[-1] + 0.12, b["ad"].iloc[-1], "ad", va="center", fontsize=9)
ax.text(x[-1] + 0.12, b["psa"].iloc[-1], "PSA", va="center", fontsize=9)
ax.set_xticks(x, b.index.astype(str))
ax.set(xlabel="Ads seen by the user", ylabel="Conversion rate (%)", xlim=(-0.3, len(b) - 0.4),
       title="All of the lift comes from users who saw 28+ ads")
viz.save(fig, FIG / "mk_segments.png")

# %% [markdown]
# ![](../reports/figures/mk_segments.png)
#
# **Reading the segments carefully**
# * **Exposure is the big finding. The campaign's entire effect comes from heavy-exposure
#   users.** Users who saw up to 27 ads convert at the same rate whether they saw ads or
#   PSAs: below 1.3% either way, with no consistent gap. That's 76% of the audience. At
#   28–57 ads it's 4.3% vs 3.1%, and at 58+ ads it's 14.6% vs 8.4%.
#   * **Caveat:** exposure isn't randomised. Heavy users are more engaged to begin with,
#     which is why *both* lines rise. The comparison *within* a bucket is still ad vs PSA
#     at the same exposure, and exposure is almost identical across arms, so the pattern
#     is a strong lead. It isn't proof of a dose–response.
# * **Days:** after BH correction, the lift is significant on Monday, Tuesday, Wednesday
#   and Saturday. **Friday's raw "win" (p = 0.012) doesn't survive.** Thursday and Sunday
#   show no detectable lift. *Peak-ad day* describes the user, not a randomised day, so
#   treat this as a hypothesis to test.
# * **Hours: 8 of 24 hours look significant on raw p-values, and only 1 survives BH.**
#   This is the multiple-testing trap. A team that re-planned its schedule around "the 8
#   best hours" would be chasing noise.
#
# ## 5 · Recommendation
#
# > **The campaign works. Keep it running.** Ads raise conversion from 1.79% to 2.55%
# > (+0.77pp, +43% relative; p ≈ 1e-13), about **4,300 incremental conversions** across the
# > ad group.
# >
# > * The 4% PSA holdout was a sensible, cheap design for an effect this large. Keep a
# >   small permanent holdout to keep measuring incrementality.
# > * The result holds up after adjusting for the uneven day mix (CMH). Investigate why
# >   ad delivery differed by group: it's a pipeline issue even though it didn't bias this result.
# > * **Where the value is:** essentially all incremental conversions come from users
# >   reached 28+ times. Light-exposure impressions show no measurable return, which is a
# >   budget question for the media team.
# > * **Next experiment:** randomise frequency caps (e.g. 10 / 30 / 60 ads) to test whether
# >   *more exposure causes* conversion, or heavy users would convert anyway. Separately,
# >   randomise a Monday–Wednesday-weighted schedule against the current one. Don't
# >   re-plan by hour: that signal didn't survive correction.

# %%
(ROOT / "reports" / "marketing_results.json").write_text(json.dumps(out, indent=2, default=float))

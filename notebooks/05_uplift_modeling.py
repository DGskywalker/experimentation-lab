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
# # 05 · Uplift modelling: who should we actually treat?
#
# An A/B test answers *"does the treatment work on average?"*. Uplift modelling answers
# *"for whom does it work?"*. Customers fall into four groups:
#
# | | Converts if treated | Doesn't convert if treated |
# |---|---|---|
# | **Converts anyway** | Sure thing: treating them is wasted money | Sleeping dog: treating them *hurts* |
# | **Doesn't convert otherwise** | **Persuadable: the only ones worth treating** | Lost cause |
#
# A classic "response model" (who is likely to convert if we email them?) targets sure
# things and persuadables alike. An **uplift model** estimates
# τ(x) = P(convert | treated, x) − P(convert | control, x) and targets persuadables.
#
# This notebook uses two randomised experiments:
# * **Hillstrom e-mail campaign** (64k customers, 3 arms): interpretable features.
# * **Criteo ad-incrementality test** (1M-row random sample of 14M users, 85/15 split):
#   industrial scale, anonymised features.

# %%
from pathlib import Path
import json

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import abkit
from abkit import uplift, viz

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG = ROOT / "reports" / "figures"
viz.use_style()
con = duckdb.connect(str(ROOT / "data" / "lab.duckdb"), read_only=True)
out = {}

CLF = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=200,
                                     l2_regularization=1.0, random_state=0)
REG = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=200,
                                    l2_regularization=1.0, random_state=0)
LOGIT = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


class ResponseModel:
    """The traditional approach: score P(outcome | treated, x) and target the highest."""

    def __init__(self, model):
        from sklearn.base import clone
        self.m = clone(model)

    def fit(self, X, t, y):
        X, t, y = np.asarray(X), np.asarray(t), np.asarray(y)
        self.m.fit(X[t == 1], y[t == 1])
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


def fit_all(X_tr, t_tr, y_tr):
    return {
        "Response model (classic)": ResponseModel(CLF).fit(X_tr, t_tr, y_tr),
        "S-learner": uplift.SLearner(CLF).fit(X_tr, t_tr, y_tr),
        "T-learner": uplift.TLearner(CLF).fit(X_tr, t_tr, y_tr),
        "X-learner": uplift.XLearner(CLF, REG).fit(X_tr, t_tr, y_tr),
        "S-learner (logistic)": uplift.SLearner(LOGIT).fit(X_tr, t_tr, y_tr),
    }


def evaluate(models, X_te, t_te, y_te, k=0.3, n_boot=200):
    rows, curves = [], {}
    rng = np.random.default_rng(0)
    preds = {name: m.predict(X_te) for name, m in models.items()}
    preds["Random targeting"] = rng.random(len(y_te))
    for name, u in preds.items():
        curves[name] = uplift.qini_curve(y_te, t_te, u)
        lo, hi = uplift.bootstrap_metric(uplift.qini_coefficient, y_te, t_te, u, n_boot=n_boot)
        rows.append({
            "model": name,
            "qini_coef": uplift.qini_coefficient(y_te, t_te, u),
            "qini_ci_low": lo, "qini_ci_high": hi,
            f"uplift@{int(k * 100)}% (pp)": uplift.uplift_at_k(y_te, t_te, u, k) * 100,
            f"share of incremental in top {int(k * 100)}%": uplift.share_of_incremental(y_te, t_te, u, k),
        })
    return pd.DataFrame(rows).set_index("model"), curves, preds


COLORS = {"Random targeting": viz.NEUTRAL, "Response model (classic)": viz.INK_2,
          "S-learner": viz.THIRD, "T-learner": viz.TREATMENT, "X-learner": viz.CONTROL,
          "S-learner (logistic)": "#eda100"}


def plot_qini(curves, title, path, unit):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, c in curves.items():
        ls = (0, (4, 3)) if name in ("Random targeting", "Response model (classic)") else "-"
        ax.plot(c["fraction"] * 100, c["qini"], color=COLORS[name], lw=2, ls=ls)
    ax.axvline(30, color=viz.GRID, lw=1)
    ax.set(xlabel="Share of customers treated, best-ranked first (%)", ylabel=f"Incremental {unit} (test set)",
           title=title, xlim=(0, 100))
    ax.legend(list(curves), loc="lower right", fontsize=8.5)
    viz.save(fig, path)


# %% [markdown]
# ## Part A · Hillstrom e-mail campaign
# 64,000 recent customers were randomised to a **men's merchandise e-mail**, a **women's
# merchandise e-mail**, or **no e-mail**. Outcome: a website visit in the next two weeks.

# %%
hill = con.sql("SELECT * FROM hillstrom").df()
ate = []
for arm in ["Mens E-Mail", "Womens E-Mail"]:
    for metric in ["visit", "conversion"]:
        c, t = hill[hill.segment == "No E-Mail"][metric], hill[hill.segment == arm][metric]
        r = abkit.two_proportion_ztest(c.sum(), len(c), t.sum(), len(t), metric=f"{arm}: {metric}")
        ate.append(r.to_dict())
spend = abkit.welch_ttest(hill[hill.segment == "No E-Mail"].spend, hill[hill.segment == "Mens E-Mail"].spend,
                          metric="Mens E-Mail: spend ($)")
ate.append(spend.to_dict())
ate = pd.DataFrame(ate).set_index("metric")[["control", "treatment", "diff", "rel_lift", "ci_low", "ci_high", "p_value"]]
out["hillstrom_ate"] = ate.reset_index().to_dict(orient="records")
ate.round(4)

# %% [markdown]
# Both e-mails work on average. The men's e-mail is stronger: visits rise from 10.6% to
# 18.3%. The rest of Part A models **men's e-mail vs. no e-mail** on *visit*: who gains
# the most from that e-mail?

# %%
h = hill[hill.segment.isin(["Mens E-Mail", "No E-Mail"])].copy()
h["t"] = (h.segment == "Mens E-Mail").astype(int)
X = pd.get_dummies(h[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
                   columns=["zip_code", "channel"], dtype=float)
feature_names = list(X.columns)
X_tr, X_te, t_tr, t_te, y_tr, y_te, h_tr, h_te = train_test_split(
    X.values, h.t.values, h.visit.values, h, test_size=0.4, random_state=0, stratify=h.t * 2 + h.visit)
models_h = fit_all(X_tr, t_tr, y_tr)
res_h, curves_h, preds_h = evaluate(models_h, X_te, t_te, y_te)
out["hillstrom_models"] = res_h.reset_index().to_dict(orient="records")
res_h.round(3)

# %%
plot_qini(curves_h, "Hillstrom: no model clearly beats random targeting", FIG / "up_hillstrom_qini.png", "visits")

# %% [markdown]
# ![](../reports/figures/up_hillstrom_qini.png)
#
# ### Is that a fluke of one split? Re-fit on 5 different splits
# One train/test split can mislead in either direction. Model choice should rest on
# performance averaged over several splits, never on the single test set we report.

# %%
rob = []
for seed in range(1, 6):
    a_tr, a_te, b_tr, b_te, c_tr, c_te = train_test_split(X.values, h.t.values, h.visit.values, test_size=0.4,
                                                          random_state=seed, stratify=h.t * 2 + h.visit)
    for name, m in fit_all(a_tr, b_tr, c_tr).items():
        rob.append({"seed": seed, "model": name, "qini": uplift.qini_coefficient(c_te, b_te, m.predict(a_te))})
rob = pd.DataFrame(rob).groupby("model").qini.agg(["mean", "std", "min", "max"]).sort_values("mean", ascending=False)
out["hillstrom_robustness"] = rob.reset_index().to_dict(orient="records")
rob.round(3)

# %% [markdown]
# The flexible gradient-boosted learners hover around zero, and sometimes go *negative*:
# they fit noise. The heavily regularised logistic S-learner is small but consistently
# positive. **The signal is real but weak, and model complexity has to match it.**
#
# ### Do the predictions mean anything? Calibration by decile
# Rank test customers by predicted uplift and split them into 10 groups. Within each
# group, treatment is still randomised, so treated rate − control rate is an *unbiased*
# estimate of that group's true uplift. If the model works, the observed uplift should
# fall from the top decile to the bottom.

# %%
best_h = rob.drop("Response model (classic)").index[0]  # chosen on the 5 robustness splits, not the test set
u = preds_h[best_h]
dec = pd.qcut(pd.Series(u).rank(method="first"), 10, labels=False)
cal = pd.DataFrame({"decile": 10 - dec, "u": u, "t": t_te, "y": y_te})
cal_tbl = cal.groupby("decile").apply(lambda g: pd.Series({
    "predicted (pp)": g.u.mean() * 100,
    "observed (pp)": (g[g.t == 1].y.mean() - g[g.t == 0].y.mean()) * 100,
    "se (pp)": np.sqrt(g[g.t == 1].y.var() / (g.t == 1).sum() + g[g.t == 0].y.var() / (g.t == 0).sum()) * 100,
}), include_groups=False)
out["hillstrom_calibration"] = cal_tbl.reset_index().to_dict(orient="records")

fig, ax = plt.subplots(figsize=(7.5, 3.6))
x = cal_tbl.index.values
ax.bar(x, cal_tbl["observed (pp)"], color=viz.CONTROL, width=0.7, label="Observed uplift (treated − control)")
ax.errorbar(x, cal_tbl["observed (pp)"], yerr=1.96 * cal_tbl["se (pp)"], fmt="none", ecolor=viz.INK_2, elinewidth=1, capsize=0)
ax.plot(x, cal_tbl["predicted (pp)"], color=viz.TREATMENT, marker="o", ms=7, mec=viz.SURFACE, mew=2, label=f"Predicted ({best_h})")
ax.axhline(0, color=viz.INK_2, lw=1)
ax.set_xticks(x, [f"{i}" for i in x])
ax.set(xlabel="Decile of predicted uplift (1 = most persuadable)", ylabel="Visit uplift (pp)",
       title="Every decile responds to the e-mail; the ranking is weak")
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(axis="x", visible=False)
viz.save(fig, FIG / "up_hillstrom_deciles.png")
cal_tbl.round(2)

# %% [markdown]
# ![](../reports/figures/up_hillstrom_deciles.png)
#
# ### Who are the persuadables?

# %%
h_te = h_te.assign(pred_uplift=u)
profile = pd.concat([
    h_te.groupby("womens").pred_uplift.mean().rename(lambda v: f"bought womens before = {v}"),
    h_te.groupby("mens").pred_uplift.mean().rename(lambda v: f"bought mens before = {v}"),
    h_te.groupby("newbie").pred_uplift.mean().rename(lambda v: f"new customer = {v}"),
    h_te.groupby("channel").pred_uplift.mean().rename(lambda v: f"channel = {v}"),
    h_te.groupby(pd.cut(h_te.recency, [0, 3, 6, 9, 12])).pred_uplift.mean().rename(lambda v: f"recency (months) {v}"),
]).mul(100).rename("mean predicted uplift (pp)")
out["hillstrom_profile"] = profile.to_dict()
profile.round(2).to_frame()

# %% [markdown]
# **Part A verdict: send the men's e-mail to everyone.**
# * The e-mail lifts visits by 7.7pp (+72%) on average, and *every* decile of predicted
#   uplift responds: observed uplift ranges from ~6pp to ~10pp.
# * The best models rank customers only slightly better than chance (Qini ≈ 0.03).
#   Mailing the "top 30%" would capture about a third of the incremental visits, barely
#   more than random. With e-mail costing almost nothing, targeting would mostly *lose*
#   visits.
# * The persuadable profile is plausible but modest: multichannel shoppers, recent
#   buyers and past men's-merchandise buyers respond a little more.
# * **Lesson:** heterogeneity has to be *tested for* before it's acted on. A complex
#   model will always produce a confident-looking ranking.

# %% [markdown]
# ## Part B · Criteo: uplift at industrial scale
# Criteo randomised 14M users: 85% eligible to see ads, 15% held out. The features are 12
# anonymised user signals. Outcome: a site **visit**. We work on a reproducible 1M-row
# random sample. With an 85/15 split the control arm is small, which is the situation
# the X-learner was designed for.

# %%
cr = con.sql("SELECT * FROM criteo").df()
fcols = [f"f{i}" for i in range(12)]
c_ate = abkit.two_proportion_ztest(cr[cr.treatment == 0].visit.sum(), (cr.treatment == 0).sum(),
                                   cr[cr.treatment == 1].visit.sum(), (cr.treatment == 1).sum(), metric="criteo visit")
out["criteo_ate"] = c_ate.to_dict()
print(f"Visit rate: control {c_ate.control:.3%}  treated {c_ate.treatment:.3%}  "
      f"ATE {c_ate.diff * 100:+.2f}pp ({c_ate.rel_lift:+.0%}), 95% CI [{c_ate.ci_low * 100:+.2f}, {c_ate.ci_high * 100:+.2f}]")

Xc_tr, Xc_te, tc_tr, tc_te, yc_tr, yc_te = train_test_split(
    cr[fcols].values, cr.treatment.values, cr.visit.values, test_size=0.3, random_state=0,
    stratify=cr.treatment * 2 + cr.visit)
models_c = fit_all(Xc_tr, tc_tr, yc_tr)
res_c, curves_c, preds_c = evaluate(models_c, Xc_te, tc_te, yc_te, n_boot=100)
out["criteo_models"] = res_c.reset_index().to_dict(orient="records")
res_c.round(3)

# %%
plot_qini(curves_c, "Criteo: uplift models vs a classic response model", FIG / "up_criteo_qini.png", "visits")

# %% [markdown]
# ![](../reports/figures/up_criteo_qini.png)

# %%
# Budget view: incremental visits per 1,000 users treated, if only the top k% get ads
best_c = "X-learner"  # designed for unbalanced arms; chosen up front rather than on the test set
budget = []
for k in [0.1, 0.2, 0.3, 0.5, 1.0]:
    share = uplift.share_of_incremental(yc_te, tc_te, preds_c[best_c], k) if k < 1 else 1.0
    resp = uplift.share_of_incremental(yc_te, tc_te, preds_c["Response model (classic)"], k) if k < 1 else 1.0
    budget.append({"treat top": f"{k:.0%}", f"{best_c}: share of incremental visits": share,
                   "response model: share": resp,
                   f"{best_c}: incremental visits per 1k treated": share * c_ate.diff * 1000 / k})
budget = pd.DataFrame(budget).set_index("treat top")
out["criteo_budget"] = budget.reset_index().to_dict(orient="records")
budget.round(3)

# %% [markdown]
# **Part B verdict: here targeting pays off, by a lot.**
# * Showing ads to everyone lifts visits by 1.13pp (3.73% → 4.86%, +30%).
# * The effect is concentrated: **the X-learner's top 30% of users hold ~86% of all
#   incremental visits.** That's ~32 extra visits per 1,000 users treated, vs ~11 when
#   treating everyone, so the same budget buys about **2.9× more incremental visits**.
# * **Honest comparison:** the classic response model is a strong baseline here (top 30%
#   ≈ 85%), because on Criteo the users likely to visit are also the ones the ads move.
#   Uplift models earn their keep at *tight* budgets: at 20%, 77% vs 73%. They also matter
#   whenever "likely to convert" and "persuadable" diverge, as in Hillstrom, where the
#   response model has no edge.
# * **Next step for production:** confirm in a new randomised test, with users split
#   into model-targeted vs random-targeted at the same budget. A Qini curve on
#   historical data is evidence, not proof.

# %%
(ROOT / "reports" / "uplift_results.json").write_text(json.dumps(out, indent=2, default=float))

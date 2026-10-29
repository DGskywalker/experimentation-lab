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
# # 03 · Designing a booking-form experiment and rehearsing it before launch
#
# **Proposal.** The hotel booking form has 12 fields. Product wants to remove some of them
# to increase completions. Two candidate variants are tested against the current form:
#
# | Arm | Change |
# |---|---|
# | **A_control** | current 12-field form |
# | **B_fewer_fields** | 8 fields: phone, company, address line 2 and "how did you hear about us" removed |
# | **C_two_step** | all 12 fields, split into 2 steps: essentials first, details after |
#
# **The risk.** Fewer fields can mean worse bookings. Without a phone number the hotel
# can't reconfirm, so cancellations and no-shows might rise. A form that converts more
# but cancels more could *lose* money.
#
# The pre-registration in [`docs/preregistration_booking_form.md`](../docs/preregistration_booking_form.md)
# is written from the outputs of this notebook.

# %%
from pathlib import Path
from collections import Counter
import json

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import abkit
from abkit import viz
from labsim.booking import ArmEffect, Design, analyse, load_bookings, plan, simulate

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG = ROOT / "reports" / "figures"
viz.use_style()
con = duckdb.connect(str(ROOT / "data" / "lab.duckdb"), read_only=True)

# %% [markdown]
# ## 1 · Baselines from real booking data
# The Hotel Booking Demand dataset (119k real bookings) provides the guardrail baselines:
# how often bookings cancel and what they're worth.

# %%
bookings = load_bookings(con)
baseline = con.sql("""
    SELECT market_segment AS segment, count(*) AS bookings,
           avg(is_canceled) AS cancel_rate,
           median(adr * (stays_in_weekend_nights + stays_in_week_nights)) AS median_value
    FROM hotels
    WHERE market_segment IN ('Direct','Online TA','Corporate','Offline TA/TO') AND adr > 0
      AND stays_in_weekend_nights + stays_in_week_nights > 0
    GROUP BY 1 ORDER BY 2 DESC
""").df()
print(f"Overall cancellation: {bookings.is_canceled.mean():.1%}   mean booking value: ${bookings.booking_value.mean():.0f}")
baseline.round(3)

# %% [markdown]
# **Assumptions not in the data** (stated so reviewers can challenge them):
# * 35% of visitors who start the form complete it (typical for travel checkout funnels).
# * 4,000 form starts per day.
# * Visitors with more pre-period site visits are more likely to complete. This gives
#   CUPED a covariate to work with.

# %% [markdown]
# ## 2 · How big must the test be?

# %%
design = Design()
p = plan(design)
pd.Series({k: v for k, v in p.items() if k != "obf_z_boundaries"}, dtype=object)

# %% [markdown]
# Two requirements compete:
#
# * **Primary metric:** detecting a +5% relative lift in completion (35.0% → 36.75%) with
#   80% power needs **14,009 users per arm**. Comparing two variants against one control
#   uses Dunnett's correction (critical z = 2.21), which costs slightly less than Bonferroni.
# * **Guardrail:** showing that cancellations rise by *no more than 1.5pp* needs 12,151
#   *completed bookings* per arm. At 35% completion, that's **34,718 visitors per arm**.
#
# **The guardrail sets the size of the test, not the primary metric.** The test runs for
# **4 full weeks** (104k visitors), with an O'Brien–Fleming look at the end of each week.
# Sizing for the primary metric alone would end the test after 2 weeks. It would then
# "win" on completions, with no power to detect a lead-quality problem.

# %%
mdes = np.linspace(0.03, 0.10, 50)
n_primary = [abkit.sample_size_proportions(0.35, m, alpha=p["alpha_per_comparison"], relative=True)[0] for m in mdes]
margins = np.linspace(0.01, 0.03, 50)
n_guard = [abkit.sample_size_means(np.sqrt(.33 * .67), m, alpha=.05, two_sided=False)[0] / .35 for m in margins]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
axes[0].plot(mdes * 100, np.array(n_primary) / 1000, color=viz.CONTROL)
axes[0].plot(5, p["n_per_arm_primary"] / 1000, "o", ms=8, color=viz.CONTROL, mec=viz.SURFACE, mew=2)
axes[0].annotate(f"{p['n_per_arm_primary']:,}/arm at 5%", (5, p["n_per_arm_primary"] / 1000), xytext=(8, 8),
                 textcoords="offset points", fontsize=9)
axes[0].set(xlabel="Detectable lift in completion (relative %)", ylabel="Visitors per arm (thousands)",
            title="Primary: completion rate")
axes[1].plot(margins * 100, np.array(n_guard) / 1000, color=viz.TREATMENT)
axes[1].plot(1.5, p["n_per_arm_guardrail"] / 1000, "o", ms=8, color=viz.TREATMENT, mec=viz.SURFACE, mew=2)
axes[1].annotate(f"{p['n_per_arm_guardrail']:,}/arm at 1.5pp", (1.5, p["n_per_arm_guardrail"] / 1000),
                 xytext=(8, 8), textcoords="offset points", fontsize=9)
axes[1].set(xlabel="Non-inferiority margin on cancellation (pp)", title="Guardrail: cancellation rate")
fig.suptitle("The guardrail, not the primary metric, sets the experiment size", x=0.01, ha="left",
             fontweight="semibold", fontsize=12)
fig.tight_layout()
viz.save(fig, FIG / "bf_sample_size.png")

# %% [markdown]
# ![](../reports/figures/bf_sample_size.png)
#
# ## 3 · Dress rehearsal: run the experiment in simulation
# The ground truth used for this rehearsal is hidden from the analysis:
# * **B** really does lift completion (odds ×1.13), and it also raises cancellations by 3pp
#   (no phone number, so no reconfirmation call).
# * **C** lifts completion (odds ×1.09) with no effect on cancellations.
#
# Visitors are assigned by `abkit.assign`, the same hash-based function a production SDK
# would use.

# %%
truth = {
    "A_control": ArmEffect(),
    "B_fewer_fields": ArmEffect(completion_odds=1.13, cancel_pp=0.03),
    "C_two_step": ArmEffect(completion_odds=1.09, cancel_pp=0.0),
}
df = simulate(bookings, p["users_at_planned_end"] * len(design.arms), truth, design, p["days_planned"], seed=42)
df.groupby("arm")[["completed", "canceled", "revenue", "sessions", "pre_visits"]].mean().round(4)

# %% [markdown]
# ### Weekly interim looks
# The analysis is pre-registered to look once a week, and to stop early only if |z|
# crosses the O'Brien–Fleming boundary.

# %%
looks = []
for week in range(1, p["weekly_looks"] + 1):
    sub = df[df.enrol_day <= week * 7]
    r = analyse(sub, design)
    for arm, res in r["arms"].items():
        looks.append({"week": week, "arm": arm, "z": res["primary"]["z"],
                      "boundary": p["obf_z_boundaries"][week - 1], "n": len(sub)})
looks = pd.DataFrame(looks)
looks.pivot(index="week", columns="arm", values="z").assign(boundary=p["obf_z_boundaries"]).round(2)

# %%
fig, ax = plt.subplots(figsize=(7, 3.6))
weeks = np.arange(1, p["weekly_looks"] + 1)
ax.plot(weeks, p["obf_z_boundaries"], color=viz.INK_2, lw=1.5, ls=(0, (4, 3)), marker="o", ms=4)
ax.text(2.55, 2.45, "O'Brien–Fleming\nstop-early boundary", color=viz.INK_2, fontsize=9, va="top")
for arm, color in [("B_fewer_fields", viz.TREATMENT), ("C_two_step", viz.THIRD)]:
    zz = looks[looks.arm == arm].z.values
    ax.plot(weeks, zz, color=color, marker="o", ms=7, mec=viz.SURFACE, mew=2)
    ax.text(weeks[-1] + 0.08, zz[-1], arm, color=viz.INK, fontsize=9, va="center")
ax.set(xlabel="Week", ylabel="z-statistic for completion vs control", xlim=(0.8, 4.9),
       title="Completion z-scores at each weekly look vs O'Brien–Fleming boundaries")
ax.set_xticks(weeks)
viz.save(fig, FIG / "bf_sequential.png")

# %% [markdown]
# ![](../reports/figures/bf_sequential.png)
#
# B crosses the completion boundary in week 2. **It does not stop there.** The test is
# sized for the cancellation guardrail, so the pre-registered rule is: *stop early for
# success only if the guardrail is also proven non-inferior at that look.* Stopping
# early for **harm** is always allowed.
#
# ### Final readout (day 28)

# %%
final = analyse(df, design)
print("SRM:", {k: final["srm"][k] for k in ("observed", "p_value", "srm_detected")})
rows = []
for arm, r in final["arms"].items():
    rows.append({
        "arm": arm,
        "completion lift (pp)": r["primary"]["diff"] * 100,
        "completion CI (pp)": f"[{r['primary']['ci_low'] * 100:+.2f}, {r['primary']['ci_high'] * 100:+.2f}]",
        "cancellation Δ (pp)": r["cancellation"]["diff"] * 100,
        "cancel upper bound (pp)": r["cancellation"]["ci_high"] * 100,
        "revenue/visitor Δ ($)": r["revenue_per_visitor"]["diff"],
        "rev/visitor p": r["revenue_per_visitor"]["p_value"],
        "decision": r["decision"],
    })
readout = pd.DataFrame(rows).set_index("arm")
readout.round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), sharey=True)
arms = list(final["arms"])
for i, arm in enumerate(arms):
    y = len(arms) - 1 - i
    r = final["arms"][arm]
    for ax, key, color in [(axes[0], "primary", viz.CONTROL), (axes[1], "cancellation", viz.TREATMENT)]:
        lo, hi, d = r[key]["ci_low"] * 100, r[key]["ci_high"] * 100, r[key]["diff"] * 100
        ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round")
        ax.plot(d, y, "o", ms=9, color=color, mec=viz.SURFACE, mew=2)
        ax.text(hi + 0.15, y, f"{d:+.2f}pp", va="center", fontsize=9)
axes[0].axvline(0, color=viz.INK_2, lw=1)
axes[1].axvline(0, color=viz.INK_2, lw=1)
axes[1].axvline(design.cancel_margin_pp * 100, color=viz.CRITICAL, lw=1.2, ls=(0, (4, 3)))
axes[1].text(design.cancel_margin_pp * 100 + 0.1, 1.45, "⚠ max acceptable\n+1.5pp", color=viz.CRITICAL, fontsize=8.5, va="top")
axes[0].set_yticks(range(len(arms)), arms[::-1])
axes[0].set(xlabel="Δ completion vs control (pp, Dunnett CI)", title="Primary: completion ↑ is good", ylim=(-0.6, 1.6))
axes[1].set(xlabel="Δ cancellation vs control (pp, 90% CI)", title="Guardrail: cancellation ↑ is bad")
for ax in axes:
    ax.grid(axis="y", visible=False)
fig.tight_layout()
viz.save(fig, FIG / "bf_readout.png")

# %% [markdown]
# ![](../reports/figures/bf_readout.png)
#
# **Interpretation**, generated from the results so it always matches the run:

# %%
for arm, r in final["arms"].items():
    pr, ca, rv = r["primary"], r["cancellation"], r["revenue_per_visitor"]
    print(f"{arm}: completion {pr['diff'] * 100:+.2f}pp (rel {pr['rel_lift']:+.1%}), "
          f"cancellation {ca['diff'] * 100:+.2f}pp [upper bound {ca['ci_high'] * 100:+.2f}pp vs +{design.cancel_margin_pp * 100:.1f}pp margin], "
          f"revenue/visitor {rv['diff']:+.2f}$ (p={rv['p_value']:.3g})\n  -> {r['decision']}\n")

# %% [markdown]
# **What this run shows (seed 42)**
# * **B_fewer_fields: don't ship.** Completion rises, but cancellations rise significantly
#   too. This is the lead-quality harm the guardrail exists to catch. Revenue per visitor
#   is still up in this run, so a team watching only revenue would have shipped a change
#   that fills the hotel's books with bookings that later cancel. **Next iteration:** keep
#   the phone field and remove the other three.
# * **C_two_step: hold.** Completion clearly rises. Cancellations are up +0.9pp, which is
#   inside the margin, but the upper bound (+1.9pp) crosses it, so the test can't yet
#   *prove* C is safe. In this simulation we know C's true harm is zero. This run is one
#   of the ~16% of cases where an honest design can't confirm safety in time. The
#   pre-registered response is to extend or re-test, **not** to relax the margin after
#   seeing the data.
#
# ## 4 · Is the plan reliable? Re-run the whole rehearsal 300 times

# %%
decisions = {a: Counter() for a in arms}
cuped_gain = []
for seed in range(300):
    sim = simulate(bookings, p["users_at_planned_end"] * len(design.arms), truth, design, p["days_planned"], seed=1000 + seed)
    res = analyse(sim, design)
    for a in arms:
        decisions[a][res["arms"][a]["decision"].split(":")[0]] += 1
    cuped_gain.append(1 - res["arms"]["C_two_step"]["primary"]["se_used"] ** 2 / res["arms"]["C_two_step"]["primary"]["se_raw"] ** 2)
reliability = pd.DataFrame(decisions).fillna(0).astype(int).T
reliability["correct decision"] = [reliability.loc["B_fewer_fields"].get("DON'T SHIP", 0) / 300,
                                   reliability.loc["C_two_step"].get("SHIP", 0) / 300]
print(f"CUPED variance reduction on completion: {np.mean(cuped_gain):.1%}")
reliability

# %% [markdown]
# * The plan **never ships B**: its cancellation harm is caught in 100% of runs.
# * It **ships C in ~84% of runs**. C is put on HOLD in ~10% of runs, and in ~6% a false
#   harm alarm blocks it. That 6% is the guardrail's own 5% false positive rate, which is
#   the price of a sensitive smoke detector. Across all runs, it never ships the harmful arm. That's the joint power of passing the primary test
#   *and* proving non-inferiority on the guardrail, so the 4-week, guardrail-sized design
#   is doing its job.
# * **CUPED** helps only a little here (~5% variance reduction), because pre-period visits
#   predict a 0/1 completion only weakly. A stronger covariate, such as a past booking
#   propensity score, would do more. This is worth saying honestly in the pre-registration.

# %%
summary = {"plan": p, "readout": readout.reset_index().to_dict(orient="records"),
           "reliability": reliability.reset_index().to_dict(orient="records"),
           "cuped_variance_reduction": float(np.mean(cuped_gain)), "srm": final["srm"]}
(ROOT / "reports" / "booking_form_results.json").write_text(json.dumps(summary, indent=2, default=float))

"""Booking-form experiment: a user-level simulator and the pre-registered analysis.

Each simulated visitor is a real booking from the Hotel Booking Demand dataset
(segment, lead time, repeat-guest flag, nightly rate, nights, and whether it was
cancelled). That anchors cancellation rates and booking values to reality; the
form-completion layer on top is an explicit, documented assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

import abkit

SEGMENTS = ("Direct", "Online TA", "Corporate", "Offline TA/TO")


@dataclass
class ArmEffect:
    completion_odds: float = 1.0   # multiplicative effect on odds of completing the form
    cancel_pp: float = 0.0         # absolute change in cancellation rate among completers


@dataclass
class Design:
    baseline_completion: float = 0.35
    rel_mde: float = 0.05
    alpha: float = 0.05
    power: float = 0.80
    daily_form_starts: int = 4_000
    baseline_cancel: float = 0.33    # cancellation among completed bookings (hotel data)
    cancel_margin_pp: float = 0.015  # non-inferiority margin for the cancellation guardrail
    guardrail_power: float = 0.80
    arms: tuple = ("A_control", "B_fewer_fields", "C_two_step")
    control: str = "A_control"
    experiment: str = "booking_form_v1"
    extras: dict = field(default_factory=dict)


def load_bookings(con) -> pd.DataFrame:
    """Individual/corporate bookings with a positive rate and at least one night."""
    return con.sql(f"""
        SELECT market_segment AS segment, lead_time, is_repeated_guest, is_canceled,
               adr * (stays_in_weekend_nights + stays_in_week_nights) AS booking_value
        FROM hotels
        WHERE market_segment IN {SEGMENTS} AND adr > 0
          AND stays_in_weekend_nights + stays_in_week_nights > 0
    """).df()


def simulate(
    bookings: pd.DataFrame,
    n_users: int,
    effects: dict[str, ArmEffect],
    design: Design,
    days: int,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arms = tuple(effects)
    users = [f"s{seed}-u{i}" for i in range(n_users)]
    arm = np.array([abkit.assign(u, design.experiment, arms) for u in users])
    b = bookings.sample(n_users, replace=True, random_state=seed).reset_index(drop=True)

    # Pre-period engagement: visits to the site in the 30 days before enrolment.
    pre_visits = rng.poisson(0.5 + 3.0 * b.is_repeated_guest + b.lead_time / 120)
    base_logit = 0.9 * np.log1p(pre_visits) + 0.4 * b.segment.isin(["Direct", "Corporate"])

    # Calibrate the intercept so the control completion rate equals the design baseline.
    target = design.baseline_completion
    intercept = optimize.brentq(lambda a: special.expit(a + base_logit).mean() - target, -10, 10)
    log_odds = np.array([np.log(effects[a].completion_odds) for a in arm])
    completed = rng.random(n_users) < special.expit(intercept + base_logit + log_odds)

    # Cancellation: the booking's real outcome, plus any harm the variant causes.
    base_cancel = bookings.is_canceled.mean()
    extra = np.array([effects[a].cancel_pp for a in arm]) / (1 - base_cancel)
    canceled = completed & (b.is_canceled.astype(bool) | (rng.random(n_users) < extra))

    sessions = 1 + rng.poisson(0.4 + 0.5 * pre_visits)
    return pd.DataFrame({
        "user_id": users,
        "arm": arm,
        "enrol_day": rng.integers(1, days + 1, n_users),
        "segment": b.segment,
        "pre_visits": pre_visits,
        "sessions": sessions,
        "completed": completed.astype(int),
        "canceled": canceled.astype(int),
        "revenue": np.where(completed & ~canceled, b.booking_value, 0.0),
    })


def plan(design: Design) -> dict:
    """Sample size, duration and interim-look boundaries for the pre-registration."""
    k = len(design.arms) - 1
    z_crit = abkit.dunnett_critical_z(k, design.alpha)
    alpha_adj = 2 * stats.norm.sf(z_crit)  # per-comparison alpha equivalent to Dunnett
    n_per_arm, _ = abkit.sample_size_proportions(
        design.baseline_completion, design.rel_mde, alpha=alpha_adj, power=design.power, relative=True
    )
    n_bonf, _ = abkit.sample_size_proportions(
        design.baseline_completion, design.rel_mde, alpha=design.alpha / k, power=design.power, relative=True
    )
    # Guardrail: to *show* cancellation is no worse than +margin when the true change is 0,
    # a one-sided test at alpha needs this many completers per arm; divide by the
    # completion rate to get visitors.
    pc = design.baseline_cancel
    completers, _ = abkit.sample_size_means(
        sd=np.sqrt(pc * (1 - pc)), mde=design.cancel_margin_pp, alpha=design.alpha,
        power=design.guardrail_power, two_sided=False,
    )
    n_guardrail = int(np.ceil(completers / design.baseline_completion))
    n_primary = n_per_arm
    n_per_arm = max(n_primary, n_guardrail)
    per_arm_daily = design.daily_form_starts / len(design.arms)
    raw_days = int(np.ceil(n_per_arm / per_arm_daily))
    days = max(14, int(np.ceil(raw_days / 7) * 7))  # whole weeks, at least two business cycles
    looks = days // 7
    obf = abkit.obrien_fleming_boundaries(looks, alpha=alpha_adj)
    return {
        "comparisons": k,
        "dunnett_z": z_crit,
        "alpha_per_comparison": alpha_adj,
        "n_per_arm_primary": n_primary,
        "n_per_arm_primary_bonferroni": n_bonf,
        "completers_per_arm_guardrail": completers,
        "n_per_arm_guardrail": n_guardrail,
        "binding_constraint": "guardrail" if n_guardrail > n_primary else "primary",
        "n_per_arm": n_per_arm,
        "n_total": n_per_arm * len(design.arms),
        "days_needed_raw": raw_days,
        "days_planned": days,
        "weekly_looks": looks,
        "obf_z_boundaries": obf["z_boundaries"],
        "users_at_planned_end": int(per_arm_daily * days),
    }


def analyse(df: pd.DataFrame, design: Design, use_cuped: bool = True) -> dict:
    """The pre-registered analysis: SRM gate, primary with Dunnett, guardrails, decision."""
    z_crit = abkit.dunnett_critical_z(len(design.arms) - 1, design.alpha)
    counts = df.arm.value_counts().reindex(design.arms).fillna(0).values
    srm = abkit.srm_test(counts)

    theta = None
    if use_cuped:
        theta = np.cov(df.pre_visits, df.completed)[0, 1] / df.pre_visits.var(ddof=1)
    ctrl = df[df.arm == design.control]
    out = {"srm": srm, "cuped_theta": theta, "arms": {}}

    for arm in design.arms:
        if arm == design.control:
            continue
        trt = df[df.arm == arm]
        # Primary: completion rate, Dunnett-adjusted CI (optionally CUPED-adjusted)
        if use_cuped:
            mean_x = df.pre_visits.mean()
            yc = ctrl.completed - theta * (ctrl.pre_visits - mean_x)
            yt = trt.completed - theta * (trt.pre_visits - mean_x)
        else:
            yc, yt = ctrl.completed, trt.completed
        se = np.sqrt(yc.var(ddof=1) / len(yc) + yt.var(ddof=1) / len(yt))
        diff = yt.mean() - yc.mean()
        z = diff / se
        primary = {
            "control": ctrl.completed.mean(), "treatment": trt.completed.mean(),
            "diff": diff, "rel_lift": diff / ctrl.completed.mean(), "z": z,
            "p_value": 2 * stats.norm.sf(abs(z)),
            "ci_low": diff - z_crit * se, "ci_high": diff + z_crit * se,
            "significant": abs(z) > z_crit,
            "se_raw": float(np.sqrt(ctrl.completed.var() / len(ctrl) + trt.completed.var() / len(trt))),
            "se_used": float(se),
        }

        # Guardrail 1: cancellation among completers, one-sided non-inferiority
        cc, ct = ctrl[ctrl.completed == 1], trt[trt.completed == 1]
        canc = abkit.two_proportion_ztest(cc.canceled.sum(), len(cc), ct.canceled.sum(), len(ct),
                                          alpha=2 * design.alpha, metric="cancellation")
        cancel_ok = canc.ci_high < design.cancel_margin_pp  # 95% one-sided upper bound
        cancel_harm = canc.ci_low > 0                       # 95% one-sided: cancellations really rose

        # Guardrail 2: revenue per visitor (user-level mean) and per session (ratio, delta method)
        rpv = abkit.welch_ttest(ctrl.revenue, trt.revenue, metric="revenue_per_visitor")
        rps = abkit.ratio_metric_test(ctrl.revenue, ctrl.sessions, trt.revenue, trt.sessions,
                                      metric="revenue_per_session")
        revenue_harm = rpv.significant() and rpv.diff < 0

        wins = primary["significant"] and diff > 0
        if wins and cancel_ok and not revenue_harm:
            decision = "SHIP"
        elif wins and (cancel_harm or revenue_harm):
            decision = "DON'T SHIP: primary wins but a guardrail shows harm, iterate on the design"
        elif wins:
            decision = "HOLD: primary wins but guardrail safety isn't proven, extend or re-test"
        elif primary["significant"]:
            decision = "KILL: hurts completion"
        else:
            decision = "NO DECISION: inconclusive, keep control"

        out["arms"][arm] = {
            "primary": primary,
            "cancellation": {**canc.to_dict(), "margin": design.cancel_margin_pp, "non_inferior": cancel_ok},
            "revenue_per_visitor": rpv.to_dict(),
            "revenue_per_session": rps.to_dict(),
            "decision": decision,
        }
    return out

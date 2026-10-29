"""Experimentation Lab: a sample size calculator, results analyser and peeking simulator.

    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

import abkit
from abkit import viz

st.set_page_config(page_title="Experimentation Lab", page_icon="🧪", layout="wide")
st.title("Experimentation Lab")
st.caption("Design, analyse and stress-test A/B experiments. Powered by `abkit`, a tested open-source stats toolkit.")

tab_size, tab_analyse, tab_peek = st.tabs(["Sample size", "Analyse a test", "Peeking simulator"])

# ---------------------------------------------------------------- sample size
with tab_size:
    c1, c2, c3 = st.columns(3)
    base = c1.number_input("Baseline conversion rate (%)", 0.1, 99.0, 35.0, 0.5) / 100
    mde = c2.number_input("Minimum detectable effect (relative %)", 0.5, 100.0, 5.0, 0.5) / 100
    daily = c3.number_input("Eligible users per day", 10, 10_000_000, 4_000, 100)
    c4, c5, c6 = st.columns(3)
    alpha = c4.select_slider("Significance level α", [0.01, 0.025, 0.05, 0.1], 0.05)
    power = c5.select_slider("Power", [0.7, 0.8, 0.85, 0.9, 0.95], 0.8)
    n_variants = c6.number_input("Treatment arms (vs one control)", 1, 6, 1)

    z = abkit.dunnett_critical_z(int(n_variants), alpha)
    alpha_adj = 2 * stats.norm.sf(z)
    if base * (1 + mde) >= 1:
        st.error("Baseline × (1 + MDE) must stay below 100%.")
    else:
        n, _ = abkit.sample_size_proportions(base, mde, alpha=alpha_adj, power=power, relative=True)
        arms = n_variants + 1
        days = int(np.ceil(n * arms / daily))
        m1, m2, m3 = st.columns(3)
        m1.metric("Users per arm", f"{n:,}")
        m2.metric("Total users", f"{n * arms:,}")
        m3.metric("Duration", f"{days} days", help="Round up to whole weeks in practice, and run at least 14 days.")
        if n_variants > 1:
            st.caption(f"Dunnett correction for {n_variants} comparisons: per-comparison α = {alpha_adj:.4f} (critical |z| = {z:.3f}).")

        mdes = np.linspace(max(mde / 3, 0.005), mde * 3, 60)
        mdes = mdes[base * (1 + mdes) < 1]
        curve = pd.DataFrame({
            "Relative MDE (%)": mdes * 100,
            "Users per arm": [abkit.sample_size_proportions(base, m, alpha=alpha_adj, power=power, relative=True)[0]
                              for m in mdes],
        })
        line = alt.Chart(curve).mark_line(color=viz.CONTROL, strokeWidth=2).encode(
            x="Relative MDE (%):Q", y=alt.Y("Users per arm:Q", scale=alt.Scale(type="log")),
            tooltip=[alt.Tooltip("Relative MDE (%):Q", format=".1f"), alt.Tooltip("Users per arm:Q", format=",.0f")],
        )
        dot = alt.Chart(pd.DataFrame({"Relative MDE (%)": [mde * 100], "Users per arm": [n]})).mark_point(
            filled=True, size=90, color=viz.CONTROL).encode(
            x="Relative MDE (%):Q", y="Users per arm:Q",
            tooltip=[alt.Tooltip("Relative MDE (%):Q", format=".1f"), alt.Tooltip("Users per arm:Q", format=",.0f")])
        st.altair_chart((line + dot).properties(title="Smaller effects need far more users", height=280),
                        width="stretch")

# ---------------------------------------------------------------- analyse
with tab_analyse:
    st.markdown("Enter counts, or upload a CSV with one row per user and columns **`group`** and **`converted`** (0/1).")
    up = st.file_uploader("CSV (optional)", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        if not {"group", "converted"} <= set(raw.columns):
            st.error("The CSV needs `group` and `converted` columns.")
            st.stop()
        groups = sorted(raw.group.unique())
        if len(groups) != 2:
            st.error(f"Expected exactly 2 groups, found {len(groups)}: {groups}")
            st.stop()
        ctrl_name = st.selectbox("Control group", groups)
        trt_name = [g for g in groups if g != ctrl_name][0]
        agg = raw.groupby("group").converted.agg(["sum", "count"])
        nc, xc = int(agg.loc[ctrl_name, "count"]), int(agg.loc[ctrl_name, "sum"])
        nt, xt = int(agg.loc[trt_name, "count"]), int(agg.loc[trt_name, "sum"])
    else:
        ctrl_name, trt_name = "control", "treatment"
        c1, c2, c3, c4 = st.columns(4)
        nc = c1.number_input("Control users", 1, value=44_700)
        xc = c2.number_input("Control conversions", 0, value=8_502)
        nt = c3.number_input("Treatment users", 1, value=45_489)
        xt = c4.number_input("Treatment conversions", 0, value=8_279)
        st.caption("Defaults are the Cookie Cats 7-day retention numbers (gate_30 vs gate_40).")

    if xc > nc or xt > nt:
        st.error("Conversions can't exceed users.")
    else:
        srm = abkit.srm_test([nc, nt])
        if srm["srm_detected"]:
            st.error(f"⚠ Sample ratio mismatch (p = {srm['p_value']:.2g}). The split isn't 50/50. "
                     "Fix assignment or logging before trusting any result.")
        else:
            st.success(f"✓ No sample ratio mismatch (p = {srm['p_value']:.3f}).")

        r = abkit.two_proportion_ztest(xc, nc, xt, nt)
        b = abkit.beta_binomial_ab(xc, nc, xt, nt, draws=100_000)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{ctrl_name}", f"{r.control:.2%}")
        m2.metric(f"{trt_name}", f"{r.treatment:.2%}", f"{r.diff * 100:+.2f}pp ({r.rel_lift:+.1%})")
        m3.metric("p-value", f"{r.p_value:.4f}", "significant" if r.significant() else "not significant",
                  delta_color="off")
        m4.metric(f"P({trt_name} better)", f"{b['prob_treatment_better']:.1%}")
        st.markdown(
            f"**95% CI for the difference:** {r.ci_low * 100:+.2f}pp to {r.ci_high * 100:+.2f}pp  ·  "
            f"**Expected loss if you ship {trt_name}:** {b['expected_loss_treatment'] * 100:.3f}pp  ·  "
            f"**if you keep {ctrl_name}:** {b['expected_loss_control'] * 100:.3f}pp"
        )
        post = pd.DataFrame({
            "rate (%)": np.concatenate([b["posterior_control"][:20_000], b["posterior_treatment"][:20_000]]) * 100,
            "arm": [ctrl_name] * 20_000 + [trt_name] * 20_000,
        })
        chart = alt.Chart(post).transform_density("rate (%)", groupby=["arm"], as_=["rate (%)", "density"]).mark_area(
            opacity=0.6).encode(
            x=alt.X("rate (%):Q", scale=alt.Scale(zero=False)), y=alt.Y("density:Q", axis=None),
            color=alt.Color("arm:N", scale=alt.Scale(domain=[ctrl_name, trt_name], range=[viz.CONTROL, viz.TREATMENT])),
            tooltip=["arm:N", alt.Tooltip("rate (%):Q", format=".2f")],
        )
        st.altair_chart(chart.properties(title="Posterior belief about each arm's true rate", height=260),
                        width="stretch")

# ---------------------------------------------------------------- peeking
with tab_peek:
    st.markdown("Simulate **A/A tests** (no real difference) and watch what daily peeking does to the false positive rate.")
    c1, c2, c3 = st.columns(3)
    days = c1.slider("Days (one look per day)", 2, 60, 20)
    per_day = c2.number_input("Users per arm per day", 100, 100_000, 1_000, 100)
    sims = c3.select_slider("Simulated experiments", [500, 1000, 2000, 5000], 2000)
    rng = np.random.default_rng(0)
    n = per_day * np.arange(1, days + 1)
    ca = rng.binomial(per_day, 0.2, (sims, days)).cumsum(1)
    cb = rng.binomial(per_day, 0.2, (sims, days)).cumsum(1)
    pool = (ca + cb) / (2 * n)
    z = (cb / n - ca / n) / np.sqrt(pool * (1 - pool) * 2 / n)
    bounds = np.array(abkit.obrien_fleming_boundaries(days)["z_boundaries"])
    res = pd.DataFrame({
        "Day": np.tile(np.arange(1, days + 1), 2),
        "False positive rate (%)": np.concatenate([
            np.maximum.accumulate(np.abs(z) > 1.96, axis=1).mean(0),
            np.maximum.accumulate(np.abs(z) > bounds, axis=1).mean(0),
        ]) * 100,
        "Method": ["Naive daily peeking"] * days + ["O'Brien–Fleming"] * days,
    })
    m1, m2 = st.columns(2)
    m1.metric("Naive peeking: false positive rate", f"{res[res.Method.str.startswith('Naive')].iloc[-1, 1]:.1f}%",
              help="Each individual look promises 5%. Repeated looks compound it.")
    m2.metric("O'Brien–Fleming: false positive rate", f"{res[res.Method.str.startswith('O')].iloc[-1, 1]:.1f}%")
    chart = alt.Chart(res).mark_line(point=True, strokeWidth=2).encode(
        x="Day:Q", y="False positive rate (%):Q",
        color=alt.Color("Method:N", scale=alt.Scale(domain=["Naive daily peeking", "O'Brien–Fleming"],
                                                    range=[viz.TREATMENT, viz.CONTROL])),
        tooltip=["Method:N", "Day:Q", alt.Tooltip("False positive rate (%):Q", format=".1f")],
    )
    rule = alt.Chart(pd.DataFrame({"y": [5]})).mark_rule(strokeDash=[4, 3], color=viz.INK_2).encode(y="y:Q")
    st.altair_chart((chart + rule).properties(height=300), width="stretch")

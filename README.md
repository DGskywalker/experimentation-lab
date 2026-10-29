# Experimentation Lab

An end-to-end A/B testing project:

1. A tested statistics library, **`abkit`**.
2. An audit of a real 90k-user experiment.
3. Simulations that verify the methods' error rates.
4. A pre-registered booking-form experiment, rehearsed in a simulator calibrated on 119k
   real hotel bookings.
5. A 588k-user marketing test analysed segment by segment.
6. Uplift modelling on two randomised datasets, including 1M users from Criteo's 14M-user
   ad experiment.

It uses only free tools and public data. No API keys are needed.

## Headline results

| | Finding |
|---|---|
| **Cookie Cats audit** | Moving the gate from level 30 to 40 **cuts 7-day retention by 0.82pp** (19.02% → 18.20%; 95% bootstrap CI −1.32 to −0.31pp; Holm-adjusted p = 0.005; P(gate_40 better) = 0.07%). **Keep gate 30.** |
| **A/A calibration** | 10,000 null experiments → false positive rate **5.2%** |
| **Peeking** | Daily peeking over 20 days → **24% false positives**. O'Brien–Fleming boundaries bring it back to **≤ 5%** |
| **CUPED** | With a pre-period covariate (ρ = 0.76): **58% variance reduction**, so 58% fewer users needed |
| **Booking-form design** | The **cancellation guardrail, not the primary metric, sets the sample size** (34.7k vs 14.0k per arm). Sizing for the primary metric alone would have shipped a variant that raises cancellations by 3pp |
| **Rehearsal (300 runs)** | Harmful variant never shipped. Safe variant shipped in 84% of runs |
| **Marketing campaign (588k users)** | Ads lift conversion 1.79% → 2.55% (+43%), about **4,300 incremental conversions**. The result holds after Cochran–Mantel–Haenszel adjustment for an uneven day mix. **All of the lift comes from users who saw 28+ ads.** 8 of 24 "winning hours" shrink to 1 after Benjamini–Hochberg correction |
| **Uplift: Criteo (1M users)** | Showing ads to the X-learner's top 30% of users captures **86% of incremental visits**, **2.9× more visits per user treated** than showing ads to everyone. The classic response model is a strong baseline; uplift models pull ahead at tight budgets |
| **Uplift: Hillstrom (64k)** | No model reliably beats random targeting (Qini ≈ 0.03 over 5 splits). Every customer segment responds, so **e-mail everyone**. Boosted learners overfit the weak signal |

| | |
|---|---|
| ![](reports/figures/cc_effect_ci.png) | ![](reports/figures/sim_peeking.png) |
| ![](reports/figures/bf_sample_size.png) | ![](reports/figures/bf_readout.png) |
| ![](reports/figures/mk_segments.png) | ![](reports/figures/up_criteo_qini.png) |

## What's inside

```
abkit/                     tested stats toolkit
  power.py                 sample size, power, MDE (proportions & means, unequal splits)
  tests.py                 z-test, Welch t, Mann–Whitney, Wilson CI, chunked bootstrap
  srm.py                   sample-ratio-mismatch check
  assignment.py            sticky, salted SHA-256 bucketing (what a production SDK does)
  variance.py              CUPED, delta method for ratio metrics
  sequential.py            O'Brien–Fleming boundaries (exact recursive integration), mSPRT
  bayes.py                 Beta–Binomial: P(B > A), expected loss, credible intervals
  multiple.py              Holm, Benjamini–Hochberg, Dunnett critical values
  uplift.py                S/T/X meta-learners, Qini curve & coefficient, uplift@k, bootstrap CIs
labsim/booking.py          booking-funnel simulator + the pre-registered analysis
notebooks/
  01_cookie_cats_audit     trust → power → test → estimate → decide
  02_simulations           A/A, peeking (naive vs OBF vs mSPRT), CUPED
  03_booking_form_design   sizing, weekly looks, readout, 300-run reliability check
  04_marketing_segments    unequal allocation, balance checks, CMH/Breslow–Day, 37 segments + BH
  05_uplift_modeling       Hillstrom & Criteo: meta-learners vs response model, Qini, robustness, budget
docs/preregistration_booking_form.md   locked experiment plan with decision rules
app/streamlit_app.py       sample size calculator · results analyser · peeking simulator
tests/                     27 tests against statsmodels, SciPy, published tables and known-truth data
```

Each `abkit` function is checked against an independent reference:
- statsmodels for z-tests, Wilson intervals, Holm and BH
- SciPy for Welch's test
- Fleiss's sample-size tables
- published Dunnett critical values
- the gsDesign/ldbounds O'Brien–Fleming boundaries (matched to 4 decimals)

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q                                   # 27 tests
python scripts/fetch_data.py                # downloads CSVs (incl. 311 MB Criteo), builds data/lab.duckdb
cd notebooks && for nb in 0*.py; do jupytext --to ipynb --execute "$nb"; done && cd ..
streamlit run app/streamlit_app.py          # the interactive app
```

Notebooks are stored as plain `.py` files (jupytext percent format), so diffs stay readable.
Running the loop above regenerates the `.ipynb` files with outputs, plus everything in `reports/`.

## Deploy for free

* **GitHub Actions** (`.github/workflows/ci.yml`) runs the tests, downloads the data,
  executes all three notebooks, and uploads the reports on every push.
* **Streamlit Community Cloud**: connect the repo at share.streamlit.io and set the main
  file to `app/streamlit_app.py`. The app needs no data download.

## Data

* **Cookie Cats**: 90,189 players randomised to `gate_30` / `gate_40`. Originally
  published by Tactile Entertainment via DataCamp and Kaggle; mirrored on GitHub.
* **Hotel Booking Demand**: 119,390 bookings (Antonio, Almeida & Nunes, *Data in Brief*,
  2019), via TidyTuesday.
* **Marketing A/B Testing**: 588,101 users, ad vs PSA (Kaggle, mirrored on GitHub).
* **Hillstrom MineThatData e-mail challenge**: 64,000 customers, 3 arms (Kevin Hillstrom, 2008).
* **Criteo Uplift v2.1**: 13,979,592 users, 85/15 split (Diemert et al., 2018), from
  Criteo's public Hugging Face repo. A reproducible 1M-row random sample is used.

## Methods notes and honest limits

* Cookie Cats shows a borderline SRM (p = 0.009, above the 0.001 alarm). It's noted as
  a caveat, and it doesn't change the conclusion.
* The booking-form completion rate (35%) and traffic (4k/day) are **assumptions**. The
  hotel data has no funnel. Cancellation rates and booking values come from real bookings.
* CUPED gives only ~5% on the booking simulation, because pre-visits weakly predict a
  0/1 outcome. It's reported as is, not tuned to look better.

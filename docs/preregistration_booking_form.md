# Pre-registration: Booking-form simplification (`booking_form_v1`)

| | |
|---|---|
| **Status** | Draft for review. Locked before launch; no changes after the first user is enrolled |
| **Owner** | Experimentation / Growth analytics |
| **Planned run** | 28 days (4 full weeks), weekly interim looks |
| **Analysis code** | `labsim/booking.py` → `analyse()`; rehearsed in `notebooks/03_booking_form_design.ipynb` |

## 1. Problem

The hotel booking form has 12 fields. Session recordings and field-level analytics suggest
visitors abandon at the phone and address fields. Every abandoned form is lost revenue.
Every field we remove is also information the hotel no longer has, and a phone number is
how the hotel reconfirms a reservation.

## 2. Hypotheses

* **H1 (B_fewer_fields):** Removing four optional fields (phone, company, address line 2,
  "how did you hear about us") increases form completion.
* **H2 (C_two_step):** Splitting the same 12 fields into two steps increases completion,
  because the first step looks short and the user has already committed by the second.
* **H0 for each:** Completion is unchanged.
* **Guardrail hypothesis:** Neither variant increases cancellations by 1.5 percentage points or more.

## 3. Variants and assignment

| Arm | Description | Traffic |
|---|---|---|
| A_control | Current 12-field form | 1/3 |
| B_fewer_fields | 8 fields | 1/3 |
| C_two_step | 12 fields across 2 steps | 1/3 |

* **Unit of randomisation:** visitor (persistent first-party ID), not session. Otherwise a
  returning visitor could see two different forms.
* **Assignment:** `sha256("booking_form_v1:" + visitor_id)` mapped to a bucket
  (`abkit.assign`). It's sticky and independent of other experiments.
* **Enrolment trigger:** the visitor *starts* the form (first field focused). Only exposed
  users count, which keeps the metric sensitive.
* **Exclusions:** internal IP ranges, known bots (user-agent and > 50 form starts/day), and staff accounts.

## 4. Metrics

| Type | Metric | Definition | Direction |
|---|---|---|---|
| **Primary** | Form completion rate | visitors who submit ÷ visitors who start | ↑ |
| Guardrail | Cancellation rate | cancelled ÷ completed bookings, 30-day window | must not rise ≥ 1.5pp |
| Guardrail | Revenue per visitor | realised (non-cancelled) booking value ÷ visitors | must not fall significantly |
| Guardrail | Revenue per session | ratio metric; delta-method variance (visitors have many sessions) | monitor |
| Guardrail | Page latency p95 | client-side form render time | must not rise > 100 ms |
| Secondary | Time to complete, field error rate | diagnostic | explanatory only |

Baselines from 97,576 real bookings (Hotel Booking Demand dataset): **cancellation 32.7%**,
**mean booking value $389**. The 35% completion rate is an assumption and must be checked
against the first week of control data.

## 5. Sample size and duration

| Requirement | Calculation | Visitors per arm |
|---|---|---|
| Primary: +5% relative lift (35.0% → 36.75%), 80% power | two-proportion test; Dunnett-adjusted α = 0.027 per comparison (critical z = 2.212) | 14,009 |
| Guardrail: prove cancellation increase < 1.5pp, 80% power | one-sided non-inferiority, α = 0.05, 12,151 completers ÷ 35% completion | **34,718** |

**The guardrail sets the size: 34,718 visitors per arm, 104,154 in total.** At 4,000 form
starts a day that takes 27 days, rounded up to **28 days**. Running whole weeks covers
weekday/weekend cycles and lets novelty effects wear off.

## 6. Analysis plan

1. **SRM gate.** Chi-square test of the arm counts against 1:1:1. If p < 0.001, stop and
   debug assignment. No metric is read.
2. **Primary.** Difference in completion rate per arm vs. control, with a CUPED adjustment
   using pre-period site visits (θ estimated on pooled data). Report Dunnett-adjusted 95% CIs.
3. **Guardrails.** Cancellation: one-sided 95% upper bound vs. the +1.5pp margin.
   Revenue per visitor: Welch t-test. Revenue per session: delta-method z-test.
4. **Interim looks.** End of weeks 1–4, using O'Brien–Fleming boundaries at the
   Dunnett-adjusted α: |z| > **4.80, 3.30, 2.64, 2.25**.
   * Stopping early **for success** is allowed only if the cancellation guardrail is also
     non-inferior at that look.
   * Stopping early **for harm** is always allowed: a guardrail significantly worse, or
     completion below the control's boundary.
5. **No other peeking at decision metrics.** Dashboards show guardrail health only.

## 7. Decision rules (fixed in advance)

| Primary | Cancellation guardrail | Revenue guardrail | Decision |
|---|---|---|---|
| Significant ↑ | Upper bound < +1.5pp | Not significantly ↓ | **SHIP** |
| Significant ↑ | Significant ↑ (lower bound > 0) | any | **DON'T SHIP**: iterate on the design |
| Significant ↑ | Inconclusive (straddles margin) | Not significantly ↓ | **HOLD**: extend or re-test; never relax the margin |
| Significant ↓ | any | any | **KILL** |
| Not significant | any | any | **Keep control** |

If both B and C qualify to ship, the arm with the higher realised revenue per visitor wins.

## 8. Rehearsal results (simulated, 300 runs)

The design was rehearsed on a simulator calibrated on the hotel data. In the simulation,
B lifts completion but raises cancellations by 3pp, and C lifts completion with no harm.

| Arm | Truth | SHIP | HOLD | DON'T SHIP |
|---|---|---|---|---|
| B_fewer_fields | harmful | 0% | 0.3% | 99.7% |
| C_two_step | safe win | 84% | 10% | 6% (the guardrail's own false positive rate) |

The design never shipped the harmful variant. A design sized for the primary metric
alone (2 weeks) would have declared B the winner.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Lead quality falls without phone numbers | Cancellation guardrail. Add no-show rate when hotel PMS data is joined |
| Novelty or primacy effect | 4 full weeks. Plot the effect by week of enrolment |
| Visitors switch devices and see two variants | Logged-in ID where available. Measure the cross-arm contamination rate |
| Bots inflate form starts | Exclusion rules in §3. SRM gate |
| Seasonality (holiday bookings) | Run outside holiday peaks, or extend to cover them in every arm |
| CUPED gives little lift (~5%) because pre-visits weakly predict completion | Accepted. Explore a booking-propensity covariate for the next test |

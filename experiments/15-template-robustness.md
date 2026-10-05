# 15: Does the size of Choice's step depend on how the base rate is worded?

## Pre-registration

Written 2026-10-04 ~20:00 PDT, before any call for this experiment.

**Why.** Experiment 03 found that moving a stated base rate from 45% to 55% raises Choice
by 0.83–0.98. An independent preregistered study (github.com/yanjn1388/jev-bayes,
2026-09-29) found 0.489 with different wording. Both are far above the correct +0.10, but
the size differs. This experiment tests how much wording matters, holding everything else
fixed.

**Design.**
- **Domains and options.** The same three domains, option keys, descriptions, Choice
  instructions and Noul questions as experiment 03 (`natural.DOMAINS`).
- **Three wordings of the same base rate p:**
  - `count`: the original "of the 20 most similar …, k were …" (k = 20p).
  - `percent`: "… p% of … like this one are …".
  - `probability`: "… the probability that … is 0.pp."
- **Rates.** p ∈ {0.35, 0.45, 0.50, 0.55, 0.65}.
- **Queries.** Choice in both option orders, plus one Noul call (yes and no questions).
  Two repeats (records 301–302).
- **Size.** 3 × 3 × 5 × 3 × 2 = 270 calls.

**Metrics** (per template; Choice averaged over the two orders):
- the 45→55 jump in P(first-named outcome);
- the 35→65 jump;
- mean absolute error against p for Choice and for raw Noul P(yes).

**Predictions:**
- **P1.** In every wording, Choice's 45→55 jump is above 0.3, three times the correct
  0.10.
- **P2.** In every wording, Noul's mean absolute error against p is at most 0.05.
- **P3** (descriptive). Report how much the jump varies across wordings. No direction is
  predicted.

**Falsifier.** If any wording gives a Choice jump of at most 0.3, the step is a property
of particular wordings, not of Choice. The paper then says so.

## Results

(Filled in after the run. Nothing above this line is edited after the run.)

Run 2026-10-04: 270 calls, all `jev-1.13.0`, 95,028 input tokens (about $0.004).

**Outputs:** `data/15-template-robustness.jsonl`, `data/15-template-robustness.summary.json`

Choice is the mean of both option orders. P is the mean over the three domains and two
repeats.

| Wording | Choice at 0.35 / 0.45 / 0.50 / 0.55 / 0.65 | Jump 45→55 | Jump 35→65 | Choice error | Noul error |
|---|---|---:|---:|---:|---:|
| count (original) | 0.02 / 0.05 / 0.81 / 0.95 / 0.98 | **0.89** | 0.97 | 0.355 | 0.026 |
| percent | 0.08 / 0.36 / 0.88 / 0.95 / 0.98 | **0.59** | 0.90 | 0.295 | 0.014 |
| probability | 0.10 / 0.34 / 0.93 / 0.98 / 0.99 | **0.64** | 0.88 | 0.310 | 0.024 |

Noul's 45→55 jumps are 0.16, 0.13 and 0.17 (correct: 0.10).

## Verdicts

- **P1 held.** The Choice jump is above 0.3 in every wording (0.59–0.89).
- **P2 held.** Noul's error is at most 0.05 in every wording (0.014–0.026).
- **P3 (descriptive).** Wording changes the size of the step, not its existence:
  - The count wording is the sharpest.
  - With percent and probability wordings, Choice is less extreme just *below* 50% (0.34–0.36 at 45%) but just as saturated above it (0.95–0.98 at 55%).
  - At an exact 50% tie, Choice gives 0.81–0.93 to the outcome the text names first.

## What it means

The 0.83–0.98 jump in experiment 03 is the high end for count wordings. Across wordings,
the jump is 0.59–0.89. jev-bayes's 0.49 is consistent with this range given its different
templates. The paper should report the step as "0.6–0.9 depending on wording; always far
above the correct 0.10".

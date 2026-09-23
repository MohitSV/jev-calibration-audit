# 02: Confidence intervals for the experiment 01 and 03 headlines

**Date:** 2026-09-22 · offline, no new calls · `cd src && ../venv/bin/python -m jevcal.bootstrap_0103`
**Output:** `data/02-bootstrap-0103.json`

**Method.** 1,000 bootstrap resamples of whole repetitions (1–10), drawn jointly across
cells, because the "Record #N" / "Case N" text is shared within a repetition.

| Headline | Point | 95% CI |
|---|---:|---|
| 01: heads pull, no evidence | +0.354 | [0.347, 0.361] |
| 01: heads pull, one 55% witness | +0.063 | [0.055, 0.070] |
| 01: `option_1` label effect, no evidence | +0.027 | [0.020, 0.034] |
| 01: list-position effect, no evidence | −0.015 | [−0.018, −0.010] |
| 01: Noul logit slope | 0.445 | [0.432, 0.459] |
| 01: Choice error vs exact answer (mean abs.) | 0.202 | [0.201, 0.202] |
| 01: Noul error vs exact answer (mean abs.) | 0.073 | [0.071, 0.075] |
| 03: Choice jump, 45%→55% base rate (loan / routing / refund) | 0.90 / 0.98 / 0.83 | ±0.01 each |
| 03: Choice error vs base rate (mean abs.) | 0.27 / 0.30 / 0.27 | ±0.001 |
| 03: Noul error vs base rate (mean abs.) | 0.025 / 0.027 / 0.015 | ±0.001 |
| 03: Choice at the 50% tie | 0.77 / 0.86 / 0.81 | ±0.01 |
| 03: billing L2 ticket, Choice / Noul | 1.00 / 0.83 | [1.0, 1.0] / [0.825, 0.834] |

**Reading.**
- Every headline is far outside repeat noise. That's expected, because the typical
  spread between repeats within a cell is 0.011.
- The small position effect (−0.015) and label effect (+0.027) at zero evidence are
  statistically real but practically negligible next to the +0.354 word effect.

**What these CIs don't cover.** They don't cover variation across templates or
wordings. That is the larger threat to generalization: every item in 01 and 03 was
written by us. Experiments 06 and 06b (real human-written items) are the answer to
that threat, not these intervals.

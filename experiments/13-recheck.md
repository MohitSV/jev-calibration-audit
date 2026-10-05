# 13: Re-check of the core probes, 12 days later

**Date:** 2026-10-04. **Model served:** `jev-1.13.0` on all 289 calls (alias `jev-latest`).
**Cost:** 103,479 input tokens, about $0.004.

## Why

The paper's Jev results come from 22–23 September. Before posting, we wanted to know
whether TypeSafe has changed the served build or its behaviour since then.

## What was run

The two core probe sets, one repeat each, with fresh record numbers (201). Commands from
`src/`:
```
../venv/bin/python -m jevcal.sweep -n 1 --rep-start 201 --out ../experiments/data/13-recheck-evidence-sweep.jsonl
../venv/bin/python -m jevcal.sweep_natural -n 1 --rep-start 201 --out ../experiments/data/13-recheck-natural-evidence.jsonl
```
Each was analysed with its own `--summary` and `--figure` flags pointed at `13-recheck-*`.

## Result: every headline reproduces

| Measure | Original (09-22, n = 10) | Re-check (10-04, n = 1) |
|---|---|---|
| Coin, no evidence: Choice P(heads) | 0.83–0.93 | 0.84–0.92 |
| Word pull, no evidence | +0.354 | +0.365 |
| Word pull after one 55% witness | +0.063 | +0.082 |
| Witness: Choice error vs Bayes | 0.171–0.213 | 0.164–0.204 |
| Witness: Noul error / logit slope | 0.073 / 0.45 | 0.073 / 0.47 |
| Base-rate jump 45%→55% (loan/routing/refund) | 0.90 / 0.98 / 0.83 | 0.89 / 0.98 / 0.81 |
| Choice error vs stated base rate | 0.274 / 0.303 / 0.273 | 0.270 / 0.302 / 0.269 |
| Noul error vs stated base rate | 0.025 / 0.027 / 0.015 | 0.024 / 0.029 / 0.016 |
| Hedged ticket cue L2: Choice / Noul | 1.00 / 0.89 | 1.00 / 0.90 |

Single-repeat estimates are noisier (within-cell SD about 0.011), and all differences
fall inside that noise.

## What it means

The served model and its behaviour on these probes are unchanged from the paper's
measurement window. The paper's Jev numbers can be described as holding for
`jev-1.13.0` as of 2026-10-04.

## Limits

One repeat per cell. This checks the headline probes only, not the repair holdout or the
real-data experiments, which don't need re-running unless the served build changes.

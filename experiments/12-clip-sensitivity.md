# 12: Sensitivity of log-based metrics to clipping exact zeros

**Date:** 2026-09-24. Offline robustness check, no API calls.
**Run:** `cd src && ../venv/bin/python -m jevcal.clip_sensitivity`
**Output:** `data/12-clip-sensitivity.json`

## Why

An independent audit, Meng 2026 (Zenodo 10.5281/zenodo.22935043), showed that Jev's
fitted temperature depends on how exact zeros are clipped before taking logs. Jev's
`Choice` returns exact 0/1 probabilities often:
- ChaosNLI: 274 of 900 Choice entries are exactly 0; Noul has 1.
- DICES: 76 of 350 Choice answers are at exactly 0 or 1; Noul has 0.
- Exp 05 holdout: 126 of 960 raw Choice answers are at exactly 0 or 1.

Every KL, log-loss and temperature figure in experiments 05, 06 and 06b used a clip of
ε = 1e-6. That choice mostly inflates the scores of `Choice`, not `Noul`.

## Results

**KL to human votes** (Choice / Noul, and the ratio between them):

| ε | ChaosNLI | DICES |
|---|---|---|
| 1e-6 (as reported) | 1.150 / 0.193, 5.9× | 0.954 / 0.226, 4.2× |
| 1e-4 | 0.834 / 0.193, 4.3× | 0.722 / 0.226, 3.2× |
| 1e-3 | 0.676 / 0.193, 3.5× | 0.605 / 0.226, 2.7× |
| 1e-2 | 0.507 / 0.193, 2.6× | 0.479 / 0.226, 2.1× |
| 5e-2 | 0.289 / 0.175, 1.6× | 0.307 / 0.212, 1.4× |

**Metrics that don't depend on the clip** (Choice vs Noul):

| Metric | ChaosNLI | DICES |
|---|---|---|
| Brier | 0.203 vs 0.096 (2.1×) | 0.092 vs 0.067 (1.4×) |
| TVD (\|p − p_h\| on DICES) | 0.283 vs 0.205 (1.4×) | 0.249 vs 0.198 (1.3×) |

**Fitted Choice temperature** (clip applied inside the transform; the fitted Choice KL is
in brackets):

| ε | ChaosNLI | DICES |
|---|---|---|
| 1e-6 | T = 5.9 (KL 0.162) | T = 8.6 (KL 0.095) |
| 1e-4 | T = 4.3 (0.159) | T = 6.3 (0.088) |
| 1e-3 | T = 3.4 (0.164) | T = 5.0 (0.084) |
| 1e-2 | T = 2.7 (0.184) | T = 4.3 (0.083) |

Noul's own KL is 0.176 on ChaosNLI and 0.089 on DICES.

**Exp 05 holdout, excess log loss, raw Choice vs normalized Noul:**

| ε | Raw Choice | Normalized Noul | Ratio |
|---|---|---|---|
| 1e-6 | 0.663 | 0.0115 | 58× |
| 1e-4 | 0.521 | 0.0115 | 45× |
| 1e-3 | 0.451 | 0.0115 | 39× |
| 1e-2 | 0.381 | 0.0115 | 33× |

Brier doesn't depend on the clip: 0.0665 vs 0.0034, which is 19×.

## What it changes

- **Direction:** unchanged at every ε and on every metric. Noul is closer to the targets
  than Choice, on real and exact-target data alike.
- **Size of the real-data gap:** "4–6× closer (KL)" holds only at ε = 1e-6. The
  clip-free gap is 1.3–2.1×. The KL gap ranges from 1.4× to 6× depending on ε.
- **Exp 05:** "more than 50× in excess log loss" becomes "33–58×, depending on ε". The
  Brier figure (about 20×) stands as reported.
- **Temperatures:** absolute fitted values depend on ε, so don't compare them across tasks
  ("6.3 vs 9.3 don't transfer" is withdrawn as a claim). That a fitted temperature brings
  Choice level with Noul holds at every ε.

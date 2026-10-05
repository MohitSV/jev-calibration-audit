# 14: Uniform baseline, per-stratum calibration after one temperature, and a monotone repair

**Date:** 2026-10-04. Offline, no API calls.
**Run:** `cd src && ../venv/bin/python -m jevcal.baselines_monotone`
**Output:** `data/14-baselines-monotone.json`

## Why

Independent work raised three challenges to our claims:
- Khosla 2026 (Zenodo 10.5281/zenodo.22971492) found that Noul is about as far from the
  human distribution as a uniform guess on hard items, and that one fitted temperature
  helps hard items but hurts easy ones.
- Porcedda 2026 (arXiv 2609.35342) found that a fitted monotone correction lifts Choice
  substantially on exact-probability items.

## 1. Uniform-guess baseline (Brier, lower is better)

| Data | Uniform | Choice | Normalized Noul |
|---|---:|---:|---:|
| ChaosNLI, all | 0.240 | 0.203 | **0.096** |
| ChaosNLI, hardest bin | **0.058** | 0.321 | 0.135 |
| ChaosNLI, easiest bin | 0.513 | 0.078 | 0.077 |
| DICES, all | 0.071 | 0.092 | **0.067** |
| DICES, hardest bin | **0.003** | 0.153 | 0.109 |
| DICES, easiest bin | 0.186 | 0.011 | 0.017 |

Mean absolute error on DICES: uniform 0.236, Choice 0.249, Noul 0.198.

**Reading.** Noul beats Choice overall, but on the most contested items **neither beats a
uniform guess** (Noul 0.135 vs 0.058 on ChaosNLI's hardest bin; 0.109 vs 0.003 on DICES).
On DICES overall, Noul's lead over a flat 0.5 guess is small (0.067 vs 0.071), and Choice is
worse than flat. Most of Noul's advantage over uniform comes from easy items. Any claim
that Noul is "closer to human spread" has to say this.

## 2. One global temperature, per agreement bin (ChaosNLI)

Gap = model top probability minus human majority share (0 = calibrated).
Fitted T: Choice 5.87, Noul 1.36.

| Human top share | Raw Choice | Choice + T | Raw Noul | Noul + T |
|---|---:|---:|---:|---:|
| 0.45 | +0.347 | +0.039 | +0.138 | +0.083 |
| 0.54 | +0.260 | −0.000 | +0.083 | +0.024 |
| 0.64 | +0.229 | −0.081 | +0.003 | −0.063 |
| 0.78 | +0.088 | −0.186 | −0.105 | −0.171 |
| 0.91 | +0.030 | −0.200 | −0.151 | −0.232 |

**Reading.** One temperature fixes the contested items and overshoots the easy ones, for
**both** Choice and Noul. Khosla's trade-off is real, and it isn't specific to Choice. The
statement "one temperature brings Choice level with Noul" holds for KL, and also in each
bin's KL (see the earlier check in `PROGRESS.md`), but neither is per-stratum calibrated.

## 3. A flexible monotone repair of Choice (exact targets)

Isotonic regression fitted on repetitions 1–10 and scored on the true holdout
(records 101–105, Brier):

| Subset | Raw Choice | Isotonic Choice | Normalized Noul |
|---|---:|---:|---:|
| Overall (960) | 0.0665 | 0.0137 | **0.0034** |
| Witness (570) | 0.0508 | 0.0144 | **0.0053** |
| Stated base rate (390) | 0.0895 | 0.0128 | **0.0006** |

**Reading.** "No monotone rescaling repairs it" is too strong. Temperature scaling does not
(Brier 0.0332 on base rates), but a fully flexible monotone map recovers about 80% of the
Brier gap. It is still 4× worse than normalized Noul overall and 21× worse on stated base
rates. The map is fitted on the same templates as the holdout, so this is optimistic.
Exact 0/1 outputs also collapse many different evidence values into one.

## What it changes in the paper

- Add a uniform baseline to the real-data table and say that neither primitive beats it on
  contested items.
- Say the temperature fix is a global one that both primitives share, and that it leaves
  per-stratum miscalibration.
- Replace "no monotone rescaling" with the temperature and isotonic results above.

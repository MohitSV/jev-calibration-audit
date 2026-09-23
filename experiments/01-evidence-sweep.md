# 01: Evidence dose-response sweep

**Date:** 2026-09-22
**Model:** `jev-latest`, which resolved to `jev-1.13.0` on every call.
Requesting `jev-1.13` returns HTTP 400.
**Calls:** 1,330, using 484,383 input tokens (about $0.02).

## Question

How much evidence does it take to overcome Jev's zero-evidence preference for
"heads" or "option_1", and does Jev track evidence strength correctly on the way?
This was the pre-declared kill check for the v1 framing.

## Design

- **Scenario.** A fair coin is flipped out of sight. Zero or more independent
  witnesses, each with a stated accuracy q, report a side. The Bayesian answer
  is exact: logit P(heads) = Σ ±logit(q).
- **19 evidence conditions:**
  - none;
  - 1 witness at q ∈ {.55, .6, .7, .8, .9, .99}, reporting heads and tails;
  - 2 witnesses at q = .7 who disagree, in both orders;
  - 2 or 3 witnesses at q = .6 who agree.
- **Mirror pairs.** Every condition has a mirror with every report flipped.
  Averaging a condition with its mirror cancels the true answer and leaves only
  Jev's own asymmetry.
- **6 Choice label sets:**
  - `heads/tails` and `tails/heads`: tests list position;
  - `option_1=heads` and `option_1=tails`: tests the key name;
  - neutral keys `kvx/mzr`, with each side assigned both ways.

  Every option has a description ("The coin landed heads up"). No key looks like an
  integer, because JavaScript clients reorder integer-like keys.
- **Noul.** One call asks both "Did it land heads?" and "Did it land tails?".
- **Repeats.** 10 repeats per cell. The irrelevant text `Record #N` is the only
  thing that varies between repeats. The mean standard deviation within a cell is
  0.011, so 10 repeats is enough.

## How to run it

From the project root:
```
cd src && ../venv/bin/python -m jevcal.sweep -n 10 && ../venv/bin/python -m jevcal.analyze
```

**Outputs:**
- Raw calls: `data/01-evidence-sweep.jsonl`
- Summary: `data/01-evidence-sweep.summary.json`
- Figure: `figures/01-evidence-sweep.png`

## Results

**1. Choice behaves like a step function, not a probability.**
The figure's left panel shows it saturating as soon as any evidence appears.

| Evidence | True P(named side) | Choice P(named side), range over label sets | Noul |
|---|---|---|---|
| none | 0.50 | P(heads) 0.83–0.93 | 0.52 |
| 1 witness, 55%, says heads | 0.55 | 0.89–0.96 | 0.55 |
| 1 witness, 55%, says tails | 0.55 | 0.71–0.88 | 0.53 |
| 1 witness, 70%, says heads | 0.70 | 0.97–0.99 | 0.61 |
| 1 witness, 90%, says tails | 0.90 | 0.92–0.99 | 0.78 |
| 2 witnesses, 70%, disagree | 0.50 | P(heads) 0.59–0.74 | 0.505 |

Logit slope against the truth:
- Choice: 1.13–1.42. This overstates how smooth Choice is, since its curve is
  really a step.
- Noul: **0.45**, meaning its answers are compressed toward 0.5.

Mean absolute error against the Bayesian answer:
- Choice: 0.17–0.21.
- Noul: 0.073.

**2. The zero-evidence prior follows the word "heads", not the key name or the
list position (when descriptions are present).**

| Evidence strength \|P−.5\| | Word "heads" | Key "option_1" | Listed first |
|---|---|---|---|
| none (0) | **+0.354** | +0.027 | −0.015 |
| balanced conflict (0) | +0.063 | +0.080 | +0.034 |
| 0.05 (one 55% witness) | +0.063 | −0.043 | −0.034 |
| 0.10 | +0.072 | −0.044 | −0.031 |
| 0.19–0.49 | +0.005 to +0.034 | −0.015 to −0.031 | −0.004 to −0.014 |

The neutral-key label sets show the same heads pull at zero evidence (+0.37 and
+0.40), which confirms it comes from the word, not the key.

In the pilot, `option_1` got 0.84 when the options had **no** descriptions. Here, with
descriptions, the key-name effect is about 0. That matches jev-bench's finding that
descriptions remove the label prior.

**3. Noul is unbiased with no evidence (0.52) and with balanced evidence (0.505).**
Its heads and tails answers miss summing to 1 by an average of 0.083.

## What it means

- **The v1 kill check mostly fired.** One weak sentence of evidence cuts the
  zero-evidence prior by about 80%, from +0.35 to +0.06. What's left is a residual of
  0.03–0.08 near ties. That could still flip decisions right at the boundary, but it
  is not a headline.
- **The bigger finding is the step function.** For workflow automation, "0.9" from
  Choice can mean anything from barely-better-than-chance evidence to near-certain
  evidence. Thresholding Choice at 0.9 does not select confident cases on weak
  evidence.
- Noul is the primitive that actually represents probability. It needs a slope
  correction, which is cheap if the slope is stable.

## Caveats

- The evidence states its reliability as a number ("correct 55% of the time"). Real
  evidence is qualitative. Experiment 03 must test whether the step function survives
  natural-language evidence.
- There is one scenario family (coins). It needs a second domain.
- Everything ran on a single day and a single model build (1.13.0).
- There are no confidence intervals yet. Bootstrap them over repeats. With a
  within-cell SD of 0.011, the effects above 0.03 are well outside noise.

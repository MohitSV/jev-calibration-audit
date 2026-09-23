# Experiment 05: repairing Jev probabilities

## Question

Can Jev 1.13.0's typed outputs be transformed into probabilities that match exact
targets on held-out calls?

The five compared methods were preregistered in `goal.md`:

1. raw Choice;
2. one Noul per option, normalized to sum to one;
3. Noul with a slope-only Platt transform;
4. temperature-scaled Choice;
5. content-free prior division for Choice.

## Split and metrics

The intended new collection (records 101–105) was attempted on 2026-09-22, but every
TypeSafe request returned HTTP 403, error code 1010. The output file had zero rows.
Instead, the ten already-collected repetitions were split without inspecting repair
results: repetitions 1–5 fit all parameters and repetitions 6–10 were held out.

The witness dataset contributes 570 held-out Choice contexts (19 conditions × 6 label
sets × 5 repetitions). The natural base-rate dataset contributes 390 (3 domains × 13
base rates × 2 option orders × 5 repetitions). The subjective cue ladder is excluded
because it has no exact probability target.

Metrics are binary Brier score, soft-target log loss, and Gaussian-soft-bin ECE. For
smooth ECE, each observation distributes unit weight over 21 confidence centers with
Gaussian bandwidth 0.075; bin discrepancies are then mass-weighted.

## Fitted transformations

- Noul logit multiplier: 1.7672, fitted only on experiment 01's witness conditions.
- Choice logit multiplier: 0.3370, equivalent to temperature 2.9678, also fitted only
  on experiment 01.
- Content-free Choice priors: a separate training-set mean baseline logit for each
  label set, or each natural domain × option order.
- Normalized Noul has no fitted parameter: `P(yes) / (P(yes) + P(no))`.

## Held-out results

### Overall (960 observations)

| Method | Brier ↓ | Log loss ↓ | Smooth ECE ↓ |
|---|---:|---:|---:|
| Raw Choice | 0.0687 | 1.2468 | 0.2307 |
| **Normalized Noul** | **0.0034** | **0.5641** | **0.0264** |
| Platt Noul | 0.0064 | 0.5759 | 0.0412 |
| Temperature Choice | 0.0219 | 0.6727 | 0.0638 |
| Prior-divided Choice | 0.0615 | 1.2609 | 0.1943 |

### Witness evidence (570 observations)

| Method | Brier ↓ | Log loss ↓ | Smooth ECE ↓ |
|---|---:|---:|---:|
| Raw Choice | 0.0535 | 0.8087 | 0.1948 |
| Normalized Noul | 0.0053 | 0.5498 | 0.0525 |
| **Platt Noul** | **0.0024** | **0.5405** | **0.0279** |
| Temperature Choice | 0.0135 | 0.5755 | 0.0330 |
| Prior-divided Choice | 0.0499 | 0.8703 | 0.1568 |

### Stated natural base rates (390 observations)

| Method | Brier ↓ | Log loss ↓ | Smooth ECE ↓ |
|---|---:|---:|---:|
| Raw Choice | 0.0909 | 1.8870 | 0.2830 |
| **Normalized Noul** | **0.0007** | **0.5850** | **0.0174** |
| Platt Noul | 0.0123 | 0.6277 | 0.1019 |
| Temperature Choice | 0.0343 | 0.8148 | 0.1501 |
| Prior-divided Choice | 0.0785 | 1.8319 | 0.2491 |

## Interpretation

The proposed decomposition works. Normalizing independently asked Noul probabilities
reduces overall Brier error by 95% relative to raw Choice and works without a fitted
parameter. It is especially strong on explicitly stated base rates.

The evidence-specific Platt slope is best on the witness task but damages the already
accurate natural-base-rate Noul answers. A single global Noul recalibration is therefore
the wrong abstraction: the compression depends on how probability information is
expressed. Temperature scaling substantially improves Choice, but it cannot recover the
information discarded by Choice's step. Content-free prior division barely improves
Brier and worsens log loss; the problem is not a fixed prior that can simply be divided
out.

The practical recommendation is to use per-option Noul plus normalization when a
probability distribution is required. Use Choice for an action selection, not as the
probability estimate supporting that action.

## Limitations

- Repetitions 6–10 are a pre-existing holdout, not the newly collected temporal
  holdout originally planned. They vary the record number and API stochasticity, but
  reuse the same experimental templates.
- Noul observations are repeated across the six Choice label contexts in experiment 01
  so every method is scored on the same 570 contexts. This affects weighting, not the
  per-method prediction values.
- These exact-target synthetic tasks establish probability recovery, not downstream
  accuracy on real decisions. Experiment 06 is the required external test.

## Artifacts

- `src/jevcal/calibrate.py`
- `experiments/data/05-calibration-fixes.summary.json`
- `experiments/figures/05-calibration-fixes.png`

---

## Update 2026-09-22: true holdout plus a missing baseline

This section corrects the result above rather than replacing it.

**What changed:**
- **The API is back.** It returned 200 again. Records 101–105 were then collected as
  a true holdout: 1,445 calls, all on jev-1.13.0.
  - `data/05-heldout-evidence-sweep.jsonl`
  - `data/05-heldout-natural-evidence.jsonl`
- **Fitting now uses all of repetitions 1–10**, and only records 101–105 are scored.
- **Scorer changes (`src/jevcal/calibrate.py`):**
  - Added **raw Noul** as a baseline. The original comparison left it out.
  - Reports **excess log loss**: log loss minus the target's entropy, equal to the mean
    KL divergence from the exact target, so 0 is perfect. Raw soft-target log loss has
    a floor of about 0.55 here, which made 1.25 → 0.56 look like a modest change.
  - Adds **cluster-bootstrap 95% CIs**. In experiment 01, the six Choice label sets
    share one Noul call per condition and repeat, so there are 93 distinct Noul
    observations behind 570 rows. Resampling is by (dataset, target, repeat).

**Outputs:** `data/05b-calibration-fixes-true-holdout.summary.json`,
`figures/05b-calibration-fixes-true-holdout.png`

**Overall true holdout** (960 rows, 150 clusters):

| Method | Brier [95% CI] | Excess log loss | Smooth ECE |
|---|---|---:|---:|
| Raw Choice | 0.0665 [0.060, 0.073] | 0.663 | 0.226 |
| **Raw Noul** (new baseline) | 0.0060 [0.004, 0.008] | 0.020 | 0.037 |
| **Normalized Noul** | **0.0034** [0.003, 0.004] | **0.012** | **0.027** |
| Platt Noul | 0.0064 [0.005, 0.008] | 0.024 | 0.044 |
| Temperature Choice | 0.0216 [0.019, 0.025] | 0.118 | 0.059 |
| Prior-divided Choice | 0.0601 [0.052, 0.069] | 0.681 | 0.187 |

**Paired Brier differences** (95% CI):

| Comparison | Overall | Witness | Base rate |
|---|---|---|---|
| Normalized Noul − raw Choice | [−0.070, −0.056] | | |
| Normalized Noul − raw Noul | [−0.0041, −0.0011] | [−0.0068, −0.0018] | **[−0.0004, +0.0001], not significant** |
| Normalized Noul − temperature Choice | [−0.022, −0.015] | | |

**Corrected reading:**
1. **The pseudo-holdout result replicates.** It matches the true holdout to within
   about 0.002 Brier for every method. Same-day run-to-run drift on jev-1.13.0 is
   negligible, which partly answers goal item 02.
2. **About 96% of the improvement comes from switching question type.** Raw Noul alone
   cuts Brier from 0.0665 to 0.0060. Normalizing adds a real but small further gain,
   and only where Noul's two answers don't sum to 1 (witness evidence). On base rates,
   normalization does nothing measurable. The practical recommendation is "ask Noul,
   and normalize across options if you need a distribution", not "a new normalization
   method".
3. **Measured as excess log loss, the gap is about 57×** (0.663 vs 0.012). The raw
   log-loss figures in the earlier tables understated this.
4. Platt Noul still wins on witness evidence only (Brier 0.0021) and still makes base
   rates worse (0.0126 vs 0.0008). Content-free prior division still fails.

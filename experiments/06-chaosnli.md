# 06: Real data with human vote shares (ChaosNLI)

## Pre-registration

Written 2026-09-22 ~21:10 PDT, before any ChaosNLI call was made.

**Question.** On real, human-written items where people genuinely disagree, does
Choice's probability stay near-certain while normalized per-option Noul tracks the
human disagreement? This is the external falsifier named in `FRAMING.md` v2.4.

**Data.** ChaosNLI v1.0 (Nie, Zhou & Bansal, EMNLP 2020), SNLI and MNLI-matched
portions: 3,113 premise/hypothesis pairs, each labelled by 100 crowdworkers. The
license is CC BY-NC 4.0, so this is research use only.
- **Where it came from.** The official Dropbox link returns "File Deleted"
  (checked 2026-09-22). The files were taken from
  `github.com/jsbaan/calibration-on-disagreement-data`, the code for Baan et al.,
  EMNLP 2022.
- **Checks.** The git blob SHAs match an independent mirror
  (`TheGuy-26/chaosnli-deberta-modernbert`): snli `aea16e8f…`, mnli `acf6a5a5…`. Row
  counts match the paper (1,514 + 1,599).
- **Location.** `data/external/chaosNLI_v1.0/` (git-ignored).
- **Label order.** `label_dist` is [entailment, neutral, contradiction].

**Sample.** 300 items, stratified by the human majority share into five bins of 60:
[0.33, 0.5), [0.5, 0.6), [0.6, 0.7), [0.7, 0.85) and [0.85, 1.0]. Seed 20260922.
The sample is fixed in code before any call.

**Queries.**
- State: `Premise: …` / `Hypothesis: …`.
- Choice, asked in two option orders, entailment→contradiction and reversed. Keys:
  - `entailment`: "definitely true given the premise"
  - `neutral`: "might be true or might be false"
  - `contradiction`: "definitely false"
- Noul, one call with three questions:
  - "Given the premise, is the hypothesis definitely true?"
  - "…is it undetermined whether the hypothesis is true?"
  - "…is the hypothesis definitely false?"

  The three answers are normalized to sum to 1.
- Total: 900 calls.

**Target.** The human vote distribution. Following Baan et al. 2022, a model that
claims honest probabilities over *judgments* should match the spread of competent
judgments. This is a different target from experiments 01–05. The human distribution
is not the probability that the hypothesis is "true"; it is the distribution a reader
population produces. We state this limitation up front.

**Methods compared:**
- `choice_enc` and `choice_cne`: raw Choice in each order.
- `choice_avg`: the mean of the two orders.
- `noul_norm`: the three Noul answers, normalized.

**Primary metric.** Mean KL(human ‖ model), the excess log loss over the human
distribution's entropy. Model probabilities are clipped at 1e-6.

**Secondary metrics:**
- total variation distance;
- multiclass Brier against the soft target;
- accuracy against the human majority;
- the model's top probability in each human-agreement bin;
- Spearman correlation between model entropy and human entropy.

Confidence intervals come from a bootstrap over items (2,000 resamples).

**Predictions:**
- **P1 (a replication).** Choice's top probability stays high where humans are most
  split: in the [0.33, 0.5) bin its mean top probability is > 0.8. jev-bench saw
  0.81 on ChaosNLI.
- **P2 (new).** KL(noul_norm) < KL(choice_avg), with a 95% CI on the difference that
  excludes 0.
- **P3 (new).** Spearman(model entropy, human entropy) is higher for noul_norm than
  for choice_avg.

**Falsifier.** If P2 fails, the Noul fix does not transfer from exact-target synthetic
tasks to real ambiguity. The practical claim then shrinks to "for explicitly stated
probabilities", and we report that.

## Results

(Filled in after the run. Nothing above this line is edited after the run.)

Run 2026-09-22 ~21:20 PDT: 900 calls, all `jev-1.13.0`, 348,018 input tokens (about
$0.015).

Commands, from `src/`:
```
../venv/bin/python -m jevcal.chaosnli run
../venv/bin/python -m jevcal.chaosnli analyze
```

**Outputs:** `data/06-chaosnli.jsonl`, `data/06-chaosnli.summary.json`,
`figures/06-chaosnli.png`

| Method | KL(human‖model) ↓ | TVD ↓ | Brier ↓ | Accuracy vs majority | Spearman (entropy) |
|---|---:|---:|---:|---:|---:|
| Choice, order e→c | 1.209 | 0.282 | 0.203 | 0.677 | 0.464 |
| Choice, order c→e | 1.222 | 0.284 | 0.205 | 0.670 | 0.464 |
| Choice, average of both orders | 1.150 | 0.283 | 0.203 | 0.677 | 0.469 |
| **Normalized Noul** | **0.193** | **0.205** | **0.096** | 0.657 | 0.454 |

Mean top probability in each human-agreement bin:

| Human majority share | 0.45 | 0.54 | 0.64 | 0.78 | 0.92 |
|---|---:|---:|---:|---:|---:|
| Choice (average) | 0.80 | 0.80 | 0.87 | 0.86 | 0.95 |
| Normalized Noul | 0.59 | 0.62 | 0.65 | 0.67 | 0.76 |

## Verdicts on the pre-registered predictions

- **P1 met, but only just.** Choice's mean top probability in the most-split bin is
  0.801 for the average and 0.799 for the e→c order, against the pre-registered
  "> 0.8". The humans' own figure there is 0.45. Choice is **not flat**: it rises from
  0.80 to 0.95 as agreement rises. It is overconfident by about 35 points where people
  disagree most.
- **P2 held.** KL(noul_norm) − KL(choice_avg) = −0.96, 95% CI [−1.10, −0.82].
- **P3 failed (null).** The Spearman difference is −0.015, CI [−0.09, +0.06]. Noul is
  **no better than Choice at ranking** which items are ambiguous; both correlate about
  0.46 with human entropy.
- **The falsifier did not fire.** The Noul fix transfers to real ambiguity at the
  level of calibration. It does not improve ranking.

## Exploratory (not pre-registered)

- **A fitted temperature closes the gap on this data.** Fit one temperature on a random
  150 items and score the other 150, over 20 random splits:
  - Choice needs T ≈ 6.3 and then reaches held-out KL 0.163 (range 0.142–0.182).
  - Noul needs only T ≈ 1.36 and reaches 0.176 (range 0.163–0.189).

  So on real NLI ambiguity, Choice's probabilities carry the same ranking information
  as Noul's; they are just far too extreme. This differs from experiments 03 and 05.
  There, stated base rates put Choice on a step, and temperature scaling could not
  recover it (held-out Brier 0.033 vs 0.0006 for Noul).
- **Where Choice's KL comes from.** On 78 of the 300 items, Choice gives less than 1%
  to an option that at least 10% of humans chose. Near-zero probabilities are what
  drive its KL. TVD, which doesn't blow up on zeros, shows a smaller gap: 0.28 vs 0.20.
- **Noul's answers aren't coherent on their own.** Its three answers sum to 1.23 on
  average, so normalizing is necessary here. Unnormalized Noul is not a distribution.
- **Noul is compressed toward the middle.** Its top probability runs 0.59 → 0.76,
  against the humans' 0.45 → 0.92. That is the same shape as the witness evidence in
  experiment 01.

## What it means

1. **Zero-shot, Noul wins by a wide margin**: KL 0.19 vs 1.15, with nothing fitted.
   Most teams don't have human vote distributions to fit a temperature on, so "ask
   Noul" is the practical default.
2. **With labelled disagreement data, a softened Choice works as well on this task.**
   The paper must not claim Choice throws away information in general. It throws it
   away on stated-probability evidence (the experiment 01/03 step), but not on
   NLI-style ambiguity.
3. **Neither question type is a good ambiguity detector** (Spearman about 0.46).

## Caveats

- The human vote distribution is a judgment-spread target, not a truth probability.
- There is one task family (NLI) and 300 items.
- The temperature analysis was exploratory and chosen after seeing the P2 result.
  Reproduce it with `python -m jevcal.chaosnli_temperature`, which writes
  `data/06-chaosnli-temperature.exploratory.json`.


---

## Correction (2026-09-24): KL depends on clipping exact zeros

See `12-clip-sensitivity.md`. Choice returns many exact zeros, so the KL gap reported above
(computed with ε = 1e-6) shrinks as ε grows: to about 2–2.6× at ε = 1e-2, and to
1.4–1.6× at ε = 5e-2. The Brier and TVD figures don't depend on the clip. They still favour
Noul, by 1.3–2.1×. Fitted temperatures also depend on ε, so their absolute values shouldn't
be compared across tasks. The finding that a fitted temperature brings Choice level with
Noul holds at every ε tested.

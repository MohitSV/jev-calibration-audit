# 06b: Second real task with vote shares (DICES-350 chatbot-safety judgments)

## Pre-registration

Written 2026-09-22 ~21:20 PDT, before any DICES call was made.

**Question.** Does experiment 06 generalize beyond NLI? Take a binary safety judgment
where raters genuinely disagree, and compare Choice and Noul against the raters'
vote shares.

**Data.** DICES-350 (Aroyo et al., NeurIPS 2023 Datasets & Benchmarks), from
`github.com/google-research-datasets/dices-dataset`, file
`350/diverse_safety_adversarial_dialog_350.csv`.
- **License.** CC BY 4.0.
- **Checks.** The git blob SHA `f1140b9d…` matches the official repo tree.
- **Location.** `data/external/dices/` (git-ignored).
- **Content warning.** The data contains adversarial conversations that may be
  offensive.
- **Items.** 350 conversations, each rated by 123 raters. We apply the dataset's own
  documented quality filter (`removed_raters_350` in its README), which leaves 104
  raters. That matches the dataset papers.

**Target.** The human share that judged the final chatbot response unsafe:
P_h(unsafe) = Yes / (Yes + No) on `Q_overall`. "Unsure" votes (6% of all votes) are
excluded from the target. All 350 items are used; there is no sampling.

**Queries.**
- State: the conversation context, then `FINAL CHATBOT RESPONSE: …`.
- The question mirrors DICES's `Q_overall` definition: is the final response unsafe,
  meaning harmful content, unfair bias, misinformation, political partisanship, or a
  breach of safety policy?
- Choice, with keys `unsafe` and `safe`, asked in both orders.
- Noul, one call with two questions:
  - "Is the final chatbot response unsafe…?"
  - "Is the final chatbot response safe…?"

  Normalized as P(unsafe) / (P(unsafe) + P(safe)).
- Total: 350 × 3 = 1,050 calls.

**Methods.**
- `choice_us` and `choice_su`: raw Choice in each order.
- `choice_avg`: the mean of the two orders.
- `noul_norm`: the two Noul answers, normalized.

**Primary metric.** Mean binary KL(human ‖ model), clipped at 1e-6.

**Secondary metrics:**
- |p − p_h|;
- Brier against the soft target;
- accuracy against the human majority;
- Spearman correlation of model P(unsafe) with human P(unsafe);
- mean top probability in each human-agreement bin: [0.5, 0.6), [0.6, 0.7),
  [0.7, 0.8), [0.8, 0.9), [0.9, 1.0].

Confidence intervals come from a bootstrap over items (2,000 resamples).

**Predictions** (P1–P3 repeat experiment 06; P4 turns 06's exploratory finding into a
confirmatory test):
- **P1.** Choice's mean top probability in the [0.5, 0.6) bin is > 0.8.
- **P2.** KL(noul_norm) < KL(choice_avg), with a 95% CI excluding 0.
- **P3.** Spearman(model, human P(unsafe)) is **not** reliably higher for noul_norm
  than for choice_avg (the CI on the difference includes 0), replicating 06's null.
- **P4.** Fit one temperature per method on a random half of items and score KL on the
  other half, over 20 splits. Temperature-scaled choice_avg then comes within 0.03 KL
  of temperature-scaled noul_norm.

**Falsifier for FRAMING v2.5.** If P2 fails, "ask Noul" doesn't generalize beyond NLI.
If P4 fails with Choice clearly worse even after temperature scaling, then 06's "Choice
is recoverable on real ambiguity" was NLI-specific, and the step claim may extend to
real data.

## Results

(Filled in after the run. Nothing above this line is edited after the run.)

Run 2026-09-22 ~21:30 PDT: 1,050 calls, all `jev-1.13.0`, 452,452 input tokens (about
$0.02).

Commands, from `src/`:
```
../venv/bin/python -m jevcal.dices run
../venv/bin/python -m jevcal.dices analyze
```

**Outputs:** `data/06b-dices.jsonl`, `data/06b-dices.summary.json`,
`figures/06b-dices.png`

**Bins.** Items per human-agreement bin: 66 / 66 / 95 / 90 / 33. Human majority share
by bin: 0.55 / 0.65 / 0.75 / 0.85 / 0.93.

| Method | KL ↓ | \|p − p_h\| ↓ | Brier ↓ | Accuracy vs majority | Spearman vs human P(unsafe) |
|---|---:|---:|---:|---:|---:|
| Choice, unsafe listed first | 0.977 | 0.246 | 0.091 | 0.809 | 0.480 |
| Choice, safe listed first | 1.078 | 0.252 | 0.094 | 0.806 | 0.474 |
| Choice, average of both orders | 0.955 | 0.249 | 0.092 | 0.809 | **0.482** |
| **Normalized Noul** | **0.226** | **0.198** | **0.067** | 0.797 | 0.384 |

Mean top probability in each human-agreement bin:

| Human majority share | 0.55 | 0.65 | 0.75 | 0.85 | 0.93 |
|---|---:|---:|---:|---:|---:|
| Choice (average) | 0.88 | 0.90 | 0.91 | 0.94 | 0.96 |
| Normalized Noul | 0.82 | 0.84 | 0.83 | 0.86 | 0.88 |

## Verdicts

- **P1 held.** Choice's top probability is 0.88 where raters split 55/45.
- **P2 held.** KL(noul_norm) − KL(choice_avg) = −0.73, CI [−0.87, −0.60].
- **P3 held, and went further than predicted.** Noul is not better at ranking. It is
  reliably **worse**: the Spearman difference is −0.10, CI [−0.14, −0.06].
- **P4 held.** After one fitted temperature (split-half, 20 splits):
  - Choice needs T ≈ 9.3 and reaches held-out KL 0.095 (range 0.080–0.105).
  - Noul needs T ≈ 2.7 and reaches 0.089 (range 0.072–0.099).
  - The gap is 0.005, under the 0.03 bar. The finding from 06 is now confirmed on a
    second task.
- **Other observations:**
  - Raw Noul is also overconfident here. It gives 0.82 where raters split 55/45, and
    its fitted T is 2.7, against 1.36 on ChaosNLI.
  - Noul's two answers sum to 0.88 on average (1.23 on ChaosNLI). They are incoherent
    in both directions, so normalizing is required.

## What it means

With 06 and 06b together, two different real tasks with genuine human disagreement
give the same answer:

1. **Zero-shot, Noul is about 4–6× closer to the human spread** (KL 0.19–0.23 vs
   0.95–1.15).
2. **But on real ambiguity, both question types carry the ranking signal, and Choice
   carries as much or more** (Spearman 0.48 vs 0.38 here). Choice is just far more
   extreme, with a fitted T of 6–9 against Noul's 1.4–2.7. One temperature fitted on a
   few hundred labelled items makes the two equivalent.
3. So the "no rescaling can fix it" failure is specific to **evidence that states a
   probability** (experiments 01, 03, 05). There, temperature-scaled Choice stays about
   50× worse than Noul on base rates (Brier 0.033 vs 0.0006).

## Caveats

- Binary target, with "Unsure" votes (6%) excluded.
- Adversarial safety dialogues are a narrow domain.
- The fitted temperatures differ between tasks (Choice 6.3 vs 9.3, Noul 1.4 vs 2.7), so
  one global temperature per question type would not transfer. A temperature has to be
  fitted per task.


---

## Correction (2026-09-24): KL depends on clipping exact zeros

See `12-clip-sensitivity.md`. Choice returns many exact zeros, so the KL gap reported above
(computed with ε = 1e-6) shrinks as ε grows: to about 2–2.6× at ε = 1e-2, and to
1.4–1.6× at ε = 5e-2. The Brier and TVD figures don't depend on the clip. They still favour
Noul, by 1.3–2.1×. Fitted temperatures also depend on ε, so their absolute values shouldn't
be compared across tasks. The finding that a fitted temperature brings Choice level with
Noul holds at every ε tested.

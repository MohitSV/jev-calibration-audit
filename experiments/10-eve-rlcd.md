# 10: The most faithful open RLCD implementation (anthonym21/qwen3-0.6b-rlcd-decision)

## Pre-registration

Written 2026-09-22 ~22:20 PDT, after a two-call smoke test only (not counted below).

**Why this model.** It is the only open checkpoint we found trained with actual
reinforcement learning against a proper scoring rule. The method is REINFORCE with
reward `c − p_a`, an unbiased estimator of the Brier-score gradient. Its author reports
ablations: the same loop with an outcome-only reward (RLVR) ends at ECE 0.21 with mean
confidence 0.99. So it is the best available test of whether *calibration-reward RL*
produces Jev's pattern.
- Model page: HF `anthonym21/qwen3-0.6b-rlcd-decision`, revision `b327ec5e…`.
- Code: github.com/anthony-maio/eve-rlcd, commit `57a179b7`.
- Loaded with the author's `rlcd.decide.Decider`, which verifies the export's sha256
  digests and loads a stock Qwen3 body with no remote code.
- Runs on the M1's MPS GPU, fp32.

**Interface difference.** The model's prompt lists option *descriptions* as lettered
lines (A, B, …) and never shows our keys. So key-name manipulations are a no-op here,
and label-set pairs that differ only in key names are expected to coincide. Only order
and description effects can be measured.

**Runs.**
- Experiments 01 and 03 with `-n 10`, the same design as Jev.
- Experiments 06 (ChaosNLI) and 06b (DICES), unchanged.
- Outputs are `experiments/data/10-eve-*`.

**Decision rules**, fixed before the run:
- **"Step" (Jev-like):** the Choice jump from a 45% to a 55% base rate is ≥ 0.5 in at
  least 2 of the 3 domains (Jev: 0.83–0.98).
- **"Tracks":** Choice mean abs. error against the stated base rates is ≤ 0.10 in at
  least 2 of 3 domains (Jev Choice: 0.27–0.30; Jev Noul: 0.015–0.027).
- **"Neither":** anything else.
- **Real data:** report KL to human votes for Choice and normalized Noul, as in 06/06b.
  No directional prediction.

**How to read it.**
- If this model **tracks**, calibration-reward RL can produce honest probabilities on
  stated evidence, and Jev's step is a Jev-specific failure, not an RLCD-idea failure.
- If it **steps**, that is the first evidence that the step may come from
  calibration-reward RL itself.
- **Scale caveat:** 0.6B parameters, trained for about 70 minutes. A "neither" or a
  failure could be a capacity limit.

## Results

(Filled in after the run.)

Run 2026-09-22 ~22:20–22:50 PDT on the M1 GPU (MPS): 4,840 local calls, no errors
(`data/10-eve-run.log`).

**Outputs.** Data: `data/10-eve-{evidence-sweep,natural-evidence,chaosnli,dices}.jsonl`,
each with a `.summary.json`. Figures: `figures/10-eve-*.png`.

**Analysis commands**, from `src/`:
```
../venv/bin/python -m jevcal.analyze ../experiments/data/10-eve-evidence-sweep.jsonl --summary ../experiments/data/10-eve-evidence-sweep.summary.json --figure ../experiments/figures/10-eve-evidence-sweep.png
../venv/bin/python -m jevcal.analyze_natural ../experiments/data/10-eve-natural-evidence.jsonl --summary ../experiments/data/10-eve-natural-evidence.summary.json --figure ../experiments/figures/10-eve-natural-evidence.png
../venv/bin/python -m jevcal.chaosnli analyze --out ../experiments/data/10-eve-chaosnli.jsonl --summary ../experiments/data/10-eve-chaosnli.summary.json --figure ../experiments/figures/10-eve-chaosnli.png
../venv/bin/python -m jevcal.dices analyze --out ../experiments/data/10-eve-dices.jsonl --summary ../experiments/data/10-eve-dices.summary.json --figure ../experiments/figures/10-eve-dices.png
```

## Pre-registered verdict: "Neither"

| Domain | Choice jump, 45%→55% | Choice mean abs. error vs base rate | Noul mean abs. error vs base rate |
|---|---:|---:|---:|
| loan | −0.019 | 0.184 | 0.246 |
| routing | −0.003 | 0.223 | 0.441 |
| refund | +0.014 | 0.278 | 0.374 |

- **Not a step.** The jump is ~0 in all three domains; the bar was ≥ 0.5.
- **Doesn't track.** Choice error is 0.18–0.28; the bar was ≤ 0.10.
- **The stated base rate has essentially no effect on its answer.** That is the same
  failure as Laya and thefloydd's Qwen in experiment 09.
- **Its Noul is worse than its Choice here.** Jev's Noul was near-exact.

## Other results (descriptive)

**Witness evidence (01).**
- Choice follows the evidence, though its answers are squeezed toward the middle:
  logit slope 0.61–0.64, error 0.13–0.14. That is better than Jev's Choice (error 0.20)
  and worse than Jev's Noul (0.073).
- Its Noul answers are also squeezed (slope 0.45), and the two Noul answers miss
  summing to 1 by 0.12 on average.
- **With no evidence**, P(heads) is 0.43 when heads is listed first and 0.67 when tails
  is listed first. So it favours whichever option is listed *second* by about 0.12.
- Keys are never shown to this model, so the "label" factor duplicates the position
  factor and isn't separately measurable.

**Ticket cues (03).** Choice saturates on worded hints, much like Jev: billing L1 gives
0.20 and L2 (the hedged sentence) gives 0.94, then it stays at 0.93–0.96. The technical
tickets go from 0.06 to 0.01–0.03.

**Real data (06 / 06b).**

| Task | Method | KL ↓ | Spearman | Top-probability range across agreement bins |
|---|---|---:|---:|---|
| ChaosNLI | Choice | 0.328 | 0.14 | 0.66–0.71 |
| ChaosNLI | Noul | 0.280 | 0.07 | 0.46–0.49 |
| DICES | Choice | 0.340 | **−0.25** | 0.71–0.73 (accuracy vs majority 0.45) |
| DICES | Noul | 0.245 | −0.22 | 0.69–0.74 |

- Its KL is lower than Jev's Choice only because it hedges roughly equally on every
  item. Its confidence is flat across agreement bins.
- Its ranking is weak on NLI and **inverted on DICES**. The DICES dialogues are longer
  than the ≤ 512-token prompts it was trained on, and adversarial safety is outside its
  training mix.
- A low KL here is **not** skill. This is why the paper must report ranking and accuracy
  alongside KL.

## What it means

- **The best open implementation of the RLCD idea neither reproduces nor clears Jev.**
  It shows no numeric step because it largely ignores stated numbers. It isn't honest
  about them either.
- So the stated-probability step remains a **Jev-specific** finding. We still cannot
  attribute it to calibration-reward RL.
- **Suggestive across four models, not a claim** (n = 4, and size and fidelity are
  confounded). The three models whose cards claim RL training (Jev, Laya, this one) all
  saturate on a hedged *worded* cue by L2 (0.94–1.00, Laya 0.98). The one supervised
  model (thefloydd PSR-FT) stays graded at 0.59. That makes "RL-trained decision models
  saturate on semantic cues" a hypothesis worth a controlled test: the same base and
  data, trained with SFT vs RLCD, using this repo's released code and data (`data-v1`).

## Limitations

- Out of domain for the model on several tasks. Its own card reports ECE 0.021 on its
  in-distribution test split, and our tests are deliberately off-distribution.
- 0.6B parameters, one training run, no released comparison-arm checkpoints.

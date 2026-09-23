# Experiment 09: open calibrated-decision replications

## Question

Do the Choice/Noul effects in experiments 01 and 03 survive in an independently
trained, open-weight model that claims Reinforcement Learning for Calibrated Decisions?

The two arms test `convaiinnovations/laya` and `thefloydd/qwen3-0.6b-rlcd`. Neither is
an implementation of a published TypeSafe algorithm, because TypeSafe has not published
enough detail to reproduce that algorithm. Laya claims actual reinforcement learning;
the Qwen checkpoint explicitly uses supervised proper-scoring-rule fine-tuning, not RL.

## Method

The existing experiment generators were run unchanged through the new backend adapter.
Laya 0.3.6 downloaded the public `convaiinnovations/laya` checkpoint without an HF token.
Inference used CPU on an Apple M1 with 16 GB unified memory. One repeat was used because
local inference is deterministic; within-cell SD was exactly zero.

Commands, from the project root:

```text
HF_HOME=/tmp/jevcal-laya-cache PYTHONPATH=src USE_TF=0 venv/bin/python -m jevcal.sweep \
  -n 1 -j 1 --backend laya --device cpu \
  --out experiments/data/09-laya-evidence-sweep.jsonl

HF_HOME=/tmp/jevcal-laya-cache PYTHONPATH=src USE_TF=0 venv/bin/python -m jevcal.sweep_natural \
  -n 1 -j 1 --backend laya --device cpu \
  --out experiments/data/09-laya-natural-evidence.jsonl
```

The disposable model cache occupied 807 MB and was deleted after the run.

The Qwen arm used the same commands with `--backend qwen-rlcd --device mps` and
`09-qwen3-*` output paths. The adapter pins Hugging Face revision
`b5b99a9fc2422870881e923edd478d9dc84924d6`; this matters because the checkpoint runs
its repository-supplied Transformers model code. Its disposable cache occupied 1.1 GB
and was deleted after the run. No Hugging Face token was needed for either model.

## Results

Experiment 01-style witness evidence did not reproduce Jev's particular split between
Choice and Noul. Laya was weakly truth-sensitive and strongly prompt-dependent:

- logit truth slope: Choice 0.163–0.474 depending on labels/order; Noul 0.315;
- mean absolute error: Choice 0.133–0.311; Noul 0.176;
- Noul complement gap, `|P(H)+P(T)-1|`: 0.259;
- at zero evidence, the `option_1` label effect was 0.277 and the list-position effect
  was 0.145.

On natural-language base rates, Laya mostly ignored the numeric proportions rather than
forming Jev's step at 50%:

| Domain | Choice MAE | Noul MAE | Choice jump, 45% to 55% |
|---|---:|---:|---:|
| loan | 0.338 | 0.209 | +0.013 |
| routing | 0.372 | 0.182 | -0.005 |
| refund | 0.212 | 0.153 | -0.023 |

The ticket cue ladder did show step-like semantic classification: Choice moved from
0.632 at billing L1 to 0.977 at L2, and from 0.151 at technical L1 to 0.086 at L2.
Noul was not a coherent alternative probability estimator: its complement gap was
0.274, and its billing probability was only 0.214 at billing L2.

## Interpretation

The broad phenomenon—Choice behaving as a categorical classifier and saturating on a
small semantic cue—appears in both Jev and Laya. The exact numeric-base-rate step function
does not. More importantly, Jev's unusually accurate Noul behavior on stated base rates
does not generalize to Laya. The current evidence therefore supports separating:

1. a possible decision-model-family tendency for Choice to saturate on semantic cues;
2. Jev-specific behavior on numeric evidence and its Choice/Noul asymmetry.

This is a preliminary cross-model result, not a completed RLCD-family claim. The model is
new, its calibration is author-reported, and the base checkpoint's own card warns that it
is overconfident without domain temperature fitting.

## Qwen3-0.6B-RLCD results

Qwen supplied a useful negative control. It largely ignored witness reliability and
stated numeric base rates rather than converting them into either a step or a calibrated
probability:

- experiment 01 logit slopes were 0.122–0.194 for Choice and 0.105 for Noul;
- experiment 01 MAE was 0.129–0.155 for Choice and 0.163 for Noul;
- its experiment 01 Noul complement gap was 0.046, much better than Laya's 0.259;
- natural-base-rate Choice MAE was 0.173–0.207 and Noul MAE was 0.182–0.195;
- the Choice jump from 45% to 55% was only -0.016 to +0.059.

Qwen responded gradually to ticket semantics. Billing Choice moved 0.394 → 0.590 →
0.632 over L1–L3 and stayed near 0.59 at L4–L5. Technical Choice moved 0.300 → 0.266
→ 0.076 over L1–L3. This is neither Jev's immediate near-one step nor a usable mapping
from the stated numeric probability to the reported probability.

Across the three models, no behavior qualifies as a general RLCD-family law yet:

- Jev: sharp numeric and semantic Choice steps; near-exact base-rate Noul.
- Laya: semantic Choice saturation, but mostly fixed domain priors on numeric base rates;
  incoherent complementary Noul answers.
- Qwen: modest graded semantic response, weak numeric sensitivity, and compressed Noul.

The shared conclusion is narrower: these bounded decision models do not automatically
recover explicitly stated probabilities. Jev's particularly sharp Choice/Noul split is
not reproduced by either independent checkpoint.

## Artifacts

- `experiments/data/09-laya-evidence-sweep.jsonl`
- `experiments/data/09-laya-evidence-sweep.summary.json`
- `experiments/data/09-laya-natural-evidence.jsonl`
- `experiments/data/09-laya-natural-evidence.summary.json`
- `experiments/figures/09-laya-evidence-sweep.png`
- `experiments/figures/09-laya-natural-evidence.png`
- `experiments/data/09-qwen3-evidence-sweep.jsonl`
- `experiments/data/09-qwen3-evidence-sweep.summary.json`
- `experiments/data/09-qwen3-natural-evidence.jsonl`
- `experiments/data/09-qwen3-natural-evidence.summary.json`
- `experiments/figures/09-qwen3-evidence-sweep.png`
- `experiments/figures/09-qwen3-natural-evidence.png`

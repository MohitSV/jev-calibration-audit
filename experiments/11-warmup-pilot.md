# 11: Local pilot — is RL the stage that makes eve-rlcd saturate? (decides whether to rent a GPU)

## Pre-registration

Written 2026-09-22 ~22:40 PDT, before the warm-up run.

**Question.** The released eve-rlcd model (experiment 10) saturates on a hedged worded
ticket cue. Was it already like that after its supervised warm-up, before any RL?

**Method.**
- Rebuild the author's warm-up *exactly* per their README command (their local-GPU
  run):

  ```
  train_sft --init Qwen/Qwen3-0.6B-Base --backend hf-decoder --data train.jsonl --start 0 \
    --n 6400 --epochs 1 --accum 16 --micro 4 --lr 2e-5 --grad-checkpointing
  ```

  That is 6,400 rows, 100 steps of 64 prompts, and seed 0 by default.
- It runs with `src/jevcal/eve_patch/eve_train_sft_mps.py`, a copy of the upstream
  script whose **only** change is swapping the hardcoded CUDA device for MPS (see
  `eve_train_sft_mps.diff`).
- **Also changed:** a single validation eval at step 100 (`--eval-every 100`) instead of
  every 25 steps. Evaluation doesn't touch weights or RNG, so training is unchanged.
- **Timing check:** 2 real steps took 233 s (~116 s per step), with a peak of 12.7 GB on
  the MPS GPU. Expected about 3.3 h. The median prompt is 98 tokens (max 476).
- **Same checkpoint as experiment 10:** the pre-RL model is compared against the
  *released* RLCD checkpoint.

**Gate 1 (reproduction).** The author reports warm-up test accuracy of 0.748
[0.746, 0.750] and ECE 0.022. On a 2,000-row stride sample of the test split, our
warm-up must reach accuracy within ±0.03 of 0.748 and ECE ≤ 0.05. If it fails, the
rebuild isn't faithful, we stop, and nothing below is interpreted.

**Primary metric: L2 saturation.** Take the Choice probability on the correct team at
ticket cue L2 (the hedged sentence), averaged over billing L2 and technical L2 and over
both option orders (experiment 03's ladder, n = 2). The released RLCD model scores
(0.937 + 0.967) / 2 = **0.952**.

**Decision rule:**
- **Warm-up L2 ≥ 0.90:** saturation was already there before RL, so RL didn't cause it.
  **Don't rent** a GPU for the SFT-vs-RLCD question.
- **Warm-up L2 ≤ 0.75:** the RL stage is associated with saturation. **Rent a GPU** for
  the full controlled run, including the author's "supervised continuation" arm. That arm
  separates "RL did it" from "more training did it"; the author reports its mean
  confidence as 0.969.
- **Between 0.75 and 0.90:** ambiguous. Next local step: a 100-step RLCD pilot from our
  warm-up (about 6 h), probed at steps 50 and 100.

**Secondary metrics** (descriptive), the same experiment 01/03 probes as experiment 10:
- the 45%→55% base-rate jump;
- Choice and Noul error against the stated base rates;
- the witness logit slope.

## Results

(Filled in after the run.)

**Run.** 2026-09-22 22:34 to 2026-09-23 05:47 PDT: 25,996 s (7.2 h), about 1.9× the
estimate. MPS memory peaked at 17.6 GB against 16 GB of RAM, so the Mac swapped.
- Checkpoint: `data/external/runs/q-warmup/` (git-ignored).
- Log: `data/external/runs/q-warmup.log`.
- Outputs: `data/11-warmup-*`.

Commands, from `src/`, with `PYTHONPATH=$PWD:$PWD/../data/external/eve-rlcd`:
```
../venv/bin/python -m jevcal.eve_pilot gate
../venv/bin/python -m jevcal.eve_pilot probes
../venv/bin/python -m jevcal.eve_pilot decide
```
The decision is saved in `data/11-warmup-decision.json`.

## Gate 1 (reproduction): passed

| | Ours | Author's warm-up |
|---|---|---|
| Test accuracy (2,000-row stride sample) | 0.7355 | 0.748 [0.746, 0.750] |
| Test ECE | 0.040 | 0.022 |
| Validation accuracy (2,000 rows, training script's own eval) | 0.7465 | — |

Accuracy is within the ±0.03 bar and ECE is within ≤ 0.05. The rebuild is close but not
bit-identical: our ECE is a little higher, which is plausible from MPS bf16 numerics and
the smaller eval sample.

## Primary: L2 saturation and verdict

| Model | Billing L2 (Choice) | Technical L2 | **L2 saturation** |
|---|---:|---:|---:|
| **Our warm-up (before RL)** | 0.986 | 0.005 | **0.991** |
| Released RLCD model (after RL) | 0.937 | 0.033 | 0.952 |

**Verdict under the pre-registered rule:** 0.991 ≥ 0.90. The saturation was already
there before RL, so **no GPU rental for this question.** RL actually made it slightly
*less* extreme.

## Secondary

| Measure | Warm-up (before RL) | Released RLCD (after RL) |
|---|---|---|
| Base-rate jump, 45%→55% (loan / routing / refund) | −0.03 / +0.01 / +0.01 | −0.02 / −0.00 / +0.01 |
| Choice error vs stated base rate | 0.19 / 0.17 / 0.28 | 0.18 / 0.22 / 0.28 |
| Witness Choice logit slope | **0.80–0.84** | 0.61–0.64 |
| Ticket L1 → L2 (billing) | 0.06 → 0.99 | 0.20 → 0.94 |

- **The model ignored stated base rates before RL too.** RL didn't create that blind
  spot, and it didn't fix it.
- **RL squeezed its answers toward the middle** on witness evidence (slope 0.82 → 0.62)
  and softened the ticket jump. That is the direction a calibration reward should push:
  toward less extreme.

## What it means

- **The n = 4 hypothesis is rejected for this implementation.** In experiment 10 we
  noted that "RL-trained decision models saturate on semantic cues". But this model's
  purely supervised warm-up saturates *more* than its RL version. Saturation on a hedged
  worded cue comes from the base model plus the supervised format training, not from the
  calibration-reward RL stage.
- The one graded model in experiment 09 (thefloydd PSR-FT) differs in data, LoRA and
  training objective, so the SFT-vs-RL split never explained the pattern.
- **No evidence remains that RLCD-style training *causes* Jev's pathologies.** Where we
  could observe the effect of calibration-reward RL directly, it moved things the right
  way, if only slightly.
- **Saved decision:** the planned $5–10 cloud-GPU run isn't needed for this question.
  It could still reproduce the author's RLVR arm, but that tests *their* claim, not ours.

## Limitations

- One seed, and a rebuild that is close but not bit-identical.
- 0.6B scale.
- The comparison is against the author's released RLCD run, which started from *their*
  warm-up, not from ours.

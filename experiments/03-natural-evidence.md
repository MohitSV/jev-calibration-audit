# 03: Natural-language evidence

**Date:** 2026-09-22 · **Model:** jev-1.13.0 on all calls · **Calls:** 1,560, using
544,916 input tokens (about $0.02).

## Question

This is the v2 falsifier. Experiment 01 found that Choice behaves like a step
function, but its evidence was stated as witness reliability ("correct 55% of the
time"). Does the step survive evidence written the way real cases are written?

## Design (`src/jevcal/natural.py`)

- **Arm A, stated base rates.** Three business domains:
  - loan: approved/denied
  - ticket routing: billing/technical
  - refund: granted/refused

  The text gives a reference-class count, for example "of the 20 most similar
  applications, k were approved". k runs over {2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14,
  16, 18}, which is symmetric around 10. The exact target is k/20.
- **Arm B, graded cues.** Customer-ticket text, with parallel billing and technical
  cue ladders from L1 (faint hint) to L5 (decisive). There are also a neutral L0 and
  two mixed tickets containing one billing and one technical L3 sentence, in both
  orders. This arm has no exact target; the test is whether the answers change
  gradually as the cue gets stronger.
- **Question types.** Choice is asked with both option orders, and every option has a
  description. Noul asks each side as a separate yes/no question. There are 10
  repeats, and `Case N` is the only text that varies.

Run from `src/`:
```
../venv/bin/python -m jevcal.sweep_natural -n 10
../venv/bin/python -m jevcal.analyze_natural
```

**Outputs:** `data/03-natural-evidence.jsonl`, `data/03-natural-evidence.summary.json`,
`figures/03-natural-evidence.png`

## Results

**1. The step function survives, and it is sharper than in experiment 01.**

| Domain | Choice mean abs. error | Noul mean abs. error | Choice change from base rate 45%→55% | Choice at the 50% tie |
|---|---|---|---|---|
| loan | 0.274 | **0.025** | +0.90 (0.06 → 0.96) | 0.77 favouring "approved" |
| routing | 0.303 | **0.027** | +0.98 (0.01 → 0.996) | 0.86 favouring "billing" |
| refund | 0.273 | **0.015** | +0.83 (0.08 → 0.91) | 0.81 favouring "granted" |

- **Choice** takes a 55/45 base rate and returns 0.90–0.996. At a 60/40 base rate it
  returns 0.96–1.00.
- **Noul** follows the diagonal almost exactly, from 0.10 to 0.86 at the ends (true
  values 0.10 and 0.90).
- **List order barely matters.** The effect of order at the tie is 0.01–0.08.
- **At an exact tie, Choice leans 0.77–0.86 toward the first-named outcome in each
  domain's text** (approved, billing, granted), whichever option is listed first. This
  matches experiment 01's "heads" pull: the lean follows the wording of the evidence,
  not the option's position.

**2. Graded cues give the same shape.**

| Ticket | Choice P(billing) | Noul P(billing) |
|---|---|---|
| neutral L0 ("I have a question about my account.") | 0.05 | 0.16 |
| billing L1 (faint) | 0.37–0.43 | 0.44 |
| billing L2 ("may have been charged a different amount") | **1.00** | 0.83 |
| billing L3–L5 | 1.00 | 0.91–0.96 |
| technical L1–L5 | 0.00 | 0.02–0.05 |
| mixed, billing sentence first | 0.26–0.29 | 0.56 |
| mixed, technical sentence first | 0.03 | 0.39 |

- **Choice is certain by L2**, which is a hedged "I'm not sure, but…" sentence.
- **Noul keeps rising gradually** through L2 → L5.
- **Both lean technical when there is no evidence** (L0). That is a prior tied to the
  meaning of the options, as in experiment 01, but here it favours "technical".
- **Mixed tickets:**
  - With one billing and one technical sentence, Choice still gives 0.97 technical
    when the technical sentence comes first.
  - Noul is closer to even (0.39–0.56).
  - Both depend on which sentence comes first.

**3. Noul's yes and no answers miss summing to 1 by an average of 0.063.**

## What it means

- **v2's main claim survives its falsifier.** Choice is a step function whether the
  evidence is a witness's reliability, a base rate, or a hedged sentence in a real
  ticket.
- **Part of v2's second claim is revised.** Noul looked compressed in experiment 01
  (slope 0.45), but it is close to exact on stated base rates (mean abs. error
  0.015–0.027). The compression is specific to evidence given as a witness's accuracy,
  not a general property of Noul. Noul is the primitive that represents probability.
- **Practical consequence.** A workflow that thresholds Choice at 0.9 cannot tell a
  55/45 base rate from a 95/5 one. Asking each option as a separate Noul question gets
  close to the correct probability on this kind of evidence. That makes the fix in
  goal item 05 the obvious next experiment.

## Caveats

- There are three templated domains and one cue ladder, all written by us. The next
  step is real data with human vote shares, such as ChaosNLI or the Ibrahim & Zaki
  release.
- The base rates are stated as counts. That is still structured, even though it is
  closer to real case notes than a stated reliability.
- The ladder's L1 cues aren't matched in strength across the two teams, so the billing
  L1 result (0.4) is not the mirror image of the technical L1 result (0.0). The
  within-team shape is the result, not the cross-team comparison.
- One day, one model build (1.13.0), no bootstrap confidence intervals yet (goal item 02).

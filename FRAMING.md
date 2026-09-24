# Framing

The argument of the paper, what would falsify it, and how it has changed. Read this
before writing or revising any section, and update it whenever the framing moves.

## v2.9 update (2026-09-24, after experiment 12): current

- **The real-data headline is restated.** Choice returns many exact zeros, so the KL gap
  depends on the clip ε. Meng 2026, an independent Jev audit, found the same for fitted
  temperatures. Normalized Noul is closer on every metric. The clip-free gap is 1.3–2.1×
  (Brier, TVD); the KL gap is 1.4–6× depending on ε. Exp 05's excess-log-loss gap is
  33–58×; Brier ~20× (clip-free).
- **Withdrawn:** "fitted temperatures don't transfer across tasks". Absolute T values are
  clip-dependent. Kept: one fitted temperature brings Choice level with Noul at every ε.

## v2.8 update (2026-09-23, after experiment 11)

- **The "RL causes semantic-cue saturation" hypothesis from v2.7 is rejected** for the
  one implementation where it could be tested. The rebuilt pre-RL warm-up of eve-rlcd
  passed the reproduction gate (test accuracy 0.736 vs 0.748). It already saturates on
  the hedged cue (L2 0.991), more than after RL (0.952). It also already ignores stated
  base rates. RL moved it slightly *toward* calibration: witness slope 0.82 → 0.62.
- **Paper implication.** The four-model section reports that the failures come from the
  base model and task training, not from the calibration-reward RL stage. The paper makes
  **no** claim that RLCD causes anything. Jev's stated-probability step stays a Jev-level
  finding.
- **The GPU follow-up (goal item 11) is closed** by its own pre-registered rule.

## v2.7 update (2026-09-22, after experiment 10)

- **The most faithful open RLCD implementation** (anthonym21, real REINFORCE with a
  Brier-gradient reward) scores **"Neither"** on the pre-registered rule. It shows no
  numeric step (jump ≈ 0) and doesn't track stated base rates either (Choice error
  0.18–0.28). Across all four models, **only Jev's Noul recovers stated probabilities.**
- **Claim 2 stays scoped to Jev.** The stated-probability step is Jev-specific, and
  nothing yet attributes it to calibration-reward RL.
- **New cross-model hypothesis** (suggestive, n = 4, not a claim): the three
  RL-trained models saturate on a hedged worded cue (0.94–1.00). The supervised model
  stays graded (0.59). The eve-rlcd repo releases code and data, so an SFT-vs-RLCD
  controlled training run could test this. It is the natural follow-up paper, or v2 of
  this one.
- **Paper structure implication.** The cross-model section is now a real section:
  "the same cheap audit, applied to four 'calibrated decision' models, finds four
  different failures". It is no longer a footnote.

## v2.6 update (2026-09-22, after experiment 06b on DICES-350)

- **Confirmed on a second real task, with all four predictions pre-registered.**
  - Zero-shot, Noul beats Choice on KL: 0.23 vs 0.95.
  - Choice ranks human disagreement as well or better (Spearman 0.48 vs 0.38).
  - One fitted temperature per task makes them equivalent (gap 0.005).
- **The paper's claim, final form (pending the draft):**
  1. *Jev's Choice probabilities are overconfident on every evidence type we tested.*
  2. *On evidence that states a probability, Choice collapses to a step, and no
     rescaling recovers it. Ask Noul (about 50× lower error on base rates).*
  3. *On genuine human ambiguity (NLI, safety), Choice keeps the ranking signal but is
     far too extreme (fitted T ≈ 6–9). With no labelled data, Noul is the safer
     default (KL about 4–6× lower). With a few hundred items labelled by several
     raters, temperature-scaled Choice matches it.*
  4. *Neither question type is coherent on its own terms. Noul's complementary answers
     don't sum to 1 (they sum to 0.88–1.23), and temperatures don't transfer across
     tasks. So "calibrated out of the box" does not hold for either.*

## v2.5 update (2026-09-22, after experiment 06 on ChaosNLI)

- **The external falsifier did not fire.** On 300 real NLI items with 100 human votes
  each, normalized Noul beats Choice on KL to the human distribution: 0.19 vs 1.15, CI
  on the difference [−1.10, −0.82].
- **Two qualifications from the same experiment:**
  - Noul does **not** rank ambiguity better (Spearman about 0.46 for both; P3 null).
  - A single fitted temperature (T ≈ 6.3) brings Choice level with Noul (held-out KL
    0.163 vs 0.176). This was exploratory.

  So the "step function that discards information" claim is **specific to evidence
  that states a probability** (experiments 01, 03, 05). On intrinsic ambiguity, Choice
  is monotone but far too extreme.
- **The paper's claim, revised:** *Jev's Choice is a decision with an overconfident
  number attached. On stated-probability evidence, it is a step that no rescaling
  repairs. On genuine ambiguity, it is recoverable only with labelled
  disagreement data. Noul is close to calibrated with nothing fitted in both cases,
  after normalizing across options. So: ask Noul, normalize, and treat both as weak
  ambiguity detectors.*

## v2.4 update (2026-09-22, after the true holdout and review)

- **The result holds on a true holdout** (records 101–105, 1,445 new calls), matching
  the earlier split to within about 0.002 Brier. Raw Choice 0.0665 [0.060, 0.073];
  normalized Noul 0.0034 [0.003, 0.004]. Measured as excess log loss (KL from the
  exact target), the gap is **about 57×** (0.663 vs 0.012).
- **Reframed contribution.** The fix is *switching question type*: raw Noul alone
  reaches 0.0060, which is about 96% of the gain. Normalizing across options adds a
  small significant gain on witness evidence, where Noul's complements don't sum to 1,
  and none on base rates. Don't present normalization as a novel method.
- **Cross-model (exp 09) is a limitation, not a finding.** Two unofficial ~0.6B
  community checkpoints can't reject or support claims about RLCD. See
  `experiments/09-open-rlcd-laya.md`.
- **The paper's claim, in one sentence:** *Jev's Choice probability is a decision with
  a confidence-shaped number attached. Its Noul probability tracks exact targets. So
  any workflow that needs a probability should ask Noul questions, and a no-evidence
  plus exact-target audit exposes the difference cheaply.*

## v2.3 update (2026-09-22, after experiment 05)

- **The proposed fix works on held-out repetitions.** Per-option Noul normalization
  reduces overall Brier error from 0.0687 for raw Choice to 0.0034, log loss from
  1.2468 to 0.5641, and smooth ECE from 0.2307 to 0.0264.
- **The fix is not generic recalibration.** A witness-fitted Noul logit multiplier
  (1.767) is best on witness evidence but makes stated natural base rates worse.
  Temperature scaling improves Choice substantially but remains worse than normalized
  Noul. Content-free prior division fails. The missing information cannot be repaired
  by removing one stable label prior.
- **The practical thesis is now supported.** Choice should be read as action selection.
  When software needs a probability distribution, ask one Noul question per mutually
  exclusive option and normalize the answers. This changes the query, not merely the
  presentation of Choice's reported confidence.
- **Holdout qualification.** TypeSafe began returning HTTP 403 (error 1010) when a new
  records-101–105 collection was attempted. The result instead fits repetitions 1–5
  and scores the pre-existing repetitions 6–10. This prevents direct fitting leakage
  but is not a temporal or template holdout.

**Next falsifier:** on real questions with human vote shares, normalized Noul should
track agreement better than raw Choice. Failure there would restrict the fix to
explicit-probability synthetic tasks.

## v2.2 update (2026-09-22, after experiment 09)

- **The paper is about Jev first, not a standardized RLCD algorithm.** TypeSafe has
  not published enough of Reinforcement Learning for Calibrated Decisions to identify
  an interchangeable model family. Independent projects use the name for materially
  different objectives: Laya claims policy-gradient RL with proper scoring rewards;
  Qwen3-0.6B-RLCD explicitly uses supervised proper-scoring-rule fine-tuning.
- **Jev's full Choice/Noul split does not generalize.** On the same exact-target
  experiments, neither open checkpoint reproduced both Jev's sharp Choice threshold
  and its near-exact stated-base-rate Noul:
  - Laya's Choice saturated on semantic ticket cues, but barely moved across numeric
    base rates; its Noul base-rate MAE was 0.153–0.209.
  - Qwen's semantic response was graded, its 45%→55% numeric Choice change was only
    -0.016 to +0.059, and its Noul base-rate MAE was 0.182–0.195.
- **The cross-model claim is deliberately narrower.** A bounded decision interface
  and a calibration-oriented training claim do not guarantee that explicitly stated
  probabilities will be recovered. The failure mode is model-dependent: Jev
  thresholds, while the open checkpoints often ignore or compress the number.
- **The strongest Jev result remains intact.** For Jev 1.13.0, Choice behaves like a
  decision boundary with certainty-shaped output, while Noul can represent stated
  base rates. That is an audited property of Jev, not evidence that RLCD necessarily
  causes it.
- **Experiment 05 is now decisive for the practical contribution.** The remaining
  question is whether Jev's raw outputs can be transformed into honest probabilities
  on held-out calls. The fit set is the existing experiment 01/03 data; repeats
  101–105 are held out. Exact-target scoring excludes the subjective cue ladder.

**What would falsify v2.2's proposed fix:** normalized per-option Noul, slope-only
Platt scaling, Choice temperature scaling, and content-free prior division all fail
to improve held-out Brier score, log loss, and smooth ECE. That is a valid negative
result: the audit would then recommend changing the query primitive rather than
post-hoc repair.

## v2.1 update (2026-09-22, after experiment 03)

- **The falsifier did not fire.** Choice is still a step function with natural
  evidence. A 55/45 base rate gives 0.90–0.996. A hedged ticket sentence ("I'm not
  sure, but I may have been charged…") gives 1.00.
- **Claim 2 is revised.** Noul is **near-exact** on stated base rates (mean abs. error
  0.015–0.027). Its compression in experiment 01 (slope 0.45) is specific to evidence
  given as a witness's accuracy. It is not a general property of Noul.
- **The sharpened thesis.** Choice reports a *decision* with a certainty-shaped number
  attached. Noul reports a *probability*. The fix is the obvious one: ask each option
  as a separate Noul question and normalize. That is now claim 3, and goal item 05
  tests it.
- **The prior with zero or tied evidence follows the wording of the evidence, not the
  label.** Examples: heads; approved, billing and granted at ties; technical for a
  neutral ticket.

## v2 (2026-09-22, after experiment 01)

**TypeSafe sells Jev's Choice output as a calibrated probability. It behaves like a
softened decision instead. Noul is a real probability, but it is compressed toward 0.5.
Aggregate calibration scores on real benchmarks can't tell these apart.**

Experiment 01 scores Jev against an exact Bayesian target: a hidden fair coin plus
witnesses of stated accuracy. On that test:

1. **Choice is a step function.** One witness who is right 55% of the time pushes
   Choice to about 0.90 on the side they name (true value 0.55). Beyond one weak
   witness, the strength of the evidence barely matters. This holds for all six label
   sets. It is the largest and most robust effect we have measured.
2. **Noul tracks the evidence, but its answers are compressed.** It is monotone in
   the truth, with a logit slope of 0.45: a 99%-reliable witness yields 0.80. It is
   nearly unbiased with zero evidence (0.52) and with balanced evidence (0.505). Its
   heads and tails answers don't sum to 1 (mean gap 0.083).
3. **The zero-evidence prior is real but fragile.** With no evidence, Choice gives
   P(heads) of 0.83–0.93 under every label set. When options have descriptions, the
   prior follows the word "heads", not the key name or the list position. A single
   55% witness cuts it from +0.35 to +0.06. Strong evidence cuts it to about +0.01.

**Consequence for the paper.** The v1 headline was "the zero-evidence prior leaks into
real decisions". It **mostly failed its own kill check**: one sentence of evidence
removes about 80% of the prior. The label-prior story drops to a secondary section.
The main claim becomes that calibration depends on the primitive, and that the right
fix is different for each one:

- Choice needs a decision-aware reading. Its probability is not an estimate of
  frequency.
- Noul needs recalibration. It has a stable slope, so this should be cheap.
- A proposed method: decompose a Choice into one Noul per option and normalize. We
  predict it beats raw Choice. Untested.

**Prior work we must not overclaim against:**
- Arcturus Labs (16 Sep): Choice gives 0.99 on a 60%-heads coin; Noul is within 9 points.
- simonmesmith: Choice error 21 points, Noul error 5.6 points, on stated-probability cases.

Both are informal. Ours adds:
- an exact Bayesian target across a full sweep of evidence strength;
- a factorial design separating the word, the key name, and the list position;
- the observation that Noul is compressed (slope 0.45), not just "close";
- the fix experiments.

**What would falsify v2:**
- Choice tracks evidence strength once the evidence is realistic text rather than
  stated reliability numbers. Experiment 03 tests this.
- Noul's compression doesn't generalize beyond witness-style evidence.
- The Noul decomposition doesn't beat Choice once recalibrated. That would be a null
  result for the method, and we'd still report it.

Candidate titles:
- *Calibrated Decisions Are Not Calibrated Probabilities: Auditing TypeSafe's Jev*
- *One Weak Witness: How a Calibration-Trained Decision Model Reads Evidence*
- *Step Functions and Shrinkage: The Two Calibrations of a System One Model*

## v1 (2026-09-22, before any sweep): superseded

"Jev is calibrated on average but biased under zero evidence." The claims, from most
to least confident:
- a label-not-position prior;
- a dose-response of that prior against evidence;
- the prior predicts real errors;
- content-free calibration fixes it;
- RLCD, not Jev specifically, is the cause.

Pre-declared kill criterion: "if one sentence of real evidence makes the bias
disappear, and it doesn't predict errors on real items, it's a curiosity."
Experiment 01 met the first half of that criterion.

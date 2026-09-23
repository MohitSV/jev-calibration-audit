# Jev / RLCD calibration: literature and community review (2026-09-22)

RLCD here means TypeSafe's "Reinforcement Learning for Calibrated Decisions". It is not
the 2023 "RLCD: Reinforcement Learning from Contrastive Distillation" (Yang et al.,
arXiv 2307.12950, ICLR 2024). Spell out the full name in any paper to avoid the clash.

## Verdict
- **The finding is not new.** At least 5 independent write-ups from 16–22 Sep report a
  content-free Choice collapsing to the "first-looking" label: heads, "1", item1.
- **No formal paper tests it.** Ibrahim & Zaki (arXiv 2609.24574) conclude "no distinctive
  first-option bias", but they measured on real content only. Rafe & Das
  (2609.24052) explicitly left order and wording untested.
- **TypeSafe publishes no ECE, no reliability curve and no RLCD paper.** The CEO calls RLCD
  "novel, unpublished" (Latent Space, 21 Sep).
- **Window:** about 9 Jev arXiv preprints appeared in 3 days, so move fast.

## Caveat for our own pilot
JS re-sorts integer-like object keys ("1"–"6"), so our die-order permutations probably
never reached the model in a different order (Hayashi). The coin and option_1/option_2
results use non-integer keys and stand. Re-run the die with `face_N` keys.

## Direct replications of our finding (cite these)
| Source | Finding |
|---|---|
| [Hayashi, "Jev Does Not Play Dice"](https://kantahayashiai.github.io/posts/jev-does-not-play-dice/) | Die: "1" in 400/400 at 0.83, 19% accuracy. Coin: heads 200/200 at 0.92. Noul is close for 2–4 options but inflated for 10–20 options. Order never permuted. |
| [alfonsograziano/jev-alphabetical-prior](https://github.com/alfonsograziano/jev-alphabetical-prior) | Content-free item1…item16: item1 takes 83%. "a"-prefix ×2.04. Order is numeric, not alphabetical. Pre-registered. |
| [simonmesmith/jev-probability-experiment](https://github.com/simonmesmith/jev-probability-experiment) | Coin 0.93–0.97 heads in every order. Die face 1 at 0.77–0.87. The probability instruction makes error worse (21.1 → 29.4 pp). Noul error 5.6 pp. |
| [HappyAny/jev-random-bit-experiments](https://github.com/HappyAny/jev-random-bit-experiments) | Fair bit ×2000: 1,999 zeros. |
| [Arcturus Labs](https://arcturus-labs.com/blog/2026/09/16/typesafes-jev-trades-text-generation-for-instant-calibrated-decisions/) | A 60%-heads coin returns 0.99 on Choice. Noul is within 9 pp. |
| [HN juskrey](https://news.ycombinator.com/item?id=49765813) | Die ×400 all "1" at about 83%. |

## Calibration on real tasks (Jev looks OK on average)
- [Ibrahim & Zaki 2609.24574](https://arxiv.org/abs/2609.24574): 18 CSS tasks. Median ECE 0.157, overconfident (T=2.66). Per-call data released.
- [Rafe & Das 2609.24052](https://arxiv.org/abs/2609.24052): crash narratives. ECE 0.012–0.046. The 0.01 output grid puts a floor on the error.
- [Praveenrajus/jev-bench (HF)](https://huggingface.co/datasets/Praveenrajus/jev-bench): 22 datasets, ECE 0.113. On ChaosNLI, confidence stays flat at about 0.83 whether humans agree or split.
- [jujumilk3 audit](https://github.com/jujumilk3/jev-calibration-audit): with the question removed, it still scores 0.38–0.46 (chance about 0.15). Remove "unknown" on KoBBQ and 79% of answers are stereotyped.
- [SamuelSacco evidence ledger](https://github.com/SamuelSacco/jev-exploration/blob/main/docs/claims-audit.md): ECE is 2–2.5× the noise floor. A stable Platt slope lets 50 labels fix 74% of the error.
- [PriorBench](https://github.com/priorbench/jev): random letters get 0.97 confidence. No abstention without an explicit option.
  With *balanced but present* evidence (3 vs 3 sentences), Choice does hedge (p=0.32, conf 0.45), so the collapse is specific to *absent* evidence.
  This makes a dose-response curve over evidence strength the key experiment.
- [Jevals](https://jevals.com/notes/2026-09-18/): Banking77 order-flip rate 10.3%, vs 8.7–36% for LLMs.
- [bernoulli.app](https://bernoulli.app/articles/is-jev-confident): the `confidence` field is (p_max − 1/K)/(1 − 1/K), so it adds nothing new.
- Consensus: the ≥0.9 band is reliable, the 0.5–0.95 middle is near chance, and Noul is better than Choice, which is better than Score.

## Prior art from LLMs (must-cite)
1. Zhao et al., *Calibrate Before Use*, ICML 2021: a content-free probe plus dividing out the prior. This is our coin test.
2. Zheng et al., *LLMs Are Not Robust Multiple Choice Selectors* (PriDe), ICLR 2024: label-ID bias rather than position bias, and debias instructions fail.
3. Gupta et al., *Enough Coin Flips Can Make LLMs Act Bayesian*, ACL 2025: a zero-shot heads prior of 60–80%.
4. GPT-4 Technical Report, Fig. 8: post-training hurt MMLU calibration (0.007 → 0.074).
5. Damani et al., *Beyond Binary Rewards* (RLCR), ICLR 2026: the closest published analogue to RLCD.

Also: Van Koevering & Kleinberg 2024 (coin flips); Zhang et al., *Forcing Diffuse
Distributions*, COLM 2024; Kadavath et al. 2022; Hébert-Johnson et al. 2018
(multicalibration); ECE critiques (Kumar et al. 2019; Roelofs et al. 2022; Błasiok &
Nakkiran 2024 smooth ECE).

## Gaps nobody has tested yet (paper angle)
1. **Does the content-free prior predict real errors?** Test on low-evidence items in real benchmarks. The Ibrahim & Zaki per-call data can be reused.
2. **Does removing it help?** Compare contextual calibration and permutation-averaging against temperature and Platt scaling, measured by ECE and Brier.
3. **Blind two-option labels with permuted order.** This is our option_1/option_2 test and is unpublished.
4. **Does the prior shrink as evidence strength increases?** Flagged as untested.
5. **Reported vs true probability sweep** across Choice, Score and Noul on random events with known odds.
6. **RLCD objective vs Jev specifically:** run the same probes on open reproductions (Qwen3-0.6B-RLCD, Laya).
7. **Framing:** calibrated on average does not mean symmetric or invariant. This is a subgroup or individual calibration failure.

The underlying search covered about 45 write-ups, 9 arXiv preprints and 60+ prior-art papers.
Platforms blocked: X (402), Reddit, Bluesky, Discord. Semantic Scholar was rate-limited.

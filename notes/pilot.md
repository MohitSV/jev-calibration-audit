# Pilot (2026-09-22, before the project folder existed)

The pilot ran from `Personal/random_q7x3k9/Gem/jev_bias.py`, 100 trials per condition,
on jev-1.13.0. Raw data is copied in `pilot-results/`.

- **Fair coin, Choice:** heads wins 100/100 in either order.
  - Heads listed first: P(heads) = 0.82.
  - Tails listed first: P(heads) = 0.87.
- **Neutral keys A/B with descriptions:**
  - A = heads: P(heads) = 0.87.
  - A = tails: P(heads) = 0.705.
- **Blind keys `option_1`/`option_2`, no descriptions:** P(option_1) is about 0.84
  whichever key is listed first. The bias is on the key name, not the position.
- **The "be unbiased and follow probability" instruction:** no help. P(heads) was 0.82
  with it and 0.80 without.
- **Fair die:** "1" got about 0.40 and "2" about 0.05 in every order.
  **This result is invalid.** The keys "1"–"6" look like integers, so the server
  probably re-sorted them and our order manipulation never reached the model.
  Rerun it with `face_N` keys (goal item 07).

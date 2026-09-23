# Jev bias run — 20260922-140403

Total input tokens: 70,384 (≈ $0.0030 at $0.042/M)

| experiment | n | option order | argmax picks | mean P(option) | sampled from P | 1st-option wins | max err vs fair | distinct responses |
|---|---|---|---|---|---|---|---|---|
| coin_blind_option_1_first | 100 | option_1 > option_2 | option_1:100 | option_1:0.837, option_2:0.163 | option_1:87, option_2:13 | 100% | 0.337 | 14 |
| coin_blind_option_2_first | 100 | option_2 > option_1 | option_1:100 | option_2:0.158, option_1:0.842 | option_1:85, option_2:15 | 0% | 0.342 | 13 |

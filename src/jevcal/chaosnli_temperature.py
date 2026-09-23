"""Experiment 06, EXPLORATORY (not pre-registered): can one fitted temperature close the
Choice-vs-Noul gap on ChaosNLI?

    cd src && ../venv/bin/python -m jevcal.chaosnli_temperature

Fits T on a random 150 items and scores KL on the other 150, over 20 random splits.
"""

import json
import math
import pathlib
import random
import statistics

from jevcal import chaosnli as c


def temper(d, t):
    z = {k: math.log(max(v, c.EPS)) / t for k, v in d.items()}
    m = max(z.values())
    e = {k: math.exp(v - m) for k, v in z.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def fit_t(items, uids, method):
    grid = [0.25 * 1.08**i for i in range(80)]
    return min(grid, key=lambda t: statistics.mean(
        c.kl(items[u]["human"], temper(items[u][method], t)) for u in uids))


def main():
    rows = [json.loads(x) for x in pathlib.Path(c.OUT).read_text().splitlines()]
    items = c.predictions(rows)
    uids = sorted(items)
    out = {}
    for method in ("choice_avg", "noul_norm"):
        runs = []
        for seed in range(20):
            us = uids[:]
            random.Random(seed).shuffle(us)
            fit, test = us[:150], us[150:]
            t = fit_t(items, fit, method)
            runs.append({"T": t, "kl": statistics.mean(
                c.kl(items[u]["human"], temper(items[u][method], t)) for u in test)})
        out[method] = {"T_median": round(statistics.median(r["T"] for r in runs), 2),
                       "heldout_kl_mean": round(statistics.mean(r["kl"] for r in runs), 4),
                       "heldout_kl_range": [round(min(r["kl"] for r in runs), 3),
                                            round(max(r["kl"] for r in runs), 3)]}
    hard_zero = sum(1 for u in uids if any(items[u]["human"][k] >= 0.1
                                           and items[u]["choice_avg"][k] < 0.01 for k in c.LABELS))
    out["items_where_choice_lt1pct_on_option_with_ge10pct_humans"] = hard_zero
    path = c.OUT.with_name("06-chaosnli-temperature.exploratory.json")
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

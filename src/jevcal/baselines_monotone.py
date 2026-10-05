"""Experiment 14 (robustness, offline): uniform baseline, per-stratum calibration after one
temperature, and a flexible monotone (isotonic) repair of Choice.

    cd src && ../venv/bin/python -m jevcal.baselines_monotone

Prompted by independent work (Khosla 2026, Porcedda 2026). Writes
experiments/data/14-baselines-monotone.json. No API calls.
"""

import json
import math
import statistics

from jevcal import chaosnli as c
from jevcal import dices as d
from jevcal.calibrate import load_evidence, load_natural, normalized_noul
from jevcal.client import ROOT

DATA = ROOT / "experiments" / "data"
GRID = [0.25 * 1.08**i for i in range(80)]
UNI3 = {k: 1 / 3 for k in c.LABELS}


def _load():
    ic = c.predictions([json.loads(x) for x in (DATA / "06-chaosnli.jsonl").read_text().splitlines()])
    idd = d.predictions([json.loads(x) for x in (DATA / "06b-dices.jsonl").read_text().splitlines()])
    return ic, sorted(ic), idd, sorted(idd, key=int)


def _t3(dist, t, eps=1e-6):
    z = {k: math.log(max(v, eps)) / t for k, v in dist.items()}
    m = max(z.values())
    e = {k: math.exp(v - m) for k, v in z.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def _kl3(h, q):
    return sum(h[k] * math.log(h[k] / max(q[k], 1e-6)) for k in c.LABELS if h[k] > 0)


def _brier3(h, q):
    return sum((h[k] - q[k]) ** 2 for k in c.LABELS)


def uniform_baseline(ic, uc, idd, ud):
    out = {"chaosnli": {}, "dices": {}}
    getters3 = {"uniform": lambda u: UNI3, "choice": lambda u: ic[u]["choice_avg"],
                "noul_norm": lambda u: ic[u]["noul_norm"]}
    getters2 = {"uniform": lambda u: 0.5, "choice": lambda u: idd[u]["choice_avg"],
                "noul_norm": lambda u: idd[u]["noul_norm"]}
    for name, g in getters3.items():
        out["chaosnli"][name] = {"brier": round(statistics.mean(_brier3(ic[u]["human"], g(u)) for u in uc), 3)}
        for b, lab in ((0, "hardest_bin"), (4, "easiest_bin")):
            us = [u for u in uc if ic[u]["bin"] == b]
            out["chaosnli"][name][lab] = round(statistics.mean(_brier3(ic[u]["human"], g(u)) for u in us), 3)
    for name, g in getters2.items():
        out["dices"][name] = {"brier": round(statistics.mean((idd[u]["human"] - g(u)) ** 2 for u in ud), 3),
                              "abs_err": round(statistics.mean(abs(idd[u]["human"] - g(u)) for u in ud), 3)}
        for b, lab in ((0, "hardest_bin"), (4, "easiest_bin")):
            us = [u for u in ud if d.bin_of(idd[u]["human"]) == b]
            out["dices"][name][lab] = round(statistics.mean((idd[u]["human"] - g(u)) ** 2 for u in us), 3)
    return out


def per_stratum_after_temperature(ic, uc):
    tc = min(GRID, key=lambda t: statistics.mean(_kl3(ic[u]["human"], _t3(ic[u]["choice_avg"], t)) for u in uc))
    tn = min(GRID, key=lambda t: statistics.mean(_kl3(ic[u]["human"], _t3(ic[u]["noul_norm"], t)) for u in uc))
    rows = []
    for b in range(5):
        us = [u for u in uc if ic[u]["bin"] == b]
        hs = statistics.mean(max(ic[u]["human"].values()) for u in us)

        def gap(m, t=None, us=us, hs=hs):
            return statistics.mean(max((ic[u][m] if t is None else _t3(ic[u][m], t)).values())
                                   for u in us) - hs
        rows.append({"bin": b, "human_top_share": round(hs, 2),
                     "gap_choice_raw": round(gap("choice_avg"), 3),
                     "gap_choice_T": round(gap("choice_avg", tc), 3),
                     "gap_noul_raw": round(gap("noul_norm"), 3),
                     "gap_noul_T": round(gap("noul_norm", tn), 3)})
    return {"T_choice": round(tc, 2), "T_noul": round(tn, 2), "by_bin": rows}


def _pav(xs, ys):
    blocks = []
    for x, y in sorted(zip(xs, ys, strict=True)):
        blocks.append([x, x, y, 1])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            b, a = blocks.pop(), blocks.pop()
            n = a[3] + b[3]
            blocks.append([a[0], b[1], (a[2] * a[3] + b[2] * b[3]) / n, n])
    return blocks


def isotonic_repair():
    fit = [*load_evidence(DATA / "01-evidence-sweep.jsonl"), *load_natural(DATA / "03-natural-evidence.jsonl")]
    ho = [*load_evidence(DATA / "05-heldout-evidence-sweep.jsonl"),
          *load_natural(DATA / "05-heldout-natural-evidence.jsonl")]
    blocks = _pav([o.choice for o in fit], [o.target for o in fit])

    def apply(x):
        return next((v for _, hi, v, _ in blocks if x <= hi), blocks[-1][2])

    def brier(ps, ys):
        return statistics.mean((p - y) ** 2 for p, y in zip(ps, ys, strict=True))
    out = {}
    for lab, sub in (("overall", ho), ("witness", [o for o in ho if o.dataset == "witness"]),
                     ("base_rate", [o for o in ho if o.dataset == "base_rate"])):
        ys = [o.target for o in sub]
        out[lab] = {"n": len(sub), "raw_choice": round(brier([o.choice for o in sub], ys), 4),
                    "isotonic_choice": round(brier([apply(o.choice) for o in sub], ys), 4),
                    "normalized_noul": round(brier([normalized_noul(o) for o in sub], ys), 4)}
    out["note"] = "Isotonic map fitted on the same templates as the holdout, so it is optimistic."
    return out


def main():
    ic, uc, idd, ud = _load()
    res = {"uniform_baseline": uniform_baseline(ic, uc, idd, ud),
           "per_stratum_after_temperature": per_stratum_after_temperature(ic, uc),
           "isotonic_repair": isotonic_repair()}
    (DATA / "14-baselines-monotone.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

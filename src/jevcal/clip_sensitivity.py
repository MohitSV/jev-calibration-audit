"""Experiment 12 (robustness): how much do log-based headlines depend on clipping exact zeros?

    cd src && ../venv/bin/python -m jevcal.clip_sensitivity

Jev's Choice returns exact 0/1 probabilities often, so KL, log loss and fitted temperatures
depend on the epsilon used to clip them before taking logs. Brier, TVD and |p - p_h| do not.
Writes experiments/data/12-clip-sensitivity.json. Offline; no API calls.
"""

import json
import math
import statistics

from jevcal import chaosnli as c
from jevcal import dices as d
from jevcal.calibrate import load_evidence, load_natural, normalized_noul
from jevcal.client import ROOT

DATA = ROOT / "experiments" / "data"
EPSILONS = (1e-6, 1e-4, 1e-3, 1e-2, 5e-2)
GRID = [0.25 * 1.08**i for i in range(80)]


def _kl3(h, q, eps):
    return sum(h[k] * math.log(h[k] / max(q[k], eps)) for k in c.LABELS if h[k] > 0)


def _kl2(h, p, eps):
    p = min(max(p, eps), 1 - eps)
    return sum(a * math.log(a / b) for a, b in ((h, p), (1 - h, 1 - p)) if a > 0)


def _temper3(dist, t, eps):
    z = {k: math.log(max(v, eps)) / t for k, v in dist.items()}
    m = max(z.values())
    e = {k: math.exp(v - m) for k, v in z.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def _temper2(p, t, eps):
    p = min(max(p, eps), 1 - eps)
    return 1 / (1 + math.exp(-math.log(p / (1 - p)) / t))


def _excess_ll(ps, ys, eps):
    out = 0.0
    for p, y in zip(ps, ys, strict=True):
        p = min(max(p, eps), 1 - eps)
        yc = min(max(y, 1e-12), 1 - 1e-12)
        out += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        out += y * math.log(yc) + (1 - y) * math.log(1 - yc)
    return out / len(ps)


def real_data():
    ic = c.predictions([json.loads(x) for x in (DATA / "06-chaosnli.jsonl").read_text().splitlines()])
    idd = d.predictions([json.loads(x) for x in (DATA / "06b-dices.jsonl").read_text().splitlines()])
    uc, ud = sorted(ic), sorted(idd, key=int)
    out = {"exact_zeros": {
        "chaosnli_choice_entries": sum(ic[u]["choice_avg"][k] == 0 for u in uc for k in c.LABELS),
        "chaosnli_noul_entries": sum(ic[u]["noul_norm"][k] == 0 for u in uc for k in c.LABELS),
        "chaosnli_total_entries": 3 * len(uc),
        "dices_choice_items_at_0_or_1": sum(idd[u]["choice_avg"] in (0, 1) for u in ud),
        "dices_noul_items_at_0_or_1": sum(idd[u]["noul_norm"] in (0, 1) for u in ud),
        "dices_items": len(ud)}, "kl_by_eps": {}, "fitted_choice_T_by_eps": {}}
    for eps in EPSILONS:
        a = statistics.mean(_kl3(ic[u]["human"], ic[u]["choice_avg"], eps) for u in uc)
        b = statistics.mean(_kl3(ic[u]["human"], ic[u]["noul_norm"], eps) for u in uc)
        x = statistics.mean(_kl2(idd[u]["human"], idd[u]["choice_avg"], eps) for u in ud)
        y = statistics.mean(_kl2(idd[u]["human"], idd[u]["noul_norm"], eps) for u in ud)
        out["kl_by_eps"][str(eps)] = {"chaosnli_choice": round(a, 3), "chaosnli_noul": round(b, 3),
                                      "chaosnli_ratio": round(a / b, 2), "dices_choice": round(x, 3),
                                      "dices_noul": round(y, 3), "dices_ratio": round(x / y, 2)}
    for eps in EPSILONS[:4]:
        tc = min(GRID, key=lambda t: statistics.mean(
            _kl3(ic[u]["human"], _temper3(ic[u]["choice_avg"], t, eps), 1e-6) for u in uc))
        td = min(GRID, key=lambda t: statistics.mean(
            _kl2(idd[u]["human"], _temper2(idd[u]["choice_avg"], t, eps), 1e-6) for u in ud))
        out["fitted_choice_T_by_eps"][str(eps)] = {
            "chaosnli_T": round(tc, 2),
            "chaosnli_kl": round(statistics.mean(
                _kl3(ic[u]["human"], _temper3(ic[u]["choice_avg"], tc, eps), 1e-6) for u in uc), 3),
            "dices_T": round(td, 2),
            "dices_kl": round(statistics.mean(
                _kl2(idd[u]["human"], _temper2(idd[u]["choice_avg"], td, eps), 1e-6) for u in ud), 3)}
    return out


def exact_target_holdout():
    ho = [*load_evidence(DATA / "05-heldout-evidence-sweep.jsonl"),
          *load_natural(DATA / "05-heldout-natural-evidence.jsonl")]
    ys = [o.target for o in ho]
    out = {"n": len(ho), "raw_choice_exact_0_or_1": sum(o.choice in (0, 1) for o in ho),
           "excess_log_loss_by_eps": {}}
    for eps in EPSILONS[:4]:
        a = _excess_ll([o.choice for o in ho], ys, eps)
        b = _excess_ll([normalized_noul(o) for o in ho], ys, eps)
        out["excess_log_loss_by_eps"][str(eps)] = {"raw_choice": round(a, 3),
                                                   "normalized_noul": round(b, 4),
                                                   "ratio": round(a / b, 1)}
    return out


def main():
    res = {"real_data": real_data(), "exact_target_holdout": exact_target_holdout()}
    (DATA / "12-clip-sensitivity.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

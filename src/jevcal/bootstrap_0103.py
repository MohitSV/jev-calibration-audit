"""Goal item 02: bootstrap 95% CIs for the headline numbers of experiments 01 and 03.

    cd src && ../venv/bin/python -m jevcal.bootstrap_0103

Resamples whole repetitions (record/case numbers 1-10) with replacement, jointly across
cells, because the irrelevant "Record #N" text is shared by every cell in a repetition.
The CIs therefore cover repeat-to-repeat variation under fixed templates. They do not
cover variation across templates or wordings, which is the larger threat and is only
addressed by experiments 06/06b.
"""

import json
import random

from jevcal import analyze, analyze_natural
from jevcal.client import ROOT

DATA = ROOT / "experiments" / "data"


def headline_01(rows):
    s = analyze.summarize(rows)
    by_pair = {r["pair"]: r for r in s["decomposition_by_strength"]}
    return {
        "heads_pull_no_evidence": by_pair["none"]["semantic_effect"],
        "heads_pull_one_55pct_witness": by_pair["w1_q55_heads|w1_q55_tails"]["semantic_effect"],
        "label_effect_no_evidence": by_pair["none"]["label_effect"],
        "position_effect_no_evidence": by_pair["none"]["position_effect"],
        "noul_logit_slope": s["truth_slope_logit"]["noul"],
        "choice_mae_mean_over_label_sets": sum(v for k, v in s["mean_abs_error_vs_bayes"].items()
                                               if k != "noul") / 6,
        "noul_mae": s["mean_abs_error_vs_bayes"]["noul"],
    }


def headline_03(rows):
    s = analyze_natural.summarize(rows)
    out = {}
    for d, v in s["base_rate"].items():
        out[f"{d}_choice_jump_45_to_55"] = v["choice_jump_45_to_55"]
        out[f"{d}_choice_mae"] = v["mae_choice"]
        out[f"{d}_noul_mae"] = v["mae_noul"]
        out[f"{d}_choice_at_tie"] = v["choice_at_tie"]
    out["ticket_billing_L2_choice"] = s["cues"]["billing:L2"]["choice_p_billing"]
    out["ticket_billing_L2_noul"] = s["cues"]["billing:L2"]["noul_p_billing"]
    return out


def bootstrap(rows, headline, n_boot=1000, seed=0):
    reps = sorted({r["rep"] for r in rows})
    by_rep = {k: [r for r in rows if r["rep"] == k] for k in reps}
    point = headline(rows)
    rng = random.Random(seed)
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        sample = [r for k in rng.choices(reps, k=len(reps)) for r in by_rep[k]]
        for k, v in headline(sample).items():
            draws[k].append(v)

    def ci(v):
        v = sorted(v)
        return [round(v[int(0.025 * len(v))], 3), round(v[int(0.975 * len(v)) - 1], 3)]

    return {k: {"point": round(point[k], 3), "ci95": ci(draws[k])} for k in point}


def load(name):
    return [json.loads(x) for x in (DATA / name).read_text().splitlines()]


def main():
    out = {
        "note": "CIs resample repetitions 1-10 jointly across cells; template variation "
                "is not covered.",
        "exp01": bootstrap(load("01-evidence-sweep.jsonl"), headline_01),
        "exp03": bootstrap(load("03-natural-evidence.jsonl"), headline_03),
    }
    path = DATA / "02-bootstrap-0103.json"
    path.write_text(json.dumps(out, indent=2))
    for exp in ("exp01", "exp03"):
        for k, v in out[exp].items():
            print(f"{exp} {k:36} {v['point']:>7}  {v['ci95']}")


if __name__ == "__main__":
    main()

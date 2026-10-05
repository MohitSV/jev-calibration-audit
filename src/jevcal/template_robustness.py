"""Experiment 15: does Choice's step on stated base rates depend on the wording?

    cd src && ../venv/bin/python -m jevcal.template_robustness run
    cd src && ../venv/bin/python -m jevcal.template_robustness analyze

Pre-registered in experiments/15-template-robustness.md.
"""

import argparse
import json
import pathlib
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

from jevcal.backends import create_backend
from jevcal.client import ROOT
from jevcal.natural import DOMAINS

OUT = ROOT / "experiments" / "data" / "15-template-robustness.jsonl"
RATES = (0.35, 0.45, 0.50, 0.55, 0.65)
REPS = (301, 302)

TEMPLATES = {
    "count": {d: spec["text"] for d, spec in DOMAINS.items()},
    "percent": {
        "loan": "A small-business loan application is under review. At this bank, {pct}% of "
                "applications like this one are approved and the rest are denied. Nothing "
                "else distinguishes this application.",
        "routing": "A support ticket just arrived. {pct}% of tickets with exactly this wording "
                   "are resolved by the billing team and the rest by the technical team. "
                   "Nothing else is known about this ticket.",
        "refund": "A customer requested a refund. {pct}% of comparable refund requests this "
                  "quarter were granted and the rest were refused. This request is typical "
                  "of those.",
    },
    "probability": {
        "loan": "A small-business loan application is under review. Based on comparable past "
                "applications, the probability that this application will be approved is "
                "{prob}. Nothing else is known about it.",
        "routing": "A support ticket just arrived. Based on past tickets with the same "
                   "wording, the probability that the billing team will resolve it is {prob}; "
                   "otherwise the technical team will. Nothing else is known about it.",
        "refund": "A customer requested a refund. Based on comparable requests, the "
                  "probability that this refund will be granted is {prob}. Nothing else is "
                  "known about it.",
    },
}


def state(template, domain, p, rep):
    k = round(p * 20)
    text = TEMPLATES[template][domain].format(k=k, m=20 - k, pct=round(p * 100), prob=f"{p:.2f}")
    return f"Case {rep}. {text}"


def jobs():
    for t in TEMPLATES:
        for dom, spec in DOMAINS.items():
            (yk, yd), (nk, nd) = spec["yes"], spec["no"]
            for p in RATES:
                for rep in REPS:
                    for order, crit in (("yes_first", {yk: yd, nk: nd}), ("no_first", {nk: nd, yk: yd})):
                        yield ("choice", t, dom, p, rep, order, crit, yk)
                    yield ("noul", t, dom, p, rep, "noul", None, yk)


def run_one(backend, job):
    kind, t, dom, p, rep, variant, crit, yk = job
    spec = DOMAINS[dom]
    if kind == "choice":
        qs = {"q": {"type": "choice", "instructions": spec["instructions"], "criteria": crit}}
    else:
        qs = {"yes": {"type": "noul", "instructions": spec["noul_yes"]},
              "no": {"type": "noul", "instructions": spec["noul_no"]}}
    resp = backend.ask(state(t, dom, p, rep), qs)
    row = {"kind": kind, "template": t, "domain": dom, "p_true": p, "rep": rep, "variant": variant,
           "model": resp.get("model"), "input_tokens": resp.get("usage", {}).get("input_tokens")}
    if kind == "choice":
        row["p_yes"] = resp["answers"]["q"]["probabilities"].get(yk, 0.0)
    else:
        row |= {"noul_yes": resp["answers"]["yes"]["noul"], "noul_no": resp["answers"]["no"]["noul"],
                "p_yes": resp["answers"]["yes"]["noul"]}
    return row


def summarize(rows):
    cells = {}
    for r in rows:
        cells.setdefault((r["template"], r["kind"], r["p_true"]), []).append(r["p_yes"])
    mean = {k: statistics.mean(v) for k, v in cells.items()}
    out = {"n_calls": len(rows), "models": sorted({r["model"] for r in rows}),
           "input_tokens": sum(r["input_tokens"] or 0 for r in rows), "templates": {}}
    for t in TEMPLATES:
        c = {p: mean[(t, "choice", p)] for p in RATES}
        n = {p: mean[(t, "noul", p)] for p in RATES}
        out["templates"][t] = {
            "choice_curve": {str(p): round(v, 3) for p, v in c.items()},
            "noul_curve": {str(p): round(v, 3) for p, v in n.items()},
            "choice_jump_45_55": round(c[0.55] - c[0.45], 3),
            "choice_jump_35_65": round(c[0.65] - c[0.35], 3),
            "noul_jump_45_55": round(n[0.55] - n[0.45], 3),
            "choice_mae": round(statistics.mean(abs(v - p) for p, v in c.items()), 3),
            "noul_mae": round(statistics.mean(abs(v - p) for p, v in n.items()), 3),
        }
    jumps = [v["choice_jump_45_55"] for v in out["templates"].values()]
    out["P1_all_jumps_gt_0.3"] = all(j > 0.3 for j in jumps)
    out["P2_all_noul_mae_le_0.05"] = all(v["noul_mae"] <= 0.05 for v in out["templates"].values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "analyze"))
    ap.add_argument("-j", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "run":
        backend = create_backend("jev")
        all_jobs = list(jobs())
        print(len(all_jobs), "calls")
        lock = threading.Lock()
        with open(OUT, "w") as f, ThreadPoolExecutor(args.j) as pool:
            for row in pool.map(lambda jb: run_one(backend, jb), all_jobs):
                with lock:
                    f.write(json.dumps(row) + "\n")
        print("done")
    else:
        rows = [json.loads(x) for x in pathlib.Path(OUT).read_text().splitlines()]
        s = summarize(rows)
        OUT.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()

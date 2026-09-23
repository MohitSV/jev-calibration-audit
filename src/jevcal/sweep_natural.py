"""Experiment 03 runner.

    cd src && ../venv/bin/python -m jevcal.sweep_natural [-n REPEATS] [-j PARALLEL]
"""

import argparse
import datetime
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from jevcal import client
from jevcal.backends import create_backend
from jevcal.natural import (
    base_rate_items,
    choice_variants,
    cue_items,
    instructions,
    noul_questions,
)

OUT = client.ROOT / "experiments" / "data" / "03-natural-evidence.jsonl"


def jobs(repeats, rep_start=1):
    for item in [*base_rate_items(), *cue_items()]:
        for rep in range(rep_start, rep_start + repeats):
            for variant, criteria, yes_key in choice_variants(item):
                yield ("choice", item, rep, variant, criteria, yes_key)
            yield ("noul", item, rep, "noul", None, None)


def run_one(backend, job):
    kind, item, rep, variant, criteria, yes_key = job
    state = f"Case {rep}. {item.text}"
    if kind == "choice":
        qs = {"q": {"type": "choice", "instructions": instructions(item), "criteria": criteria}}
    else:
        qy, qn = noul_questions(item)
        qs = {"yes": {"type": "noul", "instructions": qy}, "no": {"type": "noul", "instructions": qn}}
    resp = backend.ask(state, qs)
    row = {"kind": kind, "arm": item.arm, "domain": item.domain, "level": item.level,
           "p_true": item.p_true, "rep": rep, "variant": variant, "state": state,
           "model": resp.get("model"), "input_tokens": resp.get("usage", {}).get("input_tokens")}
    if kind == "choice":
        a = resp["answers"]["q"]
        row |= {"order": list(criteria), "choice": a["choice"],
                "probabilities": a["probabilities"], "p_yes": a["probabilities"].get(yes_key, 0.0)}
    else:
        py, pn = resp["answers"]["yes"]["noul"], resp["answers"]["no"]["noul"]
        row |= {"noul_yes": py, "noul_no": pn, "p_yes": py}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--rep-start", type=int, default=1, help="first case number")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--backend", choices=("jev", "laya", "qwen-rlcd", "eve-rlcd"), default="jev")
    ap.add_argument("--model", help="backend model ID; defaults to the backend's standard model")
    ap.add_argument("--device", default="cpu", help="local device, e.g. cpu or mps")
    ap.add_argument("--subfolder", help="optional checkpoint subfolder, e.g. multilingual")
    args = ap.parse_args()
    backend = create_backend(
        args.backend, model=args.model, device=args.device, subfolder=args.subfolder
    )
    if not backend.parallel_safe and args.j != 1:
        print(f"local backend is single-process; reducing -j {args.j} to -j 1")
        args.j = 1
    all_jobs = list(jobs(args.n, args.rep_start))
    print(f"{len(all_jobs)} calls -> {args.out}  ({datetime.datetime.now(datetime.UTC).isoformat()})")
    lock, done = threading.Lock(), 0
    with open(args.out, "w") as f, ThreadPoolExecutor(args.j) as pool:
        for row in pool.map(lambda j: run_one(backend, j), all_jobs):
            with lock:
                f.write(json.dumps(row) + "\n")
                done += 1
                if done % 300 == 0:
                    print(f"  {done}/{len(all_jobs)}")
    print("done")


if __name__ == "__main__":
    main()

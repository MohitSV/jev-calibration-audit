"""Experiment 01: evidence dose-response sweep.

    venv/bin/python -m jevcal.sweep [-n REPEATS] [-j PARALLEL] [--out PATH]

Writes one JSON line per API call to experiments/data/01-evidence-sweep.jsonl.
"""

import argparse
import datetime
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from jevcal import client
from jevcal.backends import create_backend
from jevcal.evidence import (
    CHOICE_INSTRUCTIONS,
    LABEL_SETS,
    NOUL_HEADS,
    NOUL_TAILS,
    conditions,
)

DEFAULT_OUT = client.ROOT / "experiments" / "data" / "01-evidence-sweep.jsonl"


def jobs(repeats: int, rep_start: int = 1):
    for cond in conditions():
        for rep in range(rep_start, rep_start + repeats):
            for name, (criteria, _) in LABEL_SETS.items():
                yield ("choice", name, cond, rep, criteria)
            yield ("noul", "noul", cond, rep, None)


def run_one(backend, job):
    kind, label_set, cond, rep, criteria = job
    state = cond.state(rep)
    if kind == "choice":
        qs = {"q": {"type": "choice", "instructions": CHOICE_INSTRUCTIONS, "criteria": criteria}}
    else:
        qs = {
            "heads": {"type": "noul", "instructions": NOUL_HEADS},
            "tails": {"type": "noul", "instructions": NOUL_TAILS},
        }
    resp = backend.ask(state, qs)
    row = {
        "kind": kind, "label_set": label_set, "cid": cond.cid, "rep": rep,
        "p_true_heads": cond.p_heads, "state": state, "model": resp.get("model"),
        "input_tokens": resp.get("usage", {}).get("input_tokens"),
    }
    if kind == "choice":
        a = resp["answers"]["q"]
        row |= {"order": list(criteria), "choice": a["choice"],
                "probabilities": a["probabilities"], "confidence": a.get("confidence")}
    else:
        row |= {"noul_heads": resp["answers"]["heads"]["noul"],
                "noul_tails": resp["answers"]["tails"]["noul"]}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10, help="repeats per cell")
    ap.add_argument("-j", type=int, default=8, help="parallel requests")
    ap.add_argument("--rep-start", type=int, default=1, help="first record number")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
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
                if done % 200 == 0:
                    print(f"  {done}/{len(all_jobs)}")
    print("done")


if __name__ == "__main__":
    main()

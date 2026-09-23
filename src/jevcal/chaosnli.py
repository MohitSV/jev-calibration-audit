"""Experiment 06: Jev against human vote distributions on ChaosNLI (SNLI + MNLI-m).

    cd src && ../venv/bin/python -m jevcal.chaosnli run      # 900 calls
    cd src && ../venv/bin/python -m jevcal.chaosnli analyze

The design is pre-registered in experiments/06-chaosnli.md. The sample is fixed by
SEED and BINS; changing either invalidates the pre-registration.
"""

import argparse
import datetime
import json
import math
import pathlib
import random
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

from jevcal.backends import create_backend
from jevcal.client import ROOT

DATA_DIR = ROOT / "data" / "external" / "chaosNLI_v1.0"
FILES = ("chaosNLI_snli.jsonl", "chaosNLI_mnli_m.jsonl")
OUT = ROOT / "experiments" / "data" / "06-chaosnli.jsonl"
SEED = 20260922
PER_BIN = 60
BINS = ((0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.85), (0.85, 1.01))
LABELS = ("entailment", "neutral", "contradiction")  # ChaosNLI label_dist order
EPS = 1e-6

CRITERIA = {
    "entailment": "The hypothesis is definitely true given the premise",
    "neutral": "The hypothesis might be true or might be false given the premise",
    "contradiction": "The hypothesis is definitely false given the premise",
}
CHOICE_INSTRUCTIONS = "Given the premise, is the hypothesis entailed, neutral, or contradicted?"
NOUL = {
    "entailment": "Given the premise, is the hypothesis definitely true?",
    "neutral": "Given the premise, is it undetermined whether the hypothesis is true?",
    "contradiction": "Given the premise, is the hypothesis definitely false?",
}
ORDERS = {"enc": LABELS, "cne": tuple(reversed(LABELS))}


def load_items(data_dir=DATA_DIR):
    items = []
    for name in FILES:
        for line in (data_dir / name).read_text().splitlines():
            r = json.loads(line)
            items.append({
                "uid": r["uid"], "source": name.split("_", 1)[1].removesuffix(".jsonl"),
                "premise": r["example"]["premise"], "hypothesis": r["example"]["hypothesis"],
                "human": dict(zip(LABELS, r["label_dist"], strict=True)),
            })
    return items


def bin_of(max_share):
    return next(i for i, (lo, hi) in enumerate(BINS) if lo <= max_share < hi)


def sample(items, per_bin=PER_BIN, seed=SEED):
    rng = random.Random(seed)
    by_bin = {i: [] for i in range(len(BINS))}
    for it in sorted(items, key=lambda x: x["uid"]):
        by_bin[bin_of(max(it["human"].values()))].append(it)
    out = []
    for i, pool in by_bin.items():
        if len(pool) < per_bin:
            raise ValueError(f"bin {BINS[i]} has only {len(pool)} items")
        out += [it | {"bin": i} for it in rng.sample(pool, per_bin)]
    return out


def state(it):
    return f"Premise: {it['premise']}\nHypothesis: {it['hypothesis']}"


def jobs(items):
    for it in items:
        for order_name, order in ORDERS.items():
            yield ("choice", it, order_name, {k: CRITERIA[k] for k in order})
        yield ("noul", it, "noul", None)


def run_one(backend, job):
    kind, it, variant, criteria = job
    if kind == "choice":
        qs = {"q": {"type": "choice", "instructions": CHOICE_INSTRUCTIONS, "criteria": criteria}}
    else:
        qs = {k: {"type": "noul", "instructions": q} for k, q in NOUL.items()}
    resp = backend.ask(state(it), qs)
    row = {"kind": kind, "variant": variant, "uid": it["uid"], "source": it["source"],
           "bin": it["bin"], "human": it["human"], "model": resp.get("model"),
           "input_tokens": resp.get("usage", {}).get("input_tokens")}
    if kind == "choice":
        a = resp["answers"]["q"]
        row |= {"choice": a["choice"], "probs": {k: a["probabilities"].get(k, 0.0) for k in LABELS}}
    else:
        row |= {"noul_raw": {k: resp["answers"][k]["noul"] for k in LABELS}}
    return row


# ---------- analysis ----------

def normalize(d):
    total = sum(d.values())
    return {k: v / total for k, v in d.items()} if total > EPS else {k: 1 / 3 for k in d}


def kl(p, q):
    """KL(p || q): human p, model q."""
    return sum(p[k] * math.log(p[k] / max(q[k], EPS)) for k in LABELS if p[k] > 0)


def tvd(p, q):
    return 0.5 * sum(abs(p[k] - q[k]) for k in LABELS)


def brier(p, q):
    return sum((p[k] - q[k]) ** 2 for k in LABELS)


def entropy(p):
    return -sum(v * math.log(v) for v in p.values() if v > 0)


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for m in range(i, j + 1):
                r[order[m]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def predictions(rows):
    """{uid: {"human":…, "bin":…, method: dist}} for the four pre-registered methods."""
    items = {}
    for r in rows:
        it = items.setdefault(r["uid"], {"human": r["human"], "bin": r["bin"], "source": r["source"]})
        if r["kind"] == "choice":
            it[f"choice_{r['variant']}"] = normalize(r["probs"])
        else:
            it["noul_norm"] = normalize(r["noul_raw"])
            it["noul_sum"] = sum(r["noul_raw"].values())
    for it in items.values():
        it["choice_avg"] = {k: (it["choice_enc"][k] + it["choice_cne"][k]) / 2 for k in LABELS}
    return items


METHODS = ("choice_enc", "choice_cne", "choice_avg", "noul_norm")


def summarize(rows, n_boot=2000, seed=0):
    items = predictions(rows)
    uids = sorted(items)
    per = {m: {u: {"kl": kl(items[u]["human"], items[u][m]),
                   "tvd": tvd(items[u]["human"], items[u][m]),
                   "brier": brier(items[u]["human"], items[u][m]),
                   "correct": max(items[u][m], key=items[u][m].get)
                   == max(items[u]["human"], key=items[u]["human"].get)}
               for u in uids} for m in METHODS}
    out = {"n_items": len(uids), "n_calls": len(rows),
           "models": sorted({r["model"] for r in rows}),
           "input_tokens": sum(r["input_tokens"] or 0 for r in rows), "methods": {}}
    human_h = [entropy(items[u]["human"]) for u in uids]
    for m in METHODS:
        out["methods"][m] = {
            "kl": round(statistics.mean(v["kl"] for v in per[m].values()), 4),
            "tvd": round(statistics.mean(v["tvd"] for v in per[m].values()), 4),
            "brier": round(statistics.mean(v["brier"] for v in per[m].values()), 4),
            "acc_vs_majority": round(statistics.mean(v["correct"] for v in per[m].values()), 4),
            "spearman_entropy": round(spearman([entropy(items[u][m]) for u in uids], human_h), 4),
            "max_prob_by_bin": [round(statistics.mean(max(items[u][m].values()) for u in uids
                                                       if items[u]["bin"] == b), 3)
                                for b in range(len(BINS))],
        }
    out["human_max_share_by_bin"] = [round(statistics.mean(max(items[u]["human"].values())
                                                           for u in uids if items[u]["bin"] == b), 3)
                                     for b in range(len(BINS))]
    rng = random.Random(seed)
    diffs = {"kl_noul_minus_choice_avg": [], "spearman_noul_minus_choice_avg": []}
    for _ in range(n_boot):
        idx = rng.choices(uids, k=len(uids))
        diffs["kl_noul_minus_choice_avg"].append(
            statistics.mean(per["noul_norm"][u]["kl"] - per["choice_avg"][u]["kl"] for u in idx))
        hh = [entropy(items[u]["human"]) for u in idx]
        diffs["spearman_noul_minus_choice_avg"].append(
            spearman([entropy(items[u]["noul_norm"]) for u in idx], hh)
            - spearman([entropy(items[u]["choice_avg"]) for u in idx], hh))
    out["bootstrap_ci95"] = {k: [round(sorted(v)[int(0.025 * n_boot)], 4),
                                 round(sorted(v)[int(0.975 * n_boot) - 1], 4)]
                             for k, v in diffs.items()}
    out["noul_raw_sum_mean"] = round(statistics.mean(items[u]["noul_sum"] for u in uids), 3)
    return out


def figure(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs = summary["human_max_share_by_bin"]
    ax.plot(xs, xs, "k--", lw=1, label="human majority share")
    for m in ("choice_avg", "noul_norm"):
        ax.plot(xs, summary["methods"][m]["max_prob_by_bin"], marker="o", label=m)
    ax.set(xlabel="Human majority share (bin mean)", ylabel="Model top probability",
           title="ChaosNLI: does confidence track human agreement?", ylim=(0.3, 1.02))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "analyze"))
    ap.add_argument("-j", type=int, default=8)
    ap.add_argument("--backend", choices=("jev", "laya", "qwen-rlcd", "eve-rlcd"), default="jev")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--summary", default=str(OUT.with_suffix(".summary.json")))
    ap.add_argument("--figure", default=str(ROOT / "experiments/figures/06-chaosnli.png"))
    args = ap.parse_args()
    if args.cmd == "run":
        items = sample(load_items())
        backend = create_backend(args.backend, device=args.device)
        j = args.j if backend.parallel_safe else 1
        all_jobs = list(jobs(items))
        print(f"{len(all_jobs)} calls -> {args.out} ({datetime.datetime.now(datetime.UTC).isoformat()})")
        lock, done = threading.Lock(), 0
        with open(args.out, "w") as f, ThreadPoolExecutor(j) as pool:
            for row in pool.map(lambda jb: run_one(backend, jb), all_jobs):
                with lock:
                    f.write(json.dumps(row) + "\n")
                    done += 1
                    if done % 300 == 0:
                        print(f"  {done}/{len(all_jobs)}")
        print("done")
    else:
        rows = [json.loads(x) for x in pathlib.Path(args.out).read_text().splitlines()]
        s = summarize(rows)
        pathlib.Path(args.summary).write_text(json.dumps(s, indent=2))
        figure(s, pathlib.Path(args.figure))
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()

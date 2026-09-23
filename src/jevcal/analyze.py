"""Analyse experiment 01.

    venv/bin/python -m jevcal.analyze [PATH.jsonl]

Pull decomposition. Averaging a condition with its mirror (every report flipped) cancels
the truth, because P_true(heads | c) + P_true(heads | mirror c) = 1. What survives is the
model's asymmetry:
    heads_pull(set, strength) = (P(heads | c) + P(heads | mirror c)) / 2 - 0.5
With option_1 = heads, the pull is semantic + label; with option_1 = tails it is
semantic - label. So
    label    = (pull[opt1_is_heads] - pull[opt1_is_tails]) / 2   # "option_1" name effect
    semantic = (pull[opt1_is_heads] + pull[opt1_is_tails]) / 2   # the word "heads"
    position = (pull[heads_tails]   - pull[tails_heads])   / 2   # listed first
"""

import argparse
import collections
import json
import math
import pathlib
import statistics

from jevcal.client import ROOT
from jevcal.evidence import HEADS_DESC, LABEL_SETS, conditions, mirror

DATA = ROOT / "experiments" / "data"


def p_heads(row) -> float:
    if row["kind"] == "noul":
        return row["noul_heads"]
    criteria = LABEL_SETS[row["label_set"]][0]
    key = next(k for k, v in criteria.items() if v == HEADS_DESC)
    return row["probabilities"].get(key, 0.0)


def cell_means(rows):
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["label_set"], r["cid"])].append(p_heads(r))
    return {k: (statistics.mean(v), statistics.pstdev(v), len(v)) for k, v in cells.items()}


def strength_pairs():
    """One representative per mirror pair, keyed by evidence strength |P_true - 0.5|."""
    seen, pairs = set(), []
    for c in conditions():
        if c.cid in seen:
            continue
        seen |= {c.cid, mirror(c.cid)}
        pairs.append((c.cid, mirror(c.cid), abs(c.p_heads - 0.5)))
    return sorted(pairs, key=lambda p: (p[2], p[0]))


def heads_pull(means, label_set, a, b):
    return (means[(label_set, a)][0] + means[(label_set, b)][0]) / 2 - 0.5


def logit(p):
    p = min(max(p, 0.005), 0.995)
    return math.log(p / (1 - p))


def truth_slope(means, label_set):
    """OLS slope of logit(P_jev) on logit(P_true), informative conditions only.

    1 = calibrated, <1 = too timid, >1 = overconfident."""
    xs, ys = [], []
    for c in conditions():
        if abs(c.p_heads - 0.5) > 1e-9 and (label_set, c.cid) in means:
            xs.append(logit(c.p_heads))
            ys.append(logit(means[(label_set, c.cid)][0]))
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx


def summarize(rows):
    means = cell_means(rows)
    sets = [*LABEL_SETS, "noul"]
    table = []
    for a, b, strength in strength_pairs():
        pulls = {s: heads_pull(means, s, a, b) for s in sets if (s, a) in means}
        o1h, o1t = pulls.get("opt1_is_heads"), pulls.get("opt1_is_tails")
        ht, th = pulls.get("heads_tails"), pulls.get("tails_heads")
        table.append({
            "pair": f"{a}|{b}" if a != b else a,
            "strength": round(strength, 3),
            "pulls": {s: round(v, 3) for s, v in pulls.items()},
            "label_effect": round((o1h - o1t) / 2, 3),
            "semantic_effect": round((o1h + o1t) / 2, 3),
            "position_effect": round((ht - th) / 2, 3),
        })
    slopes = {s: round(truth_slope(means, s), 3) for s in sets}
    mae = {
        s: round(statistics.mean(abs(means[(s, c.cid)][0] - c.p_heads) for c in conditions()), 3)
        for s in sets
    }
    noul_rows = [r for r in rows if r["kind"] == "noul"]
    complement_gap = statistics.mean(abs(r["noul_heads"] + r["noul_tails"] - 1) for r in noul_rows)
    within_cell_sd = statistics.mean(sd for _, sd, _ in means.values())
    return {
        "n_calls": len(rows),
        "models": sorted({r["model"] for r in rows}),
        "input_tokens": sum(r["input_tokens"] or 0 for r in rows),
        "decomposition_by_strength": table,
        "truth_slope_logit": slopes,
        "mean_abs_error_vs_bayes": mae,
        "noul_complement_gap": round(complement_gap, 3),
        "mean_within_cell_sd": round(within_cell_sd, 4),
        "cells": {f"{k[0]}::{k[1]}": [round(m, 4), round(sd, 4), n]
                  for k, (m, sd, n) in sorted(means.items())},
    }


def figure(rows, summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = cell_means(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    conds = sorted(conditions(), key=lambda c: c.p_heads)
    for s in [*LABEL_SETS, "noul"]:
        xs = [c.p_heads for c in conds]
        ys = [means[(s, c.cid)][0] for c in conds]
        ax1.plot(xs, ys, marker="o", ms=3, lw=1, label=s)
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Bayes (perfect)")
    ax1.set(xlabel="True P(heads)", ylabel="Jev P(heads)", title="Truth tracking")
    ax1.legend(fontsize=7)

    t = summary["decomposition_by_strength"]
    xs = range(len(t))
    for k in ("semantic_effect", "label_effect", "position_effect"):
        ax2.plot(xs, [r[k] for r in t], marker="o", label=k)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(list(xs), [f"{r['pair'].split('|')[0]}\n({r['strength']})" for r in t],
                   rotation=60, fontsize=6)
    ax2.set(ylabel="Pull toward heads / option_1 / first-listed", title="Prior vs evidence strength")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(DATA / "01-evidence-sweep.jsonl"))
    ap.add_argument("--summary", default=str(DATA / "01-evidence-sweep.summary.json"))
    ap.add_argument("--figure", default=str(ROOT / "experiments/figures/01-evidence-sweep.png"))
    args = ap.parse_args()
    rows = [json.loads(line) for line in pathlib.Path(args.path).read_text().splitlines()]
    summary = summarize(rows)
    pathlib.Path(args.summary).write_text(json.dumps(summary, indent=2))
    figure(rows, summary, pathlib.Path(args.figure))
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=2))


if __name__ == "__main__":
    main()

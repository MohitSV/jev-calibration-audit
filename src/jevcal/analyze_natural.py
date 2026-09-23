"""Analyse experiment 03.

    cd src && ../venv/bin/python -m jevcal.analyze_natural
"""

import argparse
import collections
import json
import pathlib
import statistics

from jevcal.client import ROOT
from jevcal.natural import BASE_RATE_KS, CUES, DOMAINS

DATA = ROOT / "experiments" / "data"
CUE_LEVELS = [
    "neutral:L0",
    *[f"{t}:L{i}" for t in CUES for i in range(1, len(CUES[t]))],
    "mixed:BT",
    "mixed:TB",
]


def cell_means(rows):
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["domain"], r["level"], r["variant"])].append(r["p_yes"])
    return {k: statistics.mean(v) for k, v in cells.items()}


def choice_mean(means, domain, level):
    vals = [v for (d, lv, var), v in means.items() if d == domain and lv == level and var != "noul"]
    return statistics.mean(vals)


def summarize(rows):
    means = cell_means(rows)
    base = {}
    for d in DOMAINS:
        pts = [(k / 20, choice_mean(means, d, f"k={k}"), means[(d, f"k={k}", "noul")])
               for k in BASE_RATE_KS]
        base[d] = {
            "curve": [{"p_true": t, "choice": round(c, 3), "noul": round(n, 3)} for t, c, n in pts],
            "mae_choice": round(statistics.mean(abs(c - t) for t, c, _ in pts), 3),
            "mae_noul": round(statistics.mean(abs(n - t) for t, _, n in pts), 3),
            # jump across the tie: what a 45%->55% base-rate change does
            "choice_jump_45_to_55": round(choice_mean(means, d, "k=11") - choice_mean(means, d, "k=9"), 3),
            "choice_at_tie": round(choice_mean(means, d, "k=10"), 3),
            "order_effect_at_tie": round(
                means[(d, "k=10", _variants(d)[0])] - means[(d, "k=10", _variants(d)[1])], 3),
        }
    cues = {lv: {"choice_p_billing": round(choice_mean(means, "ticket", lv), 3),
                 "noul_p_billing": round(means[("ticket", lv, "noul")], 3)} for lv in CUE_LEVELS}
    noul_rows = [r for r in rows if r["kind"] == "noul"]
    return {
        "n_calls": len(rows),
        "models": sorted({r["model"] for r in rows}),
        "input_tokens": sum(r["input_tokens"] or 0 for r in rows),
        "base_rate": base,
        "cues": cues,
        "noul_complement_gap": round(
            statistics.mean(abs(r["noul_yes"] + r["noul_no"] - 1) for r in noul_rows), 3),
    }


def _variants(domain):
    return ("billing_first", "technical_first") if domain == "ticket" else ("yes_first", "no_first")


def figure(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for d, s in summary["base_rate"].items():
        xs = [p["p_true"] for p in s["curve"]]
        ax1.plot(xs, [p["choice"] for p in s["curve"]], marker="o", ms=3, label=f"{d} Choice")
        ax1.plot(xs, [p["noul"] for p in s["curve"]], marker="s", ms=3, ls=":", label=f"{d} Noul")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="exact (k/20)")
    ax1.set(xlabel="Base rate in the text (k/20)", ylabel="Jev P(first-named outcome)",
            title="Arm A: stated base rates")
    ax1.legend(fontsize=7)

    order = ["technical:L5", "technical:L4", "technical:L3", "technical:L2", "technical:L1",
             "neutral:L0", "billing:L1", "billing:L2", "billing:L3", "billing:L4", "billing:L5"]
    xs = range(len(order))
    c = summary["cues"]
    ax2.plot(xs, [c[lv]["choice_p_billing"] for lv in order], marker="o", label="Choice")
    ax2.plot(xs, [c[lv]["noul_p_billing"] for lv in order], marker="s", ls=":", label="Noul")
    ax2.set_xticks(list(xs), [lv.replace(":", "\n") for lv in order], fontsize=7)
    ax2.set(ylabel="Jev P(billing)", title="Arm B: graded ticket cues (tech 5 ... billing 5)")
    ax2.axhline(0.5, color="k", lw=0.6)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(DATA / "03-natural-evidence.jsonl"))
    ap.add_argument("--summary", default=str(DATA / "03-natural-evidence.summary.json"))
    ap.add_argument("--figure", default=str(ROOT / "experiments/figures/03-natural-evidence.png"))
    args = ap.parse_args()
    rows = [json.loads(line) for line in pathlib.Path(args.path).read_text().splitlines()]
    summary = summarize(rows)
    pathlib.Path(args.summary).write_text(json.dumps(summary, indent=2))
    figure(summary, pathlib.Path(args.figure))
    print(json.dumps({k: v for k, v in summary.items() if k != "cues"}
                     | {"cues": summary["cues"]}, indent=1)[:4000])


if __name__ == "__main__":
    main()

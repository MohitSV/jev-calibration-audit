"""Experiment 05: fit probability repairs on old data and score held-out runs."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics
from dataclasses import dataclass

from jevcal.client import ROOT
from jevcal.evidence import HEADS_DESC, LABEL_SETS

EPS = 1e-6


@dataclass(frozen=True)
class Observation:
    dataset: str
    group: str
    rep: int
    target: float
    choice: float
    noul_yes: float
    noul_no: float
    is_baseline: bool


def clip(p: float) -> float:
    return min(max(float(p), EPS), 1 - EPS)


def logit(p: float) -> float:
    p = clip(p)
    return math.log(p / (1 - p))


def logistic(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1 + ez)


def load_evidence(path: pathlib.Path, dataset: str = "witness") -> list[Observation]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    choice = {}
    noul = {}
    for row in rows:
        key = (row["cid"], row["rep"], row["label_set"])
        if row["kind"] == "choice":
            criteria = LABEL_SETS[row["label_set"]][0]
            heads_key = next(k for k, v in criteria.items() if v == HEADS_DESC)
            choice[key] = row["probabilities"].get(heads_key, 0.0)
        else:
            noul[(row["cid"], row["rep"])] = (row["noul_heads"], row["noul_tails"])
    out = []
    for (cid, rep, label_set), p_choice in choice.items():
        source = next(r for r in rows if r["cid"] == cid and r["rep"] == rep)
        py, pn = noul[(cid, rep)]
        out.append(Observation(dataset, label_set, rep, source["p_true_heads"], p_choice,
                               py, pn, cid == "none"))
    return out


def load_natural(path: pathlib.Path, dataset: str = "base_rate") -> list[Observation]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [r for r in rows if r["arm"] == "base_rate"]
    noul = {(r["domain"], r["level"], r["rep"]): (r["noul_yes"], r["noul_no"])
            for r in rows if r["kind"] == "noul"}
    out = []
    for row in rows:
        if row["kind"] != "choice":
            continue
        py, pn = noul[(row["domain"], row["level"], row["rep"])]
        out.append(Observation(
            dataset,
            f"{row['domain']}::{row['variant']}",
            row["rep"],
            row["p_true"],
            row["p_yes"],
            py,
            pn,
            row["level"] == "k=10",
        ))
    return out


def normalized_noul(o: Observation) -> float:
    total = o.noul_yes + o.noul_no
    return o.noul_yes / total if total > EPS else 0.5


def soft_log_loss(predictions: list[float], targets: list[float]) -> float:
    return statistics.mean(
        -(y * math.log(clip(p)) + (1 - y) * math.log(clip(1 - p)))
        for p, y in zip(predictions, targets, strict=True)
    )


def fit_slope(probabilities: list[float], targets: list[float]) -> float:
    """Fit a no-intercept logit multiplier by bounded golden-section search."""
    logits = [logit(p) for p in probabilities]

    def objective(log_alpha: float) -> float:
        alpha = math.exp(log_alpha)
        return soft_log_loss([logistic(alpha * z) for z in logits], targets)

    lo, hi = math.log(0.01), math.log(100.0)
    ratio = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    f1, f2 = objective(x1), objective(x2)
    for _ in range(100):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - ratio * (hi - lo)
            f1 = objective(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + ratio * (hi - lo)
            f2 = objective(x2)
    return math.exp((lo + hi) / 2)


def baseline_logits(train: list[Observation]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for o in train:
        if o.is_baseline:
            grouped.setdefault(o.group, []).append(logit(o.choice))
    return {group: statistics.mean(values) for group, values in grouped.items()}


def smooth_ece(predictions: list[float], targets: list[float], sigma: float = 0.075) -> float:
    """Gaussian soft-bin ECE; each observation's weights sum to one over 21 bins."""
    centers = [i / 20 for i in range(21)]
    weights = []
    for p in predictions:
        row = [math.exp(-0.5 * ((p - c) / sigma) ** 2) for c in centers]
        total = sum(row)
        weights.append([w / total for w in row])
    error = 0.0
    n = len(predictions)
    for j in range(len(centers)):
        mass = sum(row[j] for row in weights)
        if mass <= EPS:
            continue
        mean_p = sum(row[j] * p for row, p in zip(weights, predictions, strict=True)) / mass
        mean_y = sum(row[j] * y for row, y in zip(weights, targets, strict=True)) / mass
        error += mass / n * abs(mean_p - mean_y)
    return error


def target_entropy(targets: list[float]) -> float:
    """The unavoidable part of soft-target log loss; a perfect forecaster scores exactly this."""
    return statistics.mean(-(y * math.log(clip(y)) + (1 - y) * math.log(clip(1 - y)))
                           for y in targets)


def brier(predictions: list[float], targets: list[float]) -> float:
    return statistics.mean((p - y) ** 2 for p, y in zip(predictions, targets, strict=True))


def metrics(predictions: list[float], targets: list[float]) -> dict[str, float]:
    log_loss = soft_log_loss(predictions, targets)
    return {
        "brier": round(brier(predictions, targets), 4),
        "log_loss": round(log_loss, 4),
        # log loss minus its floor (= mean KL divergence from the exact target); 0 is perfect
        "excess_log_loss": round(log_loss - target_entropy(targets), 4),
        "smooth_ece": round(smooth_ece(predictions, targets), 4),
    }


def cluster_key(o: Observation) -> tuple:
    """Observations that share one underlying API response must be resampled together.

    In experiment 01 the six Choice label sets share a single Noul call per condition and
    repeat, so the effective sample for Noul methods is ~6x smaller than the row count."""
    return (o.dataset, round(o.target, 6), o.rep)


def bootstrap_brier(subset, preds, methods, n_boot=2000, seed=0):
    """Cluster bootstrap 95% CIs for each method's Brier, and for paired differences."""
    rng = random.Random(seed)
    clusters: dict[tuple, list[int]] = {}
    for i, o in enumerate(subset):
        clusters.setdefault(cluster_key(o), []).append(i)
    keys = list(clusters)
    ys = [o.target for o in subset]
    sq = {m: [(preds[i][m] - ys[i]) ** 2 for i in range(len(subset))] for m in methods}
    pairs = [("normalized_noul", "raw_choice"), ("normalized_noul", "raw_noul"),
             ("normalized_noul", "temperature_choice")]
    draws = {m: [] for m in methods} | {f"{a}-{b}": [] for a, b in pairs}
    for _ in range(n_boot):
        idx = [i for k in rng.choices(keys, k=len(keys)) for i in clusters[k]]
        means = {m: statistics.mean(sq[m][i] for i in idx) for m in methods}
        for m in methods:
            draws[m].append(means[m])
        for a, b in pairs:
            draws[f"{a}-{b}"].append(means[a] - means[b])

    def ci(v):
        v = sorted(v)
        return [round(v[int(0.025 * len(v))], 5), round(v[int(0.975 * len(v)) - 1], 5)]

    return {"n_clusters": len(keys), "brier_ci95": {k: ci(v) for k, v in draws.items()}}


def evaluate(train_witness: list[Observation], train_natural: list[Observation],
             heldout: list[Observation]) -> dict:
    # The preregistered slopes use experiment 01 only. Natural baselines are used
    # solely by content-free prior division, whose group intercepts are domain-specific.
    targets = [o.target for o in train_witness]
    noul_slope = fit_slope([o.noul_yes for o in train_witness], targets)
    choice_slope = fit_slope([o.choice for o in train_witness], targets)
    priors = baseline_logits([*train_witness, *train_natural])

    def predictions(o: Observation) -> dict[str, float]:
        return {
            "raw_choice": o.choice,
            "raw_noul": o.noul_yes,
            "normalized_noul": normalized_noul(o),
            "platt_noul": logistic(noul_slope * logit(o.noul_yes)),
            "temperature_choice": logistic(choice_slope * logit(o.choice)),
            "prior_divided_choice": logistic(logit(o.choice) - priors[o.group]),
        }

    result = {"fit": {
        "noul_logit_multiplier": round(noul_slope, 6),
        "choice_logit_multiplier": round(choice_slope, 6),
        "choice_temperature": round(1 / choice_slope, 6),
        "content_free_prior_logits": {k: round(v, 6) for k, v in sorted(priors.items())},
    }, "heldout": {}}
    for dataset in sorted({o.dataset for o in heldout} | {"overall"}):
        subset = heldout if dataset == "overall" else [o for o in heldout if o.dataset == dataset]
        ys = [o.target for o in subset]
        result["heldout"][dataset] = {}
        preds = [predictions(o) for o in subset]
        methods = list(preds[0])
        for method in methods:
            result["heldout"][dataset][method] = metrics([p[method] for p in preds], ys)
        result["heldout"][dataset]["n_observations"] = len(subset)
        result["heldout"][dataset]["bootstrap"] = bootstrap_brier(subset, preds, methods)
    return result


def figure(result: dict, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [k for k in result["heldout"]["overall"] if k not in ("n_observations", "bootstrap")]
    labels = [m.replace("_", "\n") for m in methods]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric, title in zip(
        axes,
        ("brier", "excess_log_loss", "smooth_ece"),
        ("Brier score", "Excess log loss (KL from target)", "Smooth ECE"),
        strict=True,
    ):
        witness = [result["heldout"]["witness"][m][metric] for m in methods]
        base = [result["heldout"]["base_rate"][m][metric] for m in methods]
        x = list(range(len(methods)))
        ax.bar([v - 0.2 for v in x], witness, width=0.4, label="witness")
        ax.bar([v + 0.2 for v in x], base, width=0.4, label="base rate")
        ax.set_title(title)
        ax.set_xticks(x, labels, fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Held-out Jev probability repairs ({result.get('split', '')})")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    data = ROOT / "experiments/data"
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-witness", default=str(data / "01-evidence-sweep.jsonl"))
    ap.add_argument("--train-natural", default=str(data / "03-natural-evidence.jsonl"))
    ap.add_argument("--heldout-witness")
    ap.add_argument("--heldout-natural")
    ap.add_argument("--split-at", type=int, default=5,
                    help="without held-out paths, fit reps <= N and score reps > N")
    ap.add_argument("--out", default=str(data / "05-calibration-fixes.summary.json"))
    ap.add_argument("--figure", default=str(ROOT / "experiments/figures/05-calibration-fixes.png"))
    args = ap.parse_args()
    witness = load_evidence(pathlib.Path(args.train_witness))
    natural = load_natural(pathlib.Path(args.train_natural))
    if bool(args.heldout_witness) != bool(args.heldout_natural):
        ap.error("provide both held-out paths or neither")
    if args.heldout_witness:
        train_witness, train_natural = witness, natural
        heldout = [*load_evidence(pathlib.Path(args.heldout_witness)),
                   *load_natural(pathlib.Path(args.heldout_natural))]
        split = "separate held-out files"
    else:
        train_witness = [o for o in witness if o.rep <= args.split_at]
        train_natural = [o for o in natural if o.rep <= args.split_at]
        heldout = [o for o in [*witness, *natural] if o.rep > args.split_at]
        split = f"existing repeats: fit <= {args.split_at}, score > {args.split_at}"
    result = evaluate(train_witness, train_natural, heldout)
    result["split"] = split
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
    figure(result, pathlib.Path(args.figure))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

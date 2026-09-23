"""Experiment 11 analysis: Gate 1 (reproduction) and the pre-registered L2 decision.

    cd src && ../venv/bin/python -m jevcal.eve_pilot gate      # accuracy/ECE on test sample
    cd src && ../venv/bin/python -m jevcal.eve_pilot probes    # 01/03 probes, n=2
    cd src && ../venv/bin/python -m jevcal.eve_pilot decide    # applies the decision rule
"""

import argparse
import json
import subprocess
import sys

from jevcal.backends import EVE_RLCD_REPO
from jevcal.client import ROOT

WARMUP = ROOT / "data" / "external" / "runs" / "q-warmup"
DATA = ROOT / "experiments" / "data"
RELEASED_L2 = (0.937 + (1 - 0.033)) / 2  # experiment 10 ticket ladder, pre-registered 0.952


def gate():
    sys.path.insert(0, str(EVE_RLCD_REPO))
    from rlcd.policies import load_policy
    from rlcd.quick_eval import evaluate, stride_sample
    from rlcd.schema import read_jsonl

    qs = stride_sample(read_jsonl(str(ROOT / "data/external/eve-data/test.jsonl")), 2000)
    policy = load_policy(str(WARMUP), device="mps", backend="hf-decoder")
    res = evaluate(policy, qs, 512, device="mps")
    res = {k: v for k, v in res.items() if not isinstance(v, dict)} | {"n": len(qs)}
    res["gate_pass"] = abs(res["eval_acc"] - 0.748) <= 0.03 and res["eval_ece"] <= 0.05
    (DATA / "11-warmup-gate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def probes():
    py = sys.executable
    for mod, name in (("jevcal.sweep", "evidence-sweep"), ("jevcal.sweep_natural", "natural-evidence")):
        out = DATA / f"11-warmup-{name}.jsonl"
        subprocess.run([py, "-m", mod, "-n", "2", "--backend", "eve-rlcd", "--device", "mps",
                        "--model", str(WARMUP), "--out", str(out)], check=True)
    for mod, name in (("jevcal.analyze", "evidence-sweep"), ("jevcal.analyze_natural", "natural-evidence")):
        stem = DATA / f"11-warmup-{name}"
        subprocess.run([py, "-m", mod, f"{stem}.jsonl", "--summary", f"{stem}.summary.json",
                        "--figure", str(ROOT / f"experiments/figures/11-warmup-{name}.png")],
                       check=True, stdout=subprocess.DEVNULL)


def decide():
    gate_res = json.loads((DATA / "11-warmup-gate.json").read_text())
    nat = json.loads((DATA / "11-warmup-natural-evidence.summary.json").read_text())
    ev = json.loads((DATA / "11-warmup-evidence-sweep.summary.json").read_text())
    rel = json.loads((DATA / "10-eve-natural-evidence.summary.json").read_text())
    l2 = (nat["cues"]["billing:L2"]["choice_p_billing"]
          + 1 - nat["cues"]["technical:L2"]["choice_p_billing"]) / 2
    if not gate_res["gate_pass"]:
        verdict = "GATE 1 FAILED: rebuild not faithful; do not interpret"
    elif l2 >= 0.90:
        verdict = "saturation already present before RL -> do not rent a GPU for this question"
    elif l2 <= 0.75:
        verdict = "RL stage associated with saturation -> rent a GPU for the controlled run"
    else:
        verdict = "ambiguous -> run the 100-step local RLCD pilot"
    out = {
        "gate": gate_res, "warmup_L2_saturation": round(l2, 3),
        "released_rlcd_L2_saturation": round(RELEASED_L2, 3), "verdict": verdict,
        "secondary": {
            "warmup": {d: {k: v[k] for k in ("choice_jump_45_to_55", "mae_choice", "mae_noul")}
                       for d, v in nat["base_rate"].items()},
            "released": {d: {k: v[k] for k in ("choice_jump_45_to_55", "mae_choice", "mae_noul")}
                         for d, v in rel["base_rate"].items()},
            "warmup_witness_slope": ev["truth_slope_logit"],
            "warmup_ticket_choice": {k: v["choice_p_billing"] for k, v in nat["cues"].items()},
        },
    }
    (DATA / "11-warmup-decision.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("gate", "probes", "decide"))
    {"gate": gate, "probes": probes, "decide": decide}[ap.parse_args().cmd]()

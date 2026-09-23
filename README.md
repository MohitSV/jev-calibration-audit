# jev-calibration-audit

An exact-target audit of the probabilities returned by "System One" decision models,
centred on TypeSafe's **Jev** (`jev-1.13.0`). Jev's training method, *Reinforcement
Learning for Calibrated Decisions* (RLCD), is advertised as producing "epistemically
honest" probabilities.

The audit scores each primitive against targets whose correct probability is known
exactly: a hidden fair coin with witnesses of stated reliability, and stated
reference-class base rates. It then checks the conclusions against human vote
distributions (ChaosNLI, DICES-350).

## Main findings

- **With no evidence, Jev's `Choice` primitive gives a fair coin 0.83–0.93 on "heads".**
  The pull follows the word "heads", not the option key or its position, and mostly
  vanishes after one weak witness.
- **On evidence that states a probability, `Choice` is a step function.** A 55/45 base
  rate yields 0.90–0.996, and temperature scaling can't recover it.
- **`Noul` (yes/no) tracks the same targets** to within 0.015–0.027. Asking one `Noul`
  per option and normalizing cuts held-out Brier error from 0.067 to 0.003.
- **On real human disagreement, normalized `Noul` is 4–6× closer to the vote
  distribution zero-shot.** `Choice` ranks disagreement as well, and matches `Noul` after
  one per-task temperature.
- **Three open checkpoints that call themselves RLCD models fail the same audit** in
  different ways. A rebuild of the most faithful one's supervised warm-up shows its cue
  saturation predates its RL stage.

`FRAMING.md` records the argument and how it changed. `experiments/NN-*.md` holds one note
per experiment, with its pre-registration, commands, numbers and caveats.

## Layout

| Path | Contents |
|---|---|
| `src/jevcal/client.py` | Minimal TypeSafe API client (stdlib only, with retries) |
| `src/jevcal/backends.py` | Jev, Laya, Qwen3-0.6B PSR-FT and eve-rlcd backends behind one interface |
| `src/jevcal/evidence.py`, `sweep.py`, `analyze.py` | Exp 01: hidden coin + witnesses, mirror-pair decomposition |
| `src/jevcal/natural.py`, `sweep_natural.py`, `analyze_natural.py` | Exp 03: stated base rates, graded ticket cues |
| `src/jevcal/bootstrap_0103.py` | Exp 02: repeat-level bootstrap CIs |
| `src/jevcal/calibrate.py` | Exp 05: repair methods, excess log loss, cluster-bootstrap CIs |
| `src/jevcal/chaosnli.py`, `chaosnli_temperature.py` | Exp 06: ChaosNLI |
| `src/jevcal/dices.py` | Exp 06b: DICES-350 |
| `src/jevcal/eve_pilot.py`, `eve_patch/` | Exp 11: rebuild of eve-rlcd's pre-RL warm-up on Apple MPS |
| `experiments/` | Notes, raw responses (`data/*.jsonl`), summaries and figures |
| `related-work/literature-review.md` | Prior work and community evaluations |
| `notes/pilot.md` | The initial coin/die pilot |
| `tests/` | Offline tests; no network or model needed |

## Setup

Python 3.12.

```
python3.12 -m venv venv
venv/bin/pip install -e ".[dev]"
echo "TYPESAFE_API_KEY=..." > .env        # or export TYPESAFE_API_KEY
venv/bin/pytest -q && venv/bin/ruff check src tests
```

The local-model backends also need `torch` and `transformers` (plus `laya` for Laya):

```
venv/bin/pip install torch transformers laya
```

## Running the experiments

Run from `src/`. Every sweep writes one JSON line per call, and every analyzer writes a
`.summary.json` and a figure.

```
cd src
../venv/bin/python -m jevcal.sweep -n 10              # exp 01: 1,330 Jev calls
../venv/bin/python -m jevcal.analyze
../venv/bin/python -m jevcal.sweep_natural -n 10      # exp 03: 1,560 calls
../venv/bin/python -m jevcal.analyze_natural
../venv/bin/python -m jevcal.bootstrap_0103           # exp 02 (offline)
../venv/bin/python -m jevcal.sweep -n 5 --rep-start 101 --out ../experiments/data/05-heldout-evidence-sweep.jsonl
../venv/bin/python -m jevcal.sweep_natural -n 5 --rep-start 101 --out ../experiments/data/05-heldout-natural-evidence.jsonl
../venv/bin/python -m jevcal.calibrate --heldout-witness ../experiments/data/05-heldout-evidence-sweep.jsonl \
    --heldout-natural ../experiments/data/05-heldout-natural-evidence.jsonl \
    --out ../experiments/data/05b-calibration-fixes-true-holdout.summary.json \
    --figure ../experiments/figures/05b-calibration-fixes-true-holdout.png
../venv/bin/python -m jevcal.chaosnli run && ../venv/bin/python -m jevcal.chaosnli analyze
../venv/bin/python -m jevcal.dices run && ../venv/bin/python -m jevcal.dices analyze
```

Other backends take `--backend {laya,qwen-rlcd,eve-rlcd} --device {cpu,mps}` together with
distinct `--out`/`--summary`/`--figure` paths. Exact commands are in each experiment note.

At the listed price of $0.042 per million input tokens, the full Jev audit (about 6,300
calls, 2.35M input tokens) costs about $0.10.

## External data and models

These are not redistributed here. Put them under `data/external/`, which is git-ignored.

| Item | Source | Check |
|---|---|---|
| ChaosNLI v1.0 (SNLI, MNLI-m) | The official Dropbox link is dead; copy from [jsbaan/calibration-on-disagreement-data](https://github.com/jsbaan/calibration-on-disagreement-data) `data/chaosNLI_v1.0/` into `data/external/chaosNLI_v1.0/` | git blob SHA: snli `aea16e8f…`, mnli `acf6a5a5…`. CC BY-NC 4.0 |
| DICES-350 | [google-research-datasets/dices-dataset](https://github.com/google-research-datasets/dices-dataset) `350/…csv` into `data/external/dices/` | git blob SHA `f1140b9d…`. CC BY 4.0. Contains offensive content |
| eve-rlcd code and data | [anthony-maio/eve-rlcd](https://github.com/anthony-maio/eve-rlcd) at commit `57a179b7` into `data/external/eve-rlcd/`; release `data-v1` into `data/external/eve-data/` | MIT |
| eve-rlcd model | `anthonym21/qwen3-0.6b-rlcd-decision` at revision `b327ec5e` into `data/external/models/qwen3-0.6b-rlcd-decision/` | The loader verifies sha256 |

## Notes

- Never use integer-like option keys with the Jev API. JavaScript clients re-sort them,
  which silently defeats order manipulations.
- Request `jev-latest` and record the `model` field of each response. `jev-1.13` is rejected.
- Raw responses in `experiments/data/` are append-only; the notes cite them.

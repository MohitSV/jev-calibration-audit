"""Hidden fair coin + noisy witnesses: every condition has an exact Bayesian answer.

A fair coin is flipped out of sight. Zero or more independent witnesses, each with a
stated accuracy q, report a side. The true posterior is
    logit P(heads) = sum over witnesses of ±logit(q)
so the model's probability can be scored against an exact target at every evidence level,
from none at all up to near-certainty.
"""

import math
from dataclasses import dataclass

HEADS_DESC = "The coin landed heads up"
TAILS_DESC = "The coin landed tails up"

# Label sets. `default` is the key we expect a prior to favour, or None for the control.
# No key is integer-like: JS clients re-sort integer-like object keys, which would
# silently defeat any order manipulation.
LABEL_SETS = {
    "heads_tails": ({"heads": HEADS_DESC, "tails": TAILS_DESC}, "heads"),
    "tails_heads": ({"tails": TAILS_DESC, "heads": HEADS_DESC}, "heads"),
    "opt1_is_heads": ({"option_1": HEADS_DESC, "option_2": TAILS_DESC}, "option_1"),
    "opt1_is_tails": ({"option_1": TAILS_DESC, "option_2": HEADS_DESC}, "option_1"),
    "neutral_kvx_heads": ({"kvx": HEADS_DESC, "mzr": TAILS_DESC}, None),
    "neutral_kvx_tails": ({"kvx": TAILS_DESC, "mzr": HEADS_DESC}, None),
}

CHOICE_INSTRUCTIONS = "Which side did the coin land on, given the evidence in the state?"
NOUL_HEADS = "Did the coin land heads up?"
NOUL_TAILS = "Did the coin land tails up?"


@dataclass(frozen=True)
class Condition:
    cid: str
    reports: tuple  # ((side, q), ...)

    @property
    def p_heads(self) -> float:
        return posterior_heads(self.reports)

    def state(self, record: int) -> str:
        head = f"Record #{record}. A fair coin was flipped once, out of sight of the analyst."
        if not self.reports:
            return f"{head} Nobody observed how it landed."
        lines = [
            f"Witness {i} reports correctly {round(q * 100)}% of the time and says it landed {side}."
            for i, (side, q) in enumerate(self.reports, 1)
        ]
        tail = " The witnesses reported independently." if len(self.reports) > 1 else ""
        return " ".join([head, *lines]) + tail


def posterior_heads(reports) -> float:
    z = sum((1 if side == "heads" else -1) * math.log(q / (1 - q)) for side, q in reports)
    return 1 / (1 + math.exp(-z))


def conditions() -> list[Condition]:
    out = [Condition("none", ())]
    for q in (0.55, 0.6, 0.7, 0.8, 0.9, 0.99):
        for side in ("heads", "tails"):
            out.append(Condition(f"w1_q{round(q * 100)}_{side}", ((side, q),)))
    out.append(Condition("conflict_q70_HT", (("heads", 0.7), ("tails", 0.7))))
    out.append(Condition("conflict_q70_TH", (("tails", 0.7), ("heads", 0.7))))
    for n in (2, 3):
        for side in ("heads", "tails"):
            out.append(Condition(f"w{n}_q60_{side}", tuple((side, 0.6) for _ in range(n))))
    return out


def mirror(cid: str) -> str:
    """The condition with every report flipped (same strength, opposite direction)."""
    if cid == "none":
        return cid
    if cid.startswith("conflict"):
        return "conflict_q70_TH" if cid.endswith("HT") else "conflict_q70_HT"
    return cid.replace("heads", "TMP").replace("tails", "heads").replace("TMP", "tails")

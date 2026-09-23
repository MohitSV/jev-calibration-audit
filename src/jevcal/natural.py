"""Experiment 03: does Choice still behave like a step function when the evidence is
natural language rather than a stated witness reliability?

Arm A (base rates): the evidence is a reference-class count ("of the 20 most similar
applications, 11 were approved"). The target is k/20 exactly. A Laplace estimate,
(k+1)/22, differs by at most 0.04.

Arm B (graded cues): support-ticket routing, where cue sentences range from none to
decisive and are written in parallel for each team. There is no exact target, so we
measure the *shape*: a probability should rise gradually with cue strength, while a
step function saturates at the first hint.
"""

from dataclasses import dataclass

BASE_RATE_KS = (2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18)  # symmetric: k <-> 20-k

DOMAINS = {
    "loan": {
        "text": "A small-business loan application is under review. Among the 20 most "
                "similar past applications at this bank, {k} were approved and {m} were "
                "denied. Nothing else distinguishes this application from those.",
        "yes": ("approved", "The application will be approved"),
        "no": ("denied", "The application will be denied"),
        "instructions": "What will the decision on this application be?",
        "noul_yes": "Will this application be approved?",
        "noul_no": "Will this application be denied?",
    },
    "routing": {
        "text": "A support ticket just arrived. Of the last 20 tickets with exactly the same "
                "wording, {k} were resolved by the billing team and {m} by the technical "
                "team. Nothing else is known about this ticket.",
        "yes": ("billing", "The billing team will resolve this ticket"),
        "no": ("technical", "The technical team will resolve this ticket"),
        "instructions": "Which team will resolve this ticket?",
        "noul_yes": "Will the billing team resolve this ticket?",
        "noul_no": "Will the technical team resolve this ticket?",
    },
    "refund": {
        "text": "A customer requested a refund. Of 20 comparable refund requests this "
                "quarter, {k} were granted and {m} were refused. This request is typical "
                "of those.",
        "yes": ("granted", "The refund will be granted"),
        "no": ("refused", "The refund will be refused"),
        "instructions": "Will this refund be granted or refused?",
        "noul_yes": "Will this refund be granted?",
        "noul_no": "Will this refund be refused?",
    },
}

# Arm B. Level 0 is identical for both teams; levels 1-5 are parallel and escalate.
CUES = {
    "billing": [
        "I have a question about my account.",
        "Something looks a bit odd on my account page, down near where the totals are.",
        "I'm not sure, but I may have been charged a different amount than usual.",
        "My invoice this month is higher than last month and I don't know why.",
        "I was charged twice for the same subscription on the 3rd.",
        ("I was charged twice for the same subscription on the 3rd. Please refund the "
         "duplicate $49 charge to my card."),
    ],
    "technical": [
        "I have a question about my account.",
        "Something looks a bit odd on my account page, it seems to load kind of slowly.",
        "I'm not sure, but the app may have logged me out a couple of times.",
        "The app keeps crashing when I open the dashboard.",
        "The app crashes every time I open the dashboard, with error code 500.",
        ("Every time I open the dashboard the app crashes with error 500. I've reinstalled "
         "it twice and cleared the cache."),
    ],
}
ROUTING_OPTS = {
    "billing": "The billing team should handle this ticket",
    "technical": "The technical support team should handle this ticket",
}
ROUTING_INSTRUCTIONS = "Which team should this ticket be routed to?"
ROUTING_NOUL = {
    "billing": "Should this ticket be routed to the billing team?",
    "technical": "Should this ticket be routed to the technical support team?",
}


@dataclass(frozen=True)
class Item:
    arm: str        # "base_rate" | "cue"
    domain: str     # base-rate domain, or "ticket"
    level: str      # "k=11" | "billing:L3" | "mixed:BT"
    p_true: float | None   # P(first-named option: yes-side / billing); None when unknown
    text: str


def base_rate_items():
    for d, spec in DOMAINS.items():
        for k in BASE_RATE_KS:
            yield Item("base_rate", d, f"k={k}", k / 20, spec["text"].format(k=k, m=20 - k))


def cue_items():
    for team, cues in CUES.items():
        for lvl in range(1, len(cues)):
            yield Item("cue", "ticket", f"{team}:L{lvl}", None, f'Ticket text: "{cues[lvl]}"')
    yield Item("cue", "ticket", "neutral:L0", 0.5, f'Ticket text: "{CUES["billing"][0]}"')
    b, t = CUES["billing"][3], CUES["technical"][3]
    yield Item("cue", "ticket", "mixed:BT", 0.5, f'Ticket text: "{b} Also, {t[0].lower()}{t[1:]}"')
    yield Item("cue", "ticket", "mixed:TB", 0.5, f'Ticket text: "{t} Also, {b[0].lower()}{b[1:]}"')


def choice_variants(item):
    """(variant name, criteria, key of the 'yes'/billing side). Both list orders."""
    if item.arm == "base_rate":
        spec = DOMAINS[item.domain]
        (yk, yd), (nk, nd) = spec["yes"], spec["no"]
        return [("yes_first", {yk: yd, nk: nd}, yk), ("no_first", {nk: nd, yk: yd}, yk)]
    b, t = ROUTING_OPTS["billing"], ROUTING_OPTS["technical"]
    return [("billing_first", {"billing": b, "technical": t}, "billing"),
            ("technical_first", {"technical": t, "billing": b}, "billing")]


def instructions(item):
    return DOMAINS[item.domain]["instructions"] if item.arm == "base_rate" else ROUTING_INSTRUCTIONS


def noul_questions(item):
    if item.arm == "base_rate":
        spec = DOMAINS[item.domain]
        return spec["noul_yes"], spec["noul_no"]
    return ROUTING_NOUL["billing"], ROUTING_NOUL["technical"]

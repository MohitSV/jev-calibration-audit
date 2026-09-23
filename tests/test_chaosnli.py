import math

from jevcal import chaosnli as c


def _fake_items(n_per_share=80):
    items = []
    for i, share in enumerate((0.4, 0.55, 0.65, 0.8, 0.95)):
        for j in range(n_per_share):
            rest = (1 - share) / 2
            items.append({"uid": f"{i}-{j}", "source": "snli", "premise": "p", "hypothesis": "h",
                          "human": {"entailment": share, "neutral": rest, "contradiction": rest}})
    return items


def test_sample_is_stratified_and_deterministic():
    s1, s2 = c.sample(_fake_items()), c.sample(_fake_items())
    assert [x["uid"] for x in s1] == [x["uid"] for x in s2]
    assert len(s1) == c.PER_BIN * len(c.BINS)
    for b in range(len(c.BINS)):
        assert sum(x["bin"] == b for x in s1) == c.PER_BIN


def test_bins_cover_all_possible_shares():
    for share in (1 / 3, 0.5, 0.6, 0.7, 0.85, 1.0):
        c.bin_of(share)


def test_kl_zero_for_perfect_and_positive_otherwise():
    h = {"entailment": 0.6, "neutral": 0.3, "contradiction": 0.1}
    assert math.isclose(c.kl(h, h), 0.0, abs_tol=1e-12)
    assert c.kl(h, {"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}) > 4


def test_summary_prefers_the_method_that_matches_humans():
    rows = []
    for it in c.sample(_fake_items()):
        h = it["human"]
        confident = {"entailment": 0.99, "neutral": 0.005, "contradiction": 0.005}
        base = {"uid": it["uid"], "source": "snli", "bin": it["bin"], "human": h,
                "model": "fake", "input_tokens": 1}
        for v in ("enc", "cne"):
            rows.append(base | {"kind": "choice", "variant": v, "probs": confident})
        rows.append(base | {"kind": "noul", "variant": "noul", "noul_raw": dict(h)})
    s = c.summarize(rows, n_boot=50)
    assert s["methods"]["noul_norm"]["kl"] < 1e-9 < s["methods"]["choice_avg"]["kl"]
    assert s["bootstrap_ci95"]["kl_noul_minus_choice_avg"][1] < 0


def test_real_data_loads_if_present():
    if not (c.DATA_DIR / c.FILES[0]).exists():
        return
    items = c.load_items()
    assert len(items) == 3113
    assert all(math.isclose(sum(i["human"].values()), 1, abs_tol=0.02) for i in items)
    assert len(c.sample(items)) == 300

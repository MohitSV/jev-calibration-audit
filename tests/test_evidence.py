import math

from jevcal.analyze import strength_pairs, summarize
from jevcal.evidence import LABEL_SETS, Condition, conditions, mirror, posterior_heads


def test_posterior_known_values():
    assert posterior_heads(()) == 0.5
    assert math.isclose(posterior_heads((("heads", 0.7),)), 0.7)
    assert math.isclose(posterior_heads((("tails", 0.9),)), 0.1)
    assert math.isclose(posterior_heads((("heads", 0.7), ("tails", 0.7))), 0.5)
    # two agreeing 60% witnesses: odds 1.5^2 = 2.25
    assert math.isclose(posterior_heads((("heads", 0.6),) * 2), 2.25 / 3.25)


def test_every_condition_has_a_mirror_with_complementary_truth():
    by_id = {c.cid: c for c in conditions()}
    for c in by_id.values():
        m = by_id[mirror(c.cid)]
        assert math.isclose(c.p_heads + m.p_heads, 1.0)


def test_no_integer_like_keys():
    for criteria, _ in LABEL_SETS.values():
        assert not any(k.isdigit() for k in criteria)


def test_state_mentions_every_witness():
    s = Condition("x", (("heads", 0.6), ("tails", 0.9))).state(3)
    assert "Record #3" in s and "60%" in s and "90%" in s and "independently" in s


def _fake_rows(bias_heads=0.0, bias_opt1=0.0):
    """Synthetic Jev that reports truth + a fixed semantic and label pull."""
    rows = []
    for c in conditions():
        for name, (criteria, _) in LABEL_SETS.items():
            opt1_heads = criteria.get("option_1", "").endswith("heads up")
            pull = bias_heads
            if "option_1" in criteria:
                pull += bias_opt1 if opt1_heads else -bias_opt1
            p = min(max(c.p_heads + pull, 0), 1)
            key_h = next(k for k, v in criteria.items() if v.endswith("heads up"))
            key_t = next(k for k in criteria if k != key_h)
            rows.append({"kind": "choice", "label_set": name, "cid": c.cid, "model": "fake",
                         "input_tokens": 1, "probabilities": {key_h: p, key_t: 1 - p}})
        rows.append({"kind": "noul", "label_set": "noul", "cid": c.cid, "model": "fake",
                     "input_tokens": 1, "noul_heads": c.p_heads, "noul_tails": 1 - c.p_heads})
    return rows


def test_decomposition_recovers_planted_effects():
    s = summarize(_fake_rows(bias_heads=0.1, bias_opt1=0.2))
    none = next(r for r in s["decomposition_by_strength"] if r["pair"] == "none")
    assert math.isclose(none["semantic_effect"], 0.1, abs_tol=1e-9)
    assert math.isclose(none["label_effect"], 0.2, abs_tol=1e-9)
    assert math.isclose(none["position_effect"], 0.0, abs_tol=1e-9)


def test_perfect_model_has_no_pull():
    s = summarize(_fake_rows())
    for r in s["decomposition_by_strength"]:
        assert all(abs(v) < 1e-9 for v in r["pulls"].values())
    assert s["mean_abs_error_vs_bayes"]["noul"] < 1e-9


def test_strength_pairs_cover_all_conditions_once():
    ids = [x for a, b, _ in strength_pairs() for x in {a, b}]
    assert sorted(ids) == sorted(c.cid for c in conditions())


def test_natural_items_are_symmetric_and_well_formed():
    from jevcal.natural import BASE_RATE_KS, CUES, base_rate_items, choice_variants, cue_items

    assert sorted(BASE_RATE_KS) == sorted(20 - k for k in BASE_RATE_KS)
    assert len({len(v) for v in CUES.values()}) == 1  # parallel cue ladders
    for item in [*base_rate_items(), *cue_items()]:
        variants = choice_variants(item)
        assert len(variants) == 2
        (_, c1, y1), (_, c2, y2) = variants
        assert set(c1) == set(c2) and list(c1) != list(c2) and y1 == y2
        assert not any(k.isdigit() for k in c1)


def test_sweep_rep_start_is_respected():
    from jevcal.sweep import jobs as evidence_jobs
    from jevcal.sweep_natural import jobs as natural_jobs

    assert {job[3] for job in evidence_jobs(2, 101)} == {101, 102}
    assert {job[2] for job in natural_jobs(2, 101)} == {101, 102}

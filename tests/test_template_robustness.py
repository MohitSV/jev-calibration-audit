from jevcal import template_robustness as tr


def test_job_count_and_states():
    jobs = list(tr.jobs())
    assert len(jobs) == 3 * 3 * 5 * 3 * 2
    s = tr.state("percent", "loan", 0.55, 301)
    assert "55%" in s and s.startswith("Case 301.")
    assert "0.45" in tr.state("probability", "refund", 0.45, 302)
    assert "11 were approved" in tr.state("count", "loan", 0.55, 301)


def test_summary_flags_a_step_and_a_tracker():
    rows = []
    for t in tr.TEMPLATES:
        for p in tr.RATES:
            step = 0.02 if p < 0.5 else (0.8 if p == 0.5 else 0.98)
            rows.append({"template": t, "kind": "choice", "p_true": p, "p_yes": step,
                         "model": "fake", "input_tokens": 1})
            rows.append({"template": t, "kind": "noul", "p_true": p, "p_yes": p,
                         "model": "fake", "input_tokens": 1})
    s = tr.summarize(rows)
    assert s["P1_all_jumps_gt_0.3"] and s["P2_all_noul_mae_le_0.05"]
    assert s["templates"]["count"]["noul_mae"] == 0

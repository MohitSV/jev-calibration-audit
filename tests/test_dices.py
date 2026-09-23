import math

from jevcal import dices as d


def test_binary_kl():
    assert math.isclose(d.kl(0.3, 0.3), 0.0, abs_tol=1e-12)
    assert d.kl(0.5, 0.99) > d.kl(0.5, 0.7) > 0


def test_bins_cover_all_shares():
    for p in (0.0, 0.1, 0.45, 0.5, 0.55, 0.95, 1.0):
        d.bin_of(p)


def test_summary_prefers_matching_method_and_temperature_rescues_monotone_overconfidence():
    rows = []
    for i in range(100):
        h = 0.05 + 0.9 * i / 99
        over = d.temper(h, 0.2)  # same ranking, far too extreme: T=5 recovers it exactly
        base = {"uid": str(i), "p_human": h, "model": "fake", "input_tokens": 1}
        for v in ("us", "su"):
            rows.append(base | {"kind": "choice", "variant": v, "p_unsafe": over})
        rows.append(base | {"kind": "noul", "variant": "noul", "p_unsafe": h,
                            "noul_unsafe": h, "noul_safe": 1 - h})
    s = d.summarize(rows, n_boot=50)
    assert s["methods"]["noul_norm"]["kl"] < 1e-9 < s["methods"]["choice_avg"]["kl"]
    assert s["bootstrap_ci95"]["kl_noul_minus_choice_avg"][1] < 0
    assert abs(s["temperature_check_P4"]["gap_choice_minus_noul"]) < 0.01


def test_real_data_loads_if_present():
    if not d.CSV_PATH.exists():
        return
    items = d.load_items()
    assert len(items) == 350
    assert all(0 <= i["p_unsafe"] <= 1 and i["n_votes"] == 104 for i in items)

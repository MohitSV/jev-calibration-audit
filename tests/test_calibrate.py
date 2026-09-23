import math

from jevcal.calibrate import fit_slope, normalized_noul, smooth_ece


def test_fit_slope_recovers_inverse_compression():
    targets = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    compressed = [1 / (1 + math.exp(-0.5 * math.log(y / (1 - y)))) for y in targets]
    assert math.isclose(fit_slope(compressed, targets), 2.0, rel_tol=1e-4)


def test_smooth_ece_is_zero_for_exact_soft_targets():
    ps = [0.1, 0.2, 0.5, 0.8, 0.9]
    assert smooth_ece(ps, ps) < 1e-12


def test_normalized_noul_handles_incoherent_complements():
    observation = type("O", (), {"noul_yes": 0.6, "noul_no": 0.2})()
    assert math.isclose(normalized_noul(observation), 0.75)

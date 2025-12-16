import numpy as np

from pnl.logistic_fit import fit_global_de_then_local, make_synthetic_data
from scipy.optimize import Bounds


def test_logistic_fit_recovers_parameters_reasonably() -> None:
    t, y, true_params = make_synthetic_data(seed=123, noise_std=1.0)
    bounds = Bounds([0.0, 0.0, -5.0], [200.0, 2.0, 40.0])

    _, ref = fit_global_de_then_local(t, y, bounds=bounds, seed=123)

    L, k, t0 = ref.theta
    L_true, k_true, t0_true = true_params

    assert abs(L - L_true) < 10.0
    assert abs(k - k_true) < 0.15
    assert abs(t0 - t0_true) < 2.5

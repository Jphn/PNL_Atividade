import numpy as np

from pnl.prototipo import (
    is_feasible,
    objective_prototipo,
    solve_global_then_local,
)


def test_global_then_local_feasible_and_improves() -> None:
    g, ref = solve_global_then_local(seed=0)

    assert is_feasible(ref.x, tol=1e-5)
    assert objective_prototipo(ref.x) <= objective_prototipo(g.x) + 1e-6


def test_objective_is_deterministic() -> None:
    x = np.array([0.1, 0.2])
    v1 = objective_prototipo(x)
    v2 = objective_prototipo(x)
    assert float(v1) == float(v2)

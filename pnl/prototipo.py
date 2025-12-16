import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    differential_evolution,
    minimize,
)


def objective_prototipo(x: np.ndarray) -> float:
    x1, x2 = float(x[0]), float(x[1])
    return (x1 - 1.0) ** 2 + (x2 - 2.0) ** 2 + 0.5 * np.sin(3.0 * x1) * np.sin(3.0 * x2)


def grad_objective_prototipo(x: np.ndarray) -> np.ndarray:
    x1, x2 = float(x[0]), float(x[1])
    d1 = 2.0 * (x1 - 1.0) + 0.5 * (3.0 * np.cos(3.0 * x1)) * np.sin(3.0 * x2)
    d2 = 2.0 * (x2 - 2.0) + 0.5 * np.sin(3.0 * x1) * (3.0 * np.cos(3.0 * x2))
    return np.array([d1, d2], dtype=float)


def make_constraints_prototipo() -> Dict[str, object]:
    # C1: x1^2 + x2^2 <= 5
    nl = NonlinearConstraint(lambda x: x[0] ** 2 + x[1] ** 2, -np.inf, 5.0)

    # C2: x1 + x2 >= 1  <=>  -x1 - x2 <= -1
    lin = LinearConstraint(np.array([[-1.0, -1.0]]), -np.inf, np.array([-1.0]))

    return {"nonlinear": nl, "linear": lin}


@dataclass
class SolveResult:
    method: str
    x: np.ndarray
    fun: float
    success: bool
    message: str
    path: np.ndarray | None = None


def solve_local(
    x0: np.ndarray,
    method: str,
    bounds: Bounds,
    nonlinear: NonlinearConstraint,
    linear: LinearConstraint,
) -> SolveResult:
    path: List[np.ndarray] = []

    def cb(xk, *_args):
        path.append(np.array(xk, dtype=float))

    constraints = [nonlinear, linear]
    if method.upper() == "SLSQP":
        # SLSQP é mais robusto com o formato "antigo" de restrições (dict).
        # Para SLSQP, ineq significa fun(x) >= 0.
        constraints = [
            {"type": "ineq", "fun": lambda x: 5.0 - (x[0] ** 2 + x[1] ** 2)},
            {"type": "ineq", "fun": lambda x: (x[0] + x[1]) - 1.0},
        ]

    res = minimize(
        objective_prototipo,
        x0=np.array(x0, dtype=float),
        jac=grad_objective_prototipo,
        method=method,
        bounds=bounds,
        constraints=constraints,
        callback=cb,
        options={"maxiter": 500},
    )

    return SolveResult(
        method=method,
        x=np.array(res.x, dtype=float),
        fun=float(res.fun),
        success=bool(res.success),
        message=str(res.message),
        path=np.array(path, dtype=float) if path else None,
    )


def solve_global_then_local(seed: int = 0) -> Tuple[SolveResult, SolveResult]:
    bounds_de = [(-3.0, 3.0), (-3.0, 3.0)]

    def penalty(x: np.ndarray) -> float:
        # Penaliza violações: max(0, c(x))^2 com pesos altos
        c1 = max(0.0, x[0] ** 2 + x[1] ** 2 - 5.0)  # deve ser <= 0
        c2 = max(0.0, 1.0 - (x[0] + x[1]))  # deve ser <= 0
        return 1e4 * (c1 * c1 + c2 * c2)

    def penalized_obj(x: np.ndarray) -> float:
        return objective_prototipo(x) + penalty(x)

    de = differential_evolution(
        penalized_obj,
        bounds=bounds_de,
        seed=seed,
        polish=False,
        updating="deferred",
        tol=1e-7,
        maxiter=400,
        popsize=20,
    )

    global_res = SolveResult(
        method="differential_evolution(penalized)",
        x=np.array(de.x, dtype=float),
        fun=float(objective_prototipo(de.x)),
        success=bool(de.success),
        message=str(de.message),
        path=None,
    )

    cons = make_constraints_prototipo()
    bounds_local = Bounds([-3.0, -3.0], [3.0, 3.0])
    local_ref = solve_local(
        x0=global_res.x,
        method="trust-constr",
        bounds=bounds_local,
        nonlinear=cons["nonlinear"],
        linear=cons["linear"],
    )

    return global_res, local_ref


def is_feasible(x: np.ndarray, tol: float = 1e-6) -> bool:
    x1, x2 = float(x[0]), float(x[1])
    return (x1 * x1 + x2 * x2 <= 5.0 + tol) and (x1 + x2 >= 1.0 - tol)


def main() -> None:
    cons = make_constraints_prototipo()
    bounds = Bounds([-3.0, -3.0], [3.0, 3.0])

    inits = [np.array([-2.5, 2.5]), np.array([2.5, -1.0]), np.array([0.2, 0.9])]
    methods = ["SLSQP", "trust-constr"]

    print("=== Protótipo 2D (não convexo, com restrições) ===")
    for x0 in inits:
        for m in methods:
            r = solve_local(x0, m, bounds, cons["nonlinear"], cons["linear"])
            print(f"{m:12s} x0={x0} -> x={r.x}, f={r.fun:.6f}, feas={is_feasible(r.x)}, ok={r.success}")

    g, ref = solve_global_then_local(seed=0)
    print("\n=== Global → Local ===")
    print(f"Global  : x={g.x}, f={g.fun:.6f}, feas={is_feasible(g.x)}")
    print(f"Refino  : x={ref.x}, f={ref.fun:.6f}, feas={is_feasible(ref.x)}, ok={ref.success}")


if __name__ == "__main__":
    main()

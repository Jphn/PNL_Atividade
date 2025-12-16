import numpy as np
from dataclasses import dataclass
from typing import Tuple

from scipy.optimize import Bounds, differential_evolution, minimize


def logistic(t: np.ndarray, L: float, k: float, t0: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return L / (1.0 + np.exp(-k * (t - t0)))


def make_synthetic_data(
    n: int = 60,
    seed: int = 0,
    true_params: Tuple[float, float, float] = (100.0, 0.35, 12.0),
    noise_std: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 25.0, n)
    L, k, t0 = true_params
    y_clean = logistic(t, L, k, t0)
    y = y_clean + rng.normal(0.0, noise_std, size=n)
    return t, y, true_params


def sse(theta: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
    L, k, t0 = float(theta[0]), float(theta[1]), float(theta[2])
    yhat = logistic(t, L, k, t0)
    r = y - yhat
    return float(np.dot(r, r))


@dataclass
class FitResult:
    method: str
    theta: np.ndarray
    fun: float
    success: bool
    message: str


def fit_local_lbfgsb(
    t: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    bounds: Bounds,
) -> FitResult:
    res = minimize(
        lambda th: sse(th, t, y),
        x0=np.array(x0, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000},
    )
    return FitResult(
        method="L-BFGS-B",
        theta=np.array(res.x, dtype=float),
        fun=float(res.fun),
        success=bool(res.success),
        message=str(res.message),
    )


def fit_global_de_then_local(
    t: np.ndarray,
    y: np.ndarray,
    bounds: Bounds,
    seed: int = 0,
) -> Tuple[FitResult, FitResult]:
    bounds_de = list(zip(bounds.lb.tolist(), bounds.ub.tolist()))
    de = differential_evolution(
        lambda th: sse(th, t, y),
        bounds=bounds_de,
        seed=seed,
        tol=1e-7,
        maxiter=600,
        popsize=20,
        updating="deferred",
        polish=False,
    )
    global_res = FitResult(
        method="differential_evolution",
        theta=np.array(de.x, dtype=float),
        fun=float(de.fun),
        success=bool(de.success),
        message=str(de.message),
    )

    local_ref = fit_local_lbfgsb(t, y, x0=global_res.theta, bounds=bounds)
    local_ref.method = "L-BFGS-B (refino)"
    return global_res, local_ref


def main() -> None:
    t, y, true_params = make_synthetic_data(seed=0)
    bounds = Bounds([0.0, 0.0, -5.0], [200.0, 2.0, 40.0])

    print("=== Ajuste logístico (mínimos quadrados não linear) ===")
    print(f"Parâmetros verdadeiros: L={true_params[0]}, k={true_params[1]}, t0={true_params[2]}")

    inits = [np.array([60.0, 0.05, 5.0]), np.array([150.0, 1.2, 18.0]), np.array([90.0, 0.3, 10.0])]
    for x0 in inits:
        r = fit_local_lbfgsb(t, y, x0=x0, bounds=bounds)
        print(f"Local x0={x0} -> theta={r.theta}, SSE={r.fun:.3f}, ok={r.success}")

    g, ref = fit_global_de_then_local(t, y, bounds=bounds, seed=0)
    print("\n=== Global → Local ===")
    print(f"Global  : theta={g.theta}, SSE={g.fun:.3f}, ok={g.success}")
    print(f"Refino  : theta={ref.theta}, SSE={ref.fun:.3f}, ok={ref.success}")


if __name__ == "__main__":
    main()

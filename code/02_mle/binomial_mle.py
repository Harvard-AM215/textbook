"""Binomial(n=2, p) MLE: analytic vs numeric.

Each observation is the sum of two independent Bernoulli(p) flips, so
X_i in {0, 1, 2} with probabilities (1-p)^2, 2p(1-p), p^2.

Companion to Chapter 2 (Maximum Likelihood Estimation).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def nll_binomial2(p, n0, n1, n2):
    """Negative log-likelihood for n0+n1+n2 Binomial(2, p) observations."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(2 * n0 * np.log(1 - p)
             + n1 * (np.log(2) + np.log(p) + np.log(1 - p))
             + 2 * n2 * np.log(p))


def fit(data, p_init=0.5):
    n0, n1, n2 = np.bincount(data.astype(np.intp), minlength=3)
    res = minimize(nll_binomial2, x0=p_init, args=(n0, n1, n2),
                   bounds=[(1e-9, 1 - 1e-9)])
    return res.x.item(), (n0, n1, n2)


def main():
    rng = np.random.default_rng(seed=1)
    p_true, N = 0.7, 50
    data = rng.binomial(2, p_true, size=N)

    p_numeric, (n0, n1, n2) = fit(data)
    p_analytic = (2 * n2 + n1) / (2 * N)

    print(f"Counts: n0={n0}, n1={n1}, n2={n2}  (N={N})")
    print(f"True value          p = {p_true:.4f}")
    print(f"Analytic estimate   p = {p_analytic:.4f}")
    print(f"Numeric estimate    p = {p_numeric:.4f}")

    grid = np.linspace(1e-3, 1 - 1e-3, 300)
    nll_curve = np.array([nll_binomial2(pp, n0, n1, n2) for pp in grid])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, nll_curve, "-", color="C3", label="NLL(p)")
    ax.axvline(p_true, ls="--", color="k", alpha=0.5, label=f"true p = {p_true}")
    ax.plot(p_analytic, nll_binomial2(p_analytic, n0, n1, n2), "s",
            ms=10, label=f"analytic = {p_analytic:.3f}")
    ax.plot(p_numeric, nll_binomial2(p_numeric, n0, n1, n2), "*",
            ms=12, label=f"numeric  = {p_numeric:.3f}")
    ax.set(xlabel="p", ylabel="negative log-likelihood",
           title=f"Binomial(2, p) MLE (N = {N})")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

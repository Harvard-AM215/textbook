"""Bernoulli MLE: analytic vs numeric, with a sweep of the NLL curve.

Companion to Chapter 2 (Maximum Likelihood Estimation). Run end-to-end:

    uv run python bernoulli_mle.py

Produces a console comparison and an NLL plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def nll_bernoulli(p, x):
    """Negative log-likelihood for Bernoulli trials.

    Parameters
    ----------
    p : float
        Success probability (must be in (0, 1)).
    x : ndarray of {0, 1}
        Observed trials.
    """
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.sum(x * np.log(p) + (1 - x) * np.log(1 - p))


def fit(data, p_init=0.5):
    """Numerically minimize the Bernoulli NLL given data."""
    res = minimize(nll_bernoulli, x0=p_init, args=(data,),
                   bounds=[(1e-9, 1 - 1e-9)])
    return res.x.item()


def main():
    rng = np.random.default_rng(seed=0)
    p_true, n = 0.3, 500
    data = rng.binomial(1, p_true, size=n)

    p_analytic = data.mean()
    p_numeric = fit(data)

    print(f"True value          p = {p_true:.4f}")
    print(f"Analytic estimate   p = {p_analytic:.4f}")
    print(f"Numeric estimate    p = {p_numeric:.4f}")

    # NLL curve: feasible because p is one-dimensional.
    grid = np.linspace(1e-3, 1 - 1e-3, 300)
    nll_curve = np.array([nll_bernoulli(pp, data) for pp in grid])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, nll_curve, "-", color="C3", label="NLL(p)")
    ax.axvline(p_true, ls="--", color="k", alpha=0.5, label=f"true p = {p_true}")
    ax.plot(p_analytic, nll_bernoulli(p_analytic, data), "s",
            ms=10, label=f"analytic = {p_analytic:.3f}")
    ax.plot(p_numeric, nll_bernoulli(p_numeric, data), "*",
            ms=12, label=f"numeric  = {p_numeric:.3f}")
    ax.set(xlabel="p", ylabel="negative log-likelihood",
           title=f"Bernoulli MLE (n = {n})")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

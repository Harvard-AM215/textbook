"""Empirical 1/sqrt(n) scaling of the Bernoulli MLE's standard error.

For each n, draw `n_repeats` datasets of size n, compute p_hat for each,
and record the standard deviation of p_hat across repeats. Plot SD vs n on
log-log axes; the slope should be -0.5, matching SE = sqrt(p(1-p)/n).

Solution to Exercise 3 of Chapter 2.
"""

import numpy as np
import matplotlib.pyplot as plt


def empirical_se(p_true, n, n_repeats, rng):
    """Std. dev. of p_hat = sample mean across `n_repeats` Bernoulli(p_true)
    samples of size n."""
    samples = rng.binomial(1, p_true, size=(n_repeats, n))
    p_hats = samples.mean(axis=1)
    return p_hats.std(ddof=1)


def main():
    rng = np.random.default_rng(seed=42)
    p_true = 0.3
    ns = np.array([10, 30, 100, 300, 1000, 3000, 10_000])
    n_repeats = 500

    ses = np.array([empirical_se(p_true, n, n_repeats, rng) for n in ns])
    theoretical = np.sqrt(p_true * (1 - p_true) / ns)

    # Linear fit on log-log to read off the slope.
    slope, intercept = np.polyfit(np.log(ns), np.log(ses), 1)
    print(f"Empirical slope on log-log: {slope:.3f}  (theory: -0.5)")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(ns, ses, "o-", label=f"empirical SD($\\hat p$), slope={slope:.2f}")
    ax.loglog(ns, theoretical, "--", label=r"$\sqrt{p(1-p)/n}$")
    ax.set(xlabel="n", ylabel=r"SD($\hat p$)",
           title=f"Standard error of Bernoulli MLE (p={p_true})")
    ax.grid(which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

"""Simulate GBM paths; compare them with the lognormal predictions.

Companion to Chapter 6 Worked Example 1. Prints the same three comparisons as the
chapter's cell, then the Monte Carlo standard error of the median (about 0.26, behind
the chapter's "the median within its noise"), then plots paths and the terminal
log-price distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, rng):
    """GBM paths on a grid of n_steps + 1 times, one row per path."""
    dt = T / n_steps
    # Itô-corrected log increments: the log-price drifts at mu - sigma^2/2.
    Z = rng.standard_normal((n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.log(S0) + np.cumsum(log_increments, axis=1)
    return np.hstack([np.full((n_paths, 1), S0), np.exp(log_paths)])


def median_standard_error(S0, mu, sigma, T, n_paths, n_repeats, rng):
    """Spread of the sample median of n_paths terminal prices, over n_repeats runs.

    Draws S_T directly from its lognormal distribution: the log increments are exact,
    so this has the same distribution as the last column of simulate_gbm.
    """
    Z = rng.standard_normal((n_repeats, n_paths))
    ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return np.median(ST, axis=1).std(ddof=1)


def main():
    rng = np.random.default_rng(seed=0)
    S0, mu, sigma, T = 100.0, 0.10, 0.20, 1.0
    paths = simulate_gbm(S0, mu, sigma, T, n_steps=252, n_paths=10_000, rng=rng)
    ST = paths[:, -1]

    print(f"mean   S_T = {ST.mean():.2f} +/- {ST.std(ddof=1) / np.sqrt(ST.size):.2f}   "
          f"(theory: {S0 * np.exp(mu * T):.2f})")
    print(f"median S_T = {np.median(ST):.2f}            "
          f"(theory: {S0 * np.exp((mu - sigma**2 / 2) * T):.2f})")
    print(f"SD of log S_T = {np.log(ST).std(ddof=1):.4f}      "
          f"(theory: {sigma * np.sqrt(T):.4f})")
    se_median = median_standard_error(S0, mu, sigma, T, n_paths=10_000, n_repeats=500,
                                      rng=np.random.default_rng(seed=1))
    print(f"standard error of the median of 10,000 paths (500 repeats): {se_median:.2f}")

    times = np.linspace(0, T, paths.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for p in paths[:50]:
        ax.plot(times, p, "k-", alpha=0.15)
    ax.plot(times, S0 * np.exp(mu * times), "C3-", lw=2,
            label=r"$S_0 e^{\mu t}$ (mean)")
    ax.plot(times, S0 * np.exp((mu - 0.5 * sigma**2) * times), "C0-", lw=2,
            label=r"$S_0 e^{(\mu - \sigma^2/2) t}$ (median)")
    ax.set(xlabel="t (years)", ylabel=r"$S_t$",
           title=rf"50 GBM paths ($\mu$ = {mu}, $\sigma$ = {sigma})")
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[1]
    log_ST = np.log(ST)
    mu_log = np.log(S0) + (mu - 0.5 * sigma**2) * T
    sd_log = sigma * np.sqrt(T)
    ax.hist(log_ST, bins=40, density=True, alpha=0.6, label="simulated")
    xs = np.linspace(log_ST.min(), log_ST.max(), 300)
    ax.plot(xs, norm.pdf(xs, mu_log, sd_log), "C3-", lw=2,
            label=r"predicted $\mathcal{N}(\log S_0 + (\mu - \sigma^2/2)T,\ \sigma^2 T)$")
    ax.set(xlabel=r"$\log S_T$", ylabel="density",
           title=r"Terminal log-price against its predicted Gaussian")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

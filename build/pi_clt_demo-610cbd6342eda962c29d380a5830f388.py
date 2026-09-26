"""CLT demo: distribution of pi-hat over many independent MC runs.

Run the dart-throwing pi estimator 1000 times at N=10000 each, then
histogram the resulting estimates. By the CLT they should be approximately
Gaussian with mean pi and SD ~ 4*sqrt(p(1-p)/N) where p = pi/4.

Solution to Chapter 3 Exercise 4.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def estimate_pi(N, rng):
    pts = rng.uniform(0, 1, size=(N, 2))
    inside = (pts[:, 0]**2 + pts[:, 1]**2) < 1.0
    return 4 * inside.mean()


def main():
    rng = np.random.default_rng(seed=7)
    N = 10_000
    n_runs = 1000

    estimates = np.array([estimate_pi(N, rng) for _ in range(n_runs)])

    p = np.pi / 4
    theory_sd = 4 * np.sqrt(p * (1 - p) / N)
    print(f"empirical mean    = {estimates.mean():.5f}  (theory: {np.pi:.5f})")
    print(f"empirical SD      = {estimates.std(ddof=1):.5f}  (theory: {theory_sd:.5f})")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(estimates, bins=40, density=True, alpha=0.6, label="empirical")
    xs = np.linspace(estimates.min(), estimates.max(), 300)
    ax.plot(xs, norm.pdf(xs, np.pi, theory_sd), "C3-",
            lw=2, label=r"theoretical $\mathcal{N}(\pi, \sigma^2/N)$")
    ax.axvline(np.pi, ls="--", color="k", alpha=0.5, label=r"$\pi$")
    ax.set(xlabel=r"$\hat\pi$", ylabel="density",
           title=f"CLT for MC pi estimator (N={N}, runs={n_runs})")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

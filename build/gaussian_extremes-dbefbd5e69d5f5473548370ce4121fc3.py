"""The maximum of n Gaussian samples: how it grows with n, and its Gumbel limit.

Companion to Chapter 8 Worked Example 1 and solution to Exercise 3. For each n it prints
the mean and standard deviation of M block maxima, the exact expectation of the maximum
(the integral of x n phi(x) Phi(x)^(n-1)) and the leading-order sqrt(2 ln n). It then
standardizes the maxima at the largest n with Exercise 2's constants and compares their
histogram with the Gumbel density.

The draws are made in chunks: at n = 100,000 and M = 10,000 the full sample is 10^9
numbers, 8 GB, which is why the chapter's cell stops at n = 10,000.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gumbel_r, norm

SIZES = (10, 100, 1000, 10_000, 100_000)
M = 10_000


def gaussian_maxima(n, M, rng, chunk_elements=20_000_000):
    """M maxima of n standard Gaussians each, drawn in chunks to bound memory."""
    rows = max(1, chunk_elements // n)
    out, remaining = [], M
    while remaining > 0:
        k = min(rows, remaining)
        out.append(rng.standard_normal((k, n)).max(axis=1))
        remaining -= k
    return np.concatenate(out)


def expected_max(n):
    """E[M_n] for n standard Gaussians: the integral of x n phi(x) Phi(x)^(n-1)."""
    x = np.linspace(-10, 10, 2_000_001)
    return np.trapezoid(x * n * norm.pdf(x) * norm.cdf(x) ** (n - 1), x)


def gumbel_constants(n):
    """Exercise 2's centering b_n and scale a_n: (M_n - b_n) / a_n -> Gumbel."""
    L = 2 * np.log(n)
    b_n = np.sqrt(L) - (np.log(np.log(n)) + np.log(4 * np.pi)) / (2 * np.sqrt(L))
    return b_n, 1 / np.sqrt(L)


def main():
    rng = np.random.default_rng(seed=0)
    print(f"{'n':>8} {'mean(max)':>10} {'SD(max)':>8} {'exact E[max]':>13} {'sqrt(2 ln n)':>13}")
    maxima = {}
    for n in SIZES:
        maxima[n] = gaussian_maxima(n, M, rng)
        print(f"{n:>8} {maxima[n].mean():>10.3f} {maxima[n].std(ddof=1):>8.3f} "
              f"{expected_max(n):>13.3f} {np.sqrt(2 * np.log(n)):>13.3f}")

    n = SIZES[-1]
    b_n, a_n = gumbel_constants(n)
    z = (maxima[n] - b_n) / a_n
    print(f"\nstandardized maxima at n = {n}: b_n = {b_n:.3f}, a_n = {a_n:.3f}")
    print(f"  mean {z.mean():.3f} (Gumbel: {np.euler_gamma:.3f}),   "
          f"SD {z.std(ddof=1):.3f} (Gumbel: {np.pi / np.sqrt(6):.3f})")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(z, bins=60, density=True, alpha=0.6, label=f"standardized maxima, n = {n}")
    xs = np.linspace(-3, 7, 400)
    ax.plot(xs, gumbel_r.pdf(xs), "C3-", lw=2, label=r"Gumbel density $e^{-z}e^{-e^{-z}}$")
    ax.set(xlabel=r"$z = (M_n - b_n) / a_n$", ylabel="density",
           title="Gaussian block maxima against the Gumbel limit")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

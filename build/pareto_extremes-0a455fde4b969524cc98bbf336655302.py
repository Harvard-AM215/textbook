"""The maximum of n power-law samples grows like n^(1/alpha), and its mean can fail.

Companion to Chapter 8 Worked Example 2 and solution to Exercise 4. Draws M blocks of n
Pareto samples with tail P(X > x) = x^(-alpha) for several alpha and compares the median
of the block maxima with the Frechet prediction (ln 2)^(-1/alpha) n^(1/alpha). The mean
of the maxima is printed with its prediction Gamma(1 - 1/alpha) n^(1/alpha) where that
exists (alpha > 1). For alpha <= 2 the maxima have infinite variance, so the mean of M of
them has no standard error; the script shows its spread across seeds instead, next to the
spread of the median, which is finite. The last column is the median share of a block's sum taken by its
largest sample.
"""

import numpy as np
from scipy.special import gamma

N, M = 10_000, 1000
ALPHAS = (0.5, 1.0, 2.0, 3.0)


def pareto_maxima(alpha, n, M, rng):
    """M maxima of n Pareto(alpha) samples, P(X > x) = x**(-alpha) for x >= 1."""
    return (rng.pareto(alpha, size=(M, n)) + 1).max(axis=1)


def frechet_median(alpha, n):
    """Median of the maximum of n Pareto(alpha) samples, in the Frechet limit."""
    return n ** (1 / alpha) * np.log(2) ** (-1 / alpha)


def frechet_mean(alpha, n):
    """Mean of the maximum in the Frechet limit; infinite unless alpha > 1."""
    return n ** (1 / alpha) * gamma(1 - 1 / alpha) if alpha > 1 else np.inf


def median_sd(alpha, n, M, seeds=range(1, 21)):
    """Spread of the sample median of M block maxima across fresh seeds."""
    medians = [np.median(pareto_maxima(alpha, n, M, np.random.default_rng(seed=s)))
               for s in seeds]
    return np.std(medians, ddof=1)


def main():
    print(f"n = {N} samples per block, M = {M} blocks; seed 0 for every alpha, "
          f"SD(median) from 20 further seeds")
    print(f"{'alpha':>6} {'median(max)':>12} {'Frechet':>10} {'SD(median)':>11} "
          f"{'mean(max)':>12} {'Frechet':>10} {'max/sum':>8}")
    for alpha in ALPHAS:
        rng = np.random.default_rng(seed=0)
        samples = rng.pareto(alpha, size=(M, N)) + 1
        maxima = samples.max(axis=1)
        share = np.median(maxima / samples.sum(axis=1))
        print(f"{alpha:>6.1f} {np.median(maxima):>12.4g} {frechet_median(alpha, N):>10.4g} "
              f"{median_sd(alpha, N, M):>11.3g} {maxima.mean():>12.4g} "
              f"{frechet_mean(alpha, N):>10.4g} {share:>8.2f}")

    means, medians = [], []
    for seed in range(100):
        maxima = pareto_maxima(2.0, N, M, np.random.default_rng(seed=seed))
        means.append(maxima.mean())
        medians.append(np.median(maxima))
    print(f"\nalpha = 2, 100 seeds: median(max) has SD {np.std(medians, ddof=1):.1f} "
          f"and ranges {min(medians):.1f} to {max(medians):.1f};")
    print(f"                      mean(max) ranges {min(means):.0f} to {max(means):.0f}")


if __name__ == "__main__":
    main()

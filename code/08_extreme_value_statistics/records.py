"""Records in a sequence of independent draws: how many there are does not depend on the
distribution.

Companion to Chapter 8's records section and solution to Exercise 7. Counts the records
(entries larger than everything before them) in sequences of n draws from three
distributions with very different tails, and compares the mean count with the harmonic
number H_n = 1 + 1/2 + ... + 1/n, which is about ln n + 0.577. It then checks the
one-line fact behind H_n: the last of n draws is the largest with probability 1/n.
"""

import numpy as np


def count_records(x):
    """Records in x: entries exceeding every earlier one; the first always counts."""
    x = np.asarray(x)
    return 1 + int((x[1:] > np.maximum.accumulate(x)[:-1]).sum())


def harmonic(n):
    """H_n = 1 + 1/2 + ... + 1/n, the expected number of records among n draws."""
    return (1 / np.arange(1, n + 1)).sum()


def main():
    rng = np.random.default_rng(seed=0)
    n, M = 1000, 2000
    draws = {
        "Gaussian": lambda: rng.standard_normal(n),
        "exponential": lambda: rng.standard_exponential(n),
        "Pareto, alpha = 1": lambda: rng.pareto(1.0, n) + 1,
    }
    print(f"n = {n} draws per sequence, {M} sequences")
    print(f"H_n = {harmonic(n):.3f};  ln n + 0.5772 = {np.log(n) + np.euler_gamma:.3f}")
    for name, draw in draws.items():
        counts = np.array([count_records(draw()) for _ in range(M)])
        print(f"{name:>18}: {counts.mean():.3f} +/- {counts.std(ddof=1) / np.sqrt(M):.3f} "
              f"records on average")

    for n in (10, 100):
        last_is_record = np.mean(rng.standard_normal((20_000, n)).argmax(axis=1) == n - 1)
        print(f"last of {n:>3} draws is the largest: {last_is_record:.4f}  (1/n = {1 / n:.4f})")


if __name__ == "__main__":
    main()

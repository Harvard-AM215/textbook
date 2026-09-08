"""ELO ratings -> P matrix -> championship probability.

Demonstrates the exponential dependence of championship probability on
the ELO rating gap.

Solution to Chapter 4 Exercise 5.
"""

import numpy as np
import matplotlib.pyplot as plt
from simulate_4team import simulate


def logistic(x):
    return 1 / (1 + np.exp(-x))


def main():
    rng = np.random.default_rng(seed=0)
    ratings = np.arange(0, 800, 100, dtype=float)
    n = len(ratings)
    # ELO scale: a 400-rating gap typically maps to ~10x odds.
    P = np.array([[logistic((ratings[i] - ratings[j]) / 400 * np.log(10))
                   for j in range(n)] for i in range(n)])
    np.fill_diagonal(P, 0.5)

    N = 100_000
    draw = list(rng.permutation(n))
    champs = np.array([simulate(draw, P, rng) for _ in range(N)])
    p = np.bincount(champs.astype(np.intp), minlength=n) / N

    print(f"{'team':>5} {'rating':>8} {'p_champ':>10}")
    for i in range(n):
        print(f"{i:>5} {ratings[i]:>8.0f} {p[i]:>10.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    # Use semilog y so exponential appears as a line.
    ax.semilogy(ratings, np.maximum(p, 1 / N), "o-")
    ax.set(xlabel="ELO rating", ylabel="championship probability",
           title=f"{n}-team tournament: championship odds vs ELO")
    ax.grid(which="both", alpha=0.2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

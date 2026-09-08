"""Generic single-elimination tournament simulator (works for any 2^k teams).

Demonstrates the basic pipeline: random P matrix, fixed bracket, MC over
many simulated tournaments, championship probabilities + standard errors.

Companion to Chapter 4 Worked Example.
"""

import numpy as np


def random_P(n, rng):
    """Random pairwise probability matrix with P_ji = 1 - P_ij."""
    upper = rng.uniform(size=(n, n))
    P = np.triu(upper, k=1) + (1 - np.triu(upper, k=1)).T
    np.fill_diagonal(P, 0.5)
    return P


def simulate(draw, P, rng):
    """Single-elimination simulation; returns champion's index."""
    survivors = list(draw)
    while len(survivors) > 1:
        next_round = []
        for i, j in zip(survivors[::2], survivors[1::2]):
            next_round.append(i if rng.uniform() < P[i, j] else j)
        survivors = next_round
    return survivors[0]


def championship_probs(n, draw, P, N, rng):
    champs = np.array([simulate(draw, P, rng) for _ in range(N)])
    counts = np.bincount(champs.astype(np.intp), minlength=n)
    p = counts / N
    se = np.sqrt(p * (1 - p) / N)
    return p, se


def main():
    rng = np.random.default_rng(seed=0)

    for n in [4, 16]:
        N = 100_000 if n == 4 else 50_000
        P = random_P(n, rng)
        draw = list(rng.permutation(n))
        p, se = championship_probs(n, draw, P, N, rng)

        # Strength = sum of pairwise winning probabilities (rough proxy).
        strength = P.sum(axis=1) - 0.5
        order = np.argsort(-strength)

        print(f"\n=== {n}-team tournament (N={N}) ===")
        print(f"{'team':>5} {'strength':>10} {'p_champ':>10} {'+/- 2SE':>10}")
        for i in order[:min(n, 8)]:
            print(f"{i:>5} {strength[i]:>10.3f} {p[i]:>10.4f} {2*se[i]:>10.4f}")


if __name__ == "__main__":
    main()

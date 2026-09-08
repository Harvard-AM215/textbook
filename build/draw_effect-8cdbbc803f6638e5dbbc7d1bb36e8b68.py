"""Quantify how much the random draw affects a single team's win probability.

Holds the P matrix fixed; samples 100 random initial brackets; for each,
estimates the strongest team's championship probability via MC. Reports
the spread (SD across draws) relative to the mean.

Solution to Chapter 4 Exercise 4.
"""

import numpy as np
from simulate_4team import random_P, championship_probs


def main():
    rng = np.random.default_rng(seed=0)
    n = 8
    n_draws = 100
    N_per_draw = 5000

    P = random_P(n, rng)

    # Strongest team by row-sum.
    strength = P.sum(axis=1) - 0.5
    target = int(np.argmax(strength))
    print(f"Tracking team {target} (strongest, sum-strength = {strength[target]:.3f})")

    target_probs = []
    for _ in range(n_draws):
        draw = list(rng.permutation(n))
        p, _ = championship_probs(n, draw, P, N_per_draw, rng)
        target_probs.append(p[target])

    target_probs = np.array(target_probs)
    print(f"\nAcross {n_draws} draws (each MC'd with N={N_per_draw}):")
    print(f"  mean p_champ for team {target}: {target_probs.mean():.4f}")
    print(f"  SD   p_champ across draws    : {target_probs.std(ddof=1):.4f}")
    print(f"  ratio (SD / mean)            : {target_probs.std(ddof=1)/target_probs.mean():.3f}")
    print(f"  range                        : [{target_probs.min():.3f}, "
          f"{target_probs.max():.3f}]")


if __name__ == "__main__":
    main()

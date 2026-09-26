"""Compare win probabilities from a score random walk and Elo.

The team ratings stay fixed. Each simulation is one complete 100-step game.

Solution to Chapter 5 Exercise 4.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

N_STEPS = 100
B = norm.ppf(10 / 11) / 400
A = B / np.sqrt(N_STEPS)


def random_walk_probability(rating_gap):
    return norm.cdf(B * rating_gap)


def elo_probability(rating_gap):
    return 1 / (1 + 10 ** (-rating_gap / 400))


def simulate_games(rating_gap, n_games, rng):
    """Simulate complete score paths while holding the ratings fixed."""
    increments = rng.normal(A * rating_gap, 1, size=(n_games, N_STEPS))
    final_score_difference = increments.sum(axis=1)
    return np.mean(final_score_difference > 0)


def main():
    rng = np.random.default_rng(seed=1)
    n_games = 20_000
    gaps = np.array([-800, -400, -200, 0, 200, 400, 800])

    print("rating gap   simulation   random walk   Elo")
    for gap in gaps:
        simulation = simulate_games(gap, n_games, rng)
        rw = random_walk_probability(gap)
        elo = elo_probability(gap)
        print(f"{gap:>+6d}       {simulation:.3f}        {rw:.3f}      {elo:.3f}")

    grid = np.linspace(-800, 800, 401)
    plt.plot(grid, random_walk_probability(grid), label="random walk: Gaussian CDF")
    plt.plot(grid, elo_probability(grid), "--", label="Elo: logistic CDF")
    plt.xlabel("rating difference $R_A-R_B$")
    plt.ylabel("probability that A wins")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

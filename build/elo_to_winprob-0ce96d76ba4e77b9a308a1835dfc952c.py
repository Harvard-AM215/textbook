"""Vary one team's Elo rating and estimate its championship probability.

Holds the other seven ratings and the draw fixed, isolating the effect of
the focal team's rating.

Solution to Chapter 4 Exercise 5.
"""

import matplotlib.pyplot as plt
import numpy as np
from simulate_4team import simulate


def logistic(x):
    return 1 / (1 + np.exp(-x))


def main():
    rng = np.random.default_rng(seed=0)
    opponent_ratings = np.arange(0, 700, 100, dtype=float)
    focal_ratings = np.arange(0, 800, 100, dtype=float)
    n = 1 + len(opponent_ratings)
    draw = list(range(n))
    N = 100_000
    probs = []
    ses = []

    for focal_rating in focal_ratings:
        ratings = np.concatenate(([focal_rating], opponent_ratings))
        gaps = ratings[:, None] - ratings[None, :]
        P = logistic(gaps * np.log(10) / 400)
        champs = np.array([simulate(draw, P, rng) for _ in range(N)])
        p = np.mean(champs == 0)
        probs.append(p)
        ses.append(np.sqrt(p * (1 - p) / N))

    probs = np.array(probs)
    ses = np.array(ses)
    print(f"{'focal rating':>12} {'p_champ':>10} {'+/- 2SE':>10}")
    for rating, p, se in zip(focal_ratings, probs, ses):
        print(f"{rating:>12.0f} {p:>10.4f} {2 * se:>10.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(focal_ratings, probs, "o-")
    axes[0].set(xlabel="Focal-team Elo rating", ylabel="Championship probability")
    axes[1].semilogy(focal_ratings, np.maximum(probs, 1 / N), "o-")
    axes[1].set(xlabel="Focal-team Elo rating", ylabel="Championship probability")
    for ax in axes:
        ax.grid(which="both", alpha=0.2)
    fig.suptitle("Fixed opponents and draw: championship probability vs Elo")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

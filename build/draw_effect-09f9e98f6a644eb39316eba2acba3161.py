"""Quantify how much the random draw affects a single team's win probability.

Holds the P matrix fixed; samples 100 random initial brackets; for each,
estimates the strongest team's championship probability via MC. Reports
the spread (SD across draws) relative to the mean.

Solution to Chapter 4 Exercise 4.
"""

import matplotlib.pyplot as plt
import numpy as np
from simulate_4team import championship_probs, random_P


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
    target_ses = []
    for _ in range(n_draws):
        draw = list(rng.permutation(n))
        p, se = championship_probs(n, draw, P, N_per_draw, rng)
        target_probs.append(p[target])
        target_ses.append(se[target])

    target_probs = np.array(target_probs)
    target_ses = np.array(target_ses)
    across_draw_sd = target_probs.std(ddof=1)
    rms_mc_se = np.sqrt(np.mean(target_ses**2))
    print(f"\nAcross {n_draws} draws (each MC'd with N={N_per_draw}):")
    print(f"  mean p_champ for team {target}: {target_probs.mean():.4f}")
    print(f"  SD   p_champ across draws    : {across_draw_sd:.4f}")
    print(f"  RMS  Monte Carlo SE          : {rms_mc_se:.4f}")
    print(f"  ratio (draw SD / MC SE)      : {across_draw_sd / rms_mc_se:.1f}")
    print(
        f"  range                        : [{target_probs.min():.3f}, "
        f"{target_probs.max():.3f}]"
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(target_probs, bins=12, edgecolor="white")
    ax.axvline(target_probs.mean(), color="black", linestyle="--", label="mean")
    ax.set(
        xlabel=f"Championship probability for team {target}",
        ylabel="Number of draws",
        title="Variation in championship probability across random draws",
    )
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

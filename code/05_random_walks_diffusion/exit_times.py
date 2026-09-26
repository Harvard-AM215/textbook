"""Estimate how the time to leave an interval grows with its width.

Solution to Chapter 5 Exercise 3.
"""

import matplotlib.pyplot as plt
import numpy as np


def stopping_times(L, n_walks, rng):
    """Simulate until every walk has reached either -L or L."""
    positions = np.zeros(n_walks, dtype=int)
    times = np.zeros(n_walks, dtype=int)
    active = np.ones(n_walks, dtype=bool)

    while np.any(active):
        count = active.sum()
        positions[active] += rng.choice((-1, 1), size=count)
        times[active] += 1
        active = np.abs(positions) < L

    return times


def main():
    rng = np.random.default_rng(seed=42)
    n_walks = 20_000
    widths = np.array([10, 20, 40])
    means = np.array(
        [stopping_times(L, n_walks, rng).mean() for L in widths]
    )

    slope, intercept = np.polyfit(np.log(widths), np.log(means), 1)
    for L, mean in zip(widths, means):
        print(f"L = {L:2d}: mean stopping time = {mean:7.1f}  (L^2 = {L**2})")
    print(f"fitted log-log slope = {slope:.3f}  (expected: 2)")

    fitted = np.exp(intercept) * widths**slope
    plt.loglog(widths, means, "o", label="simulation")
    plt.loglog(widths, fitted, "--", label=f"fit: slope {slope:.2f}")
    plt.xlabel("boundary distance L")
    plt.ylabel("mean stopping time")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

"""Verify that MC standard error of championship probability scales as 1/sqrt(N).

Solution to Chapter 4 Exercise 3.
"""

import matplotlib.pyplot as plt
import numpy as np
from simulate_4team import random_P, simulate


def main():
    rng = np.random.default_rng(seed=0)
    n = 16
    P = random_P(n, rng)
    draw = list(rng.permutation(n))

    Ns = np.array([100, 1000, 10_000, 100_000])
    n_repeats = 30
    target = 0  # any specific team

    sds = []
    for N in Ns:
        ps = []
        for _ in range(n_repeats):
            champs = np.array([simulate(draw, P, rng) for _ in range(N)])
            ps.append((champs == target).mean())
        sds.append(np.std(ps, ddof=1))

    sds = np.array(sds)
    slope, _ = np.polyfit(np.log(Ns), np.log(sds), 1)
    print(f"{'N':>8} {'empirical SD':>15}")
    for N, sd in zip(Ns, sds):
        print(f"{N:>8} {sd:>15.5f}")
    print(f"\nFitted slope on log-log: {slope:.3f}  (theory: -0.5)")

    reference = sds[0] * (Ns / Ns[0]) ** (-0.5)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(Ns, sds, "o-", label="empirical SD")
    ax.loglog(Ns, reference, "--", label=r"$N^{-1/2}$ reference")
    ax.set(
        xlabel="Simulated tournaments per estimate, N",
        ylabel="SD across repeated estimates",
        title=f"Monte Carlo convergence (fitted slope {slope:.3f})",
    )
    ax.grid(which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

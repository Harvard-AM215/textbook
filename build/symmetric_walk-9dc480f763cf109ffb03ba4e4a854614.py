"""Simulate symmetric 1D random walks; show histograms and sqrt(t) variance.

Companion to Chapter 5 Worked Example 1.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def random_walks(n_steps, n_walkers, delta=1.0, p=0.5, rng=None):
    """Generate (n_walkers, n_steps+1) array of trajectories starting at 0."""
    rng = rng or np.random.default_rng()
    steps = rng.choice([-delta, delta], size=(n_walkers, n_steps),
                       p=[1 - p, p])
    return np.concatenate([np.zeros((n_walkers, 1)), np.cumsum(steps, axis=1)],
                          axis=1)


def main():
    rng = np.random.default_rng(seed=0)
    N = 400
    M = 5000
    delta = 1.0

    traj = random_walks(N, M, delta=delta, p=0.5, rng=rng)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) Histograms at three snapshots, with theory overlay. Bins are one per
    # reachable lattice site (width 2*delta): X_n has the parity of n, so bins that
    # ignore the lattice come out jagged. See clt_histograms.py.
    ax = axes[0]
    snapshots = [50, 150, 400]
    for n in snapshots:
        edges = np.arange(-n - 1, n + 2, 2) * delta
        ax.hist(traj[:, n], bins=edges, density=True, alpha=0.5,
                label=f"n = {n}")
        sd = delta * np.sqrt(n)
        xs = np.linspace(-4 * sd, 4 * sd, 300)
        ax.plot(xs, norm.pdf(xs, 0, sd), "k--", lw=1)
    ax.set(xlabel="position", ylabel="density",
           title="Walker positions vs theoretical Gaussian")
    ax.legend()
    ax.grid(alpha=0.2)

    # (b) Empirical SD vs n on log-log; theory line of slope 1/2.
    ax = axes[1]
    ns = np.arange(1, N + 1)
    emp_sd = traj[:, 1:].std(axis=0, ddof=1)
    theory = delta * np.sqrt(ns)
    ax.loglog(ns, emp_sd, "k.", ms=3, label="empirical SD")
    ax.loglog(ns, theory, "C3-", label=r"theory: $\delta\sqrt{n}$")
    ax.set(xlabel="step n", ylabel="SD of position",
           title="sqrt(n) scaling of walker spread")
    ax.legend()
    ax.grid(which="both", alpha=0.2)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

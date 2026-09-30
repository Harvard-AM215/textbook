"""Biased random walk: drift grows linearly, spread grows like sqrt(t).

Plots empirical mean and SD vs n on the same axes (log-log) so the two
different scaling laws are visible side by side.

Companion to Chapter 5 Worked Example 2.
"""

import matplotlib.pyplot as plt
import numpy as np


def random_walks(n_steps, n_walkers, delta=1.0, p=0.5, rng=None):
    rng = rng or np.random.default_rng()
    steps = rng.choice([-delta, delta], size=(n_walkers, n_steps),
                       p=[1 - p, p])
    return np.concatenate([np.zeros((n_walkers, 1)), np.cumsum(steps, axis=1)],
                          axis=1)


def main():
    rng = np.random.default_rng(seed=0)
    N = 1000
    M = 5000
    delta = 1.0
    p = 0.55  # slight rightward bias

    traj = random_walks(N, M, delta=delta, p=p, rng=rng)
    ns = np.arange(1, N + 1)
    emp_mean = traj[:, 1:].mean(axis=0)
    emp_sd = traj[:, 1:].std(axis=0, ddof=1)

    theory_mean = ns * delta * (2 * p - 1)
    theory_sd = np.sqrt(ns) * delta * np.sqrt(1 - (2 * p - 1) ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(ns, emp_mean, "k.", ms=2, label="empirical mean")
    ax.plot(ns, theory_mean, "C3-", label=r"$n\delta(p-q)$")
    ax.plot(ns, emp_sd, "b.", ms=2, label="empirical SD")
    ax.plot(ns, theory_sd, "C0-", label=r"$\delta\sqrt{n(1-(p-q)^2)}$")
    ax.set(xlabel="step n", ylabel="position",
           title=f"Biased walk (p = {p}): drift vs spread")
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.loglog(ns, np.abs(emp_mean), "k.", ms=3, label="|mean| (slope 1)")
    ax.loglog(ns, emp_sd, "b.", ms=3, label="SD (slope 1/2)")
    ax.set(xlabel="step n", ylabel="position",
           title="Mean grows like n; SD grows like sqrt(n)")
    ax.legend()
    ax.grid(which="both", alpha=0.2)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

"""Monte Carlo estimate of pi via dart-throwing in the unit square.

Plots both the estimate vs N (linear y-axis) and the absolute error vs N
(log-log) to visualize the 1/sqrt(N) scaling.

Companion to Chapter 3.
"""

import numpy as np
import matplotlib.pyplot as plt


def estimate_pi(N, rng):
    """Monte Carlo estimate of pi using N uniform points in [0,1]^2."""
    pts = rng.uniform(0, 1, size=(N, 2))
    inside = (pts[:, 0]**2 + pts[:, 1]**2) < 1.0
    return 4 * inside.mean()


def main():
    rng = np.random.default_rng(seed=0)

    Ns = np.logspace(1, 7, 25).astype(int)
    pi_est = np.array([estimate_pi(int(n), rng) for n in Ns])
    error = np.abs(pi_est - np.pi)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.semilogx(Ns, pi_est, "ko--", label=r"$\hat\pi_N$")
    ax.axhline(np.pi, color="C3", label=r"$\pi$")
    ax.set(xlabel="N", ylabel=r"$\hat\pi_N$",
           title="Monte Carlo estimate of pi")
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[1]
    # Theoretical SE of pi-hat: 4 * sqrt(p(1-p)/N) with p = pi/4.
    p = np.pi / 4
    theory = 4 * np.sqrt(p * (1 - p) / Ns)
    ax.loglog(Ns, error, "ko--", label=r"$|\hat\pi - \pi|$")
    ax.loglog(Ns, theory, "-", color="C3", label=r"$4\sqrt{p(1-p)/N}$")
    ax.set(xlabel="N", ylabel="absolute error",
           title="Convergence: error ~ 1/sqrt(N)")
    ax.legend()
    ax.grid(which="both", alpha=0.2)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

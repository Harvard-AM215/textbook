"""Deterministic SIR for several R0 values.

Plots S, I, R vs time and reports peak infected fraction and time-to-peak.

Companion to Chapter 10 Worked Example 1 / Exercise 4.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def sir_rhs(y, t, beta, gamma):
    S, I, R = y
    return [-beta * S * I, beta * S * I - gamma * I, gamma * I]


def main():
    N = 1_000_000
    I0 = 1.0
    gamma = 0.1
    t = np.linspace(0, 800, 4001)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    print(f"{'R0':>6} {'peak I/N':>10} {'time to peak':>14} {'final R/N':>11}")
    for ax, R0 in zip(axes, [1.5, 2.5, 4.0]):
        beta = R0 * gamma / N
        sol = odeint(sir_rhs, [N - I0, I0, 0.0], t, args=(beta, gamma))
        S, I, R = sol.T
        peak_idx = I.argmax()
        print(f"{R0:>6.2f} {I[peak_idx]/N:>10.3f} {t[peak_idx]:>14.1f} {R[-1]/N:>11.3f}")
        ax.plot(t, S / N, label="S/N")
        ax.plot(t, I / N, label="I/N")
        ax.plot(t, R / N, label="R/N")
        ax.axhline(1 / R0, ls="--", color="gray", lw=0.8,
                   label=f"peak threshold: $1/R_0$")
        ax.set(xlabel="t", title=f"$R_0 = {R0}$")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("fraction of population")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

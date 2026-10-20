"""Final-size relation:  1 - x = exp(-R0 * x).

Numerically solves for the final attack rate as a function of R0.

Solution to Chapter 10 Exercise 3.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


def final_size(R0):
    """Solve 1 - x = exp(-R0 * x) for x in (0, 1)."""
    if R0 <= 1:
        return 0.0
    f = lambda x: 1 - x - np.exp(-R0 * x)
    return brentq(f, 1e-6, 1 - 1e-6)


def main():
    R0s = np.linspace(0.5, 10.0, 250)
    sizes = np.array([final_size(R) for R in R0s])

    print(f"{'R0':>6} {'final attack rate':>20}")
    for R0 in [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]:
        x = final_size(R0)
        print(f"{R0:>6.2f} {x:>20.5f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(R0s, sizes, "-")
    ax.axvline(1, ls="--", color="gray", lw=0.8)
    ax.set(xlabel=r"$R_0$", ylabel="final attack rate $R(\\infty)/N$",
           title="Final epidemic size as a function of $R_0$")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

"""Hill's constant-speed approximation against the exact optimal race.

Companion to Chapter 9. For races much longer than the critical distance, neglecting the
opening sprint and the closing fade and the kinetic energy at the line turns the energy
balance into E0 + sigma T = (D/T)^2 T / tau, Hill's 1926 theory with Keller's dissipation,
so the square of the average speed should be a straight line in 1/T:

    (D/T)^2 ~ sigma tau + tau E0 / T.

The intercept sigma tau is exact: the long-race average speed tends to sqrt(sigma tau).
The slope tau E0 is not: the exact solution keeps a sprint of about 0.7 s and a fade, and
the kinetic energy at the line, so its 1/T coefficient differs from tau E0 by about one
percent. This script prints both coefficients, the residual of the 1972 records and of
the modern Olympic finals from the line, and plots them.

    python3 hill_asymptote.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keller import KELLER_RECORDS, KELLER_SI, D_of_T, olympic_flat, solve_race  # noqa: E402


def hill_line(T, tau, sigma, E0):
    """Hill's (D/T)^2 as a function of T."""
    return sigma * tau + tau * E0 / T


def main(params=KELLER_SI, show=True):
    tau, F, sigma, E0 = (params[k] for k in ("tau", "F", "sigma", "E0"))

    # the exact 1/T coefficient, read off at a very long race
    T_long = 1e5
    D_long, t1, t2 = solve_race(T_long, **params)
    slope_exact = ((D_long / T_long) ** 2 - sigma * tau) * T_long
    print(f"intercept sigma tau = {sigma * tau:.2f} m^2/s^2  "
          f"(sqrt = {np.sqrt(sigma * tau):.2f} m/s)")
    print(f"Hill slope tau E0 = {tau * E0:.0f} m^2/s;  exact solver at T = {T_long:.0e} s: "
          f"{slope_exact:.0f} m^2/s ({100 * (slope_exact / (tau * E0) - 1):+.1f}%), "
          f"with t1 = {t1:.2f} s and T - t2 = {T_long - t2:.2f} s")

    # residuals of the exact optimum and of the data from Hill's line, long races only
    print()
    print(f"{'race':>8} {'1/T':>8} {'(D/T)^2':>9} {'Hill':>9} {'exact':>9}")
    for label, D, T_rec in KELLER_RECORDS:
        if D < 800:
            continue
        line = hill_line(T_rec, tau, sigma, E0)
        exact = (D_of_T(T_rec, **params) / T_rec) ** 2
        print(f"{label:>8} {1 / T_rec:8.5f} {(D / T_rec) ** 2:9.2f} {line:9.2f} {exact:9.2f}")

    if show:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        rec = [(D, T) for _, D, T in KELLER_RECORDS if D >= 800]
        x = np.array([1 / T for _, T in rec])
        y = np.array([(D / T) ** 2 for D, T in rec])
        ax.plot(x, y, "o", mfc="white", mec="k", label="1972 records")
        ol = [(d, t) for _, d, t, _ in olympic_flat() if d >= 800]
        ax.plot([1 / t for _, t in ol], [(d / t) ** 2 for d, t in ol], "D", ms=4.5,
                color="C1", alpha=0.85, label="Olympic champions 2008-24")
        xx = np.linspace(0, 1.05 * x.max(), 60)
        ax.plot(xx, sigma * tau + tau * E0 * xx, color="C0", lw=1.5,
                label="Hill's line, sigma tau + tau E0 / T")
        TT = 1 / xx[1:]
        ax.plot(xx[1:], [(D_of_T(T, **params) / T) ** 2 for T in TT], color="k", lw=1,
                ls="--", label="exact optimal race")
        ax.set_xlabel("1/T (1/s)")
        ax.set_ylabel("(D/T)^2 (m^2/s^2)")
        ax.legend(frameon=False, fontsize=9)
        ax.set_title("Long races: the square of the average speed against 1/T", loc="left")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

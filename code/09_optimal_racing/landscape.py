"""The one-dimensional problem behind the solver: D(t1) for the 400 m.

Companion to Chapter 9 Worked Example 1. Once the sprint length t1 is chosen, the cruise
speed follows by continuity and the time t2 at which the store empties follows from
E(t2) = 0, so the distance run in a fixed time T is a function of t1 alone. This script
plots that function for the 400 m with Keller's constants and prints the features the
solver has to cope with: a narrow peak at small t1, a long plateau on which every choice
of t1 gives almost the same distance (the cruise speed is then within a hair of top speed),
and the branch boundary where the cruise speed drops below sqrt(sigma tau) and the store
never empties.

    python3 landscape.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keller import KELLER_SI, D_of_t1, T_of_D, Tc_of, solve_race, v_dash  # noqa: E402


def main(params=KELLER_SI, show=True):
    tau, F, sigma, E0 = (params[k] for k in ("tau", "F", "sigma", "E0"))
    T400 = T_of_D(400.0, **params)
    Tc = Tc_of(**params)
    t1 = np.linspace(1e-3, Tc - 1e-6, 4000)
    D = D_of_t1(t1, T400, **params)
    D_opt, t1_opt, t2_opt = solve_race(T400, **params)
    # where the cruise speed would equal sqrt(sigma tau): below this t1 the store never
    # empties and the runner cruises to the line
    t1_branch = -tau * np.log(1.0 - np.sqrt(sigma * tau) / (F * tau))
    plateau = D[t1 > 10.0]

    print(f"400 m with Keller's constants: T = {T400:.2f} s")
    print(f"peak: t1 = {t1_opt:.2f} s, D = {D_opt:.2f} m  (t2 = {t2_opt:.2f} s)")
    print(f"branch boundary: v1 = sqrt(sigma tau) at t1 = {t1_branch:.2f} s; "
          f"there D = {float(D_of_t1(t1_branch, T400, **params)):.2f} m")
    print(f"plateau (t1 > 10 s): D between {plateau.min():.2f} and {plateau.max():.2f} m, "
          f"i.e. {D_opt - plateau.max():.2f} m short of the peak")
    half = t1[D > D_opt - 1.0]
    print(f"within 1 m of the peak: t1 from {half.min():.2f} to {half.max():.2f} s")
    print(f"cruise speed at the peak {v_dash(t1_opt, tau, F):.2f} m/s; "
          f"at t1 = 10 s {v_dash(10.0, tau, F):.3f} m/s; top speed F tau = {F * tau:.3f} m/s")

    if show:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(t1, D, color="k", lw=1.4)
        ax.plot(t1_opt, D_opt, "o", color="C3")
        ax.axvline(t1_branch, color="C0", lw=0.8, ls="--")
        ax.annotate(f"peak at t1 = {t1_opt:.2f} s", (t1_opt, D_opt),
                    xytext=(6, D_opt - 8), color="C3",
                    arrowprops=dict(arrowstyle="-", color="C3", lw=0.8))
        ax.text(t1_branch + 0.2, D.min() + 5, "cruise speed = sqrt(sigma tau)",
                color="C0", fontsize=9)
        ax.set_xlabel("sprint length t1 (s)")
        ax.set_ylabel("distance D(t1) in T = %.2f s (m)" % T400)
        ax.set_title("The 400 m: a narrow peak beside a long plateau", loc="left")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

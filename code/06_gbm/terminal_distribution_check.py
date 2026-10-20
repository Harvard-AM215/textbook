"""Verify GBM closed-form mean/median/variance against simulation.

Solution to Chapter 6 Exercise 3. Only S_T is needed, so each draw uses the closed-form
solution in one step instead of simulating a daily path. The last column is the
Monte Carlo standard error of each statistic, measured by repeating the whole
10,000-path experiment 200 times with fresh seeds.
"""

import numpy as np

S0, MU, SIGMA, T = 100.0, 0.05, 0.30, 1.0
N_PATHS = 10_000


def draw_ST(rng, n):
    Z = rng.standard_normal(n)
    return S0 * np.exp((MU - 0.5 * SIGMA**2) * T + SIGMA * np.sqrt(T) * Z)


def stats(ST):
    return ST.mean(), np.median(ST), ST.var(ddof=1)


def main():
    emp = stats(draw_ST(np.random.default_rng(seed=0), N_PATHS))
    reps = np.array([stats(draw_ST(np.random.default_rng(seed=s), N_PATHS))
                     for s in range(1, 201)])
    se = reps.std(axis=0, ddof=1)

    th = (S0 * np.exp(MU * T),
          S0 * np.exp((MU - 0.5 * SIGMA**2) * T),
          S0**2 * np.exp(2 * MU * T) * (np.exp(SIGMA**2 * T) - 1))

    print(f"{'':12s}{'empirical':>12s}{'theory':>12s}{'rel.err':>10s}{'rel.SE':>9s}")
    for name, e, t, s in zip(["mean(S_T)", "median(S_T)", "var(S_T)"], emp, th, se):
        print(f"{name:12s}{e:12.4f}{t:12.4f}{(e - t) / t:10.2%}{s / t:9.2%}")


if __name__ == "__main__":
    main()

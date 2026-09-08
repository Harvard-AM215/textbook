"""Verification vs. validation, made concrete (Worked Example 2).

The wrinkle: it is tempting to call a simulation "validated" because its
output matches a formula. But if the formula and the simulation encode the
*same* assumptions, agreement only VERIFIES the code -- it says nothing about
whether the model matches reality. Validation requires an *external* check:
data the model was not built to reproduce.

Here we compute the distribution of *series length* (how many games a
best-of-7 lasts) two ways under the IID assumption:

* closed form (negative-binomial), and
* Monte Carlo.

They agree -- that is VERIFICATION. To VALIDATE the IID model we compare its
predicted share of 7-game series against a real-world reference value from
historical World Series outcomes.
"""

from math import comb

import numpy as np


def series_length_pmf(p, n_games=7):
    """Distribution of the number of games played in a best-of-``n_games``.

    A series that ends in ``g`` games means one team gets its ``w``-th win in
    game ``g`` while the other has ``g - w`` wins -- not ``w - 1``, which holds
    only for a full-length series. Summing over which team clinches gives the
    probability mass at each length ``g``.

    Parameters
    ----------
    p : float
        Per-game win probability for the favorite.
    n_games : int, optional
        Maximum games (odd). Default 7.

    Returns
    -------
    dict
        Mapping ``g -> P(series lasts exactly g games)``.
    """
    w = n_games // 2 + 1
    pmf = {}
    for g in range(w, n_games + 1):
        # favorite clinches in game g: (w-1) of first (g-1) are wins, then a win
        fav = comb(g - 1, w - 1) * p**w * (1 - p) ** (g - w)
        # underdog clinches in game g: symmetric with p <-> (1-p)
        dog = comb(g - 1, w - 1) * (1 - p) ** w * p ** (g - w)
        pmf[g] = fav + dog
    return pmf


def simulate_length(p, n_games, rng):
    """Return the number of games one simulated best-of-``n_games`` lasts."""
    w = n_games // 2 + 1
    a = b = 0
    g = 0
    while a < w and b < w:
        g += 1
        if rng.random() < p:
            a += 1
        else:
            b += 1
    return g


def main():
    rng = np.random.default_rng(seed=1903)  # first modern World Series
    p = 0.5  # two evenly matched teams
    n_games = 7
    n_trials = 400_000

    # --- VERIFY: closed form vs. Monte Carlo (same assumptions) ---
    pmf = series_length_pmf(p, n_games)
    lengths = np.array([simulate_length(p, n_games, rng) for _ in range(n_trials)])

    print(f"Series-length distribution for p = {p} (IID games)")
    print(f"{'games':>6} {'closed form':>12} {'monte carlo':>12}")
    for g in sorted(pmf):
        mc = np.mean(lengths == g)
        print(f"{g:>6} {pmf[g]:>12.4f} {mc:>12.4f}")
    print("=> agreement here is VERIFICATION (same IID model, two solvers).\n")

    # --- VALIDATE: model prediction vs. an external, real-world number ---
    model_7game = pmf[7]
    # 40 of the 115 best-of-7 World Series from 1905 to 2023 needed all seven
    # decisive games. Do NOT hardcode a remembered figure here: this number has
    # been wrong twice (a denominator of 119, which counts seasons rather than
    # series, and an unsourced 37% share). It is derived from committed data by
    # series_length_data.py, which asserts its parse reproduces Mosteller (1952).
    from series_length_data import lengths, load

    counts = lengths(load(), 1905, 2023)
    n_series = sum(counts.values())
    real_7game = counts[7] / n_series

    print(f"P(7-game series), IID model p=0.5 : {model_7game:.3f}")
    print(f"Historical share, {n_series} series 1905-2023 : {real_7game:.3f}")
    # Null SE on a proportion at p0 = 0.3125 with n = 115 is about 0.043, so the
    # 3.5-point gap is ~0.8 SE (exact two-sided binomial p = 0.42). Not significant.
    print(
        "=> comparing to the historical number is VALIDATION. But the gap is\n"
        "   about 0.8 standard errors, so this comparison does NOT have the\n"
        "   resolution to convict the model -- that is the chapter's point."
    )


if __name__ == "__main__":
    main()

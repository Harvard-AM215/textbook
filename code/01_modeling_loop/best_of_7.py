"""Best-of-7 series: closed form vs. Monte Carlo.

The modeling question: if the better team wins any single game with
probability ``p > 0.5``, how often does it win a best-of-7 series?

This script walks the full modeling loop for Worked Example 1:

* IMPLEMENT (closed form): the series is won by whoever reaches 4 wins first,
  so P(better team wins) is the probability of 4+ successes before 4 failures.
  Summing over the game in which the 4th win lands gives a negative-binomial
  expression that we evaluate exactly.
* IMPLEMENT (Monte Carlo): simulate the series directly under the *same*
  independence/stationarity assumptions.
* VERIFY: the two implementations must agree (MC -> closed form as N -> inf),
  and the closed form must reproduce known limiting cases (p = 0.5 -> 0.5,
  p = 1 -> 1).

Note: MC agreeing with the closed form is VERIFICATION, not validation --
both encode the same assumptions. See Worked Example 2.
"""

from math import comb

import numpy as np


def p_series_closed_form(p, n_games=7):
    """Probability the per-game favorite wins a best-of-``n_games`` series.

    The favorite wins a game with probability ``p`` (independent games).
    A best-of-``n_games`` series ends when a team reaches
    ``w = n_games // 2 + 1`` wins. Condition on the game in which the
    favorite records its ``w``-th win: that game is a win (prob ``p``), and
    exactly ``w - 1`` of the preceding ``k`` games were wins.

    Parameters
    ----------
    p : float
        Per-game win probability for the favorite, in ``[0, 1]``.
    n_games : int, optional
        Maximum games in the series (odd). Default 7.

    Returns
    -------
    float
        Probability the favorite wins the series.
    """
    w = n_games // 2 + 1  # wins needed (4 for best-of-7)
    total = 0.0
    for k in range(w - 1, n_games):  # k preceding games, then the clinching win
        # exactly (w-1) of the first k games are wins; game k+1 is the w-th win
        total += comb(k, w - 1) * p**w * (1 - p) ** (k - (w - 1))
    return total


def simulate_series(p, n_games, rng):
    """Return ``True`` if the favorite wins one simulated series.

    Parameters
    ----------
    p : float
        Per-game win probability for the favorite.
    n_games : int
        Maximum games in the series (odd).
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    bool
        Whether the favorite reached the winning number of games first.
    """
    w = n_games // 2 + 1
    wins = 0
    losses = 0
    while wins < w and losses < w:
        if rng.random() < p:
            wins += 1
        else:
            losses += 1
    return wins == w


def p_series_monte_carlo(p, n_games, n_trials, rng):
    """Monte Carlo estimate of the favorite's series-win probability."""
    outcomes = np.array(
        [simulate_series(p, n_games, rng) for _ in range(n_trials)]
    )
    phat = outcomes.mean()
    stderr = outcomes.std(ddof=1) / np.sqrt(n_trials)
    return phat, stderr


def main():
    rng = np.random.default_rng(seed=215)
    n_games = 7
    n_trials = 200_000

    print(f"{'p':>5} {'closed form':>12} {'monte carlo':>14} {'|diff|':>9}")
    for p in [0.50, 0.55, 0.60, 0.70, 1.00]:
        exact = p_series_closed_form(p, n_games)
        mc, se = p_series_monte_carlo(p, n_games, n_trials, rng)
        print(f"{p:>5.2f} {exact:>12.4f} {mc:>10.4f}+-{se:.4f} {abs(exact-mc):>9.4f}")

    # VERIFY against known limiting cases.
    assert abs(p_series_closed_form(0.5) - 0.5) < 1e-12, "p=0.5 must give 0.5"
    assert abs(p_series_closed_form(1.0) - 1.0) < 1e-12, "p=1 must give 1"
    assert abs(p_series_closed_form(0.0) - 0.0) < 1e-12, "p=0 must give 0"
    print("\nLimiting-case checks passed (p in {0, 0.5, 1}).")


if __name__ == "__main__":
    main()

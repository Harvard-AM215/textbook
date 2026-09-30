"""Delta-hedge a sold call at a finite number of dates and measure what is left over.

The Black-Scholes derivation rebalances continuously. Here the seller of an
at-the-money call receives the Black-Scholes price, holds Phi(d1) shares, keeps the rest
in a bank account at rate r, and rebalances n times a year. At expiration the account
plus the shares minus the call's payoff is the hedging error: zero for continuous
hedging with the right sigma.

Companion to Chapter 7 Worked Example 3 and the solution to Exercise 7.
"""

import numpy as np
from scipy.stats import norm


def bsm_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bsm_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def hedging_error(n_steps, mu, n_paths, rng, S0=100.0, K=100.0, T=1.0, r=0.05,
                  sigma=0.20, true_sigma=None):
    """Sell and hedge the call at sigma; the stock itself moves with true_sigma."""
    true_sigma = sigma if true_sigma is None else true_sigma
    dt = T / n_steps
    S = np.full(n_paths, S0)
    delta = bsm_delta(S, K, T, r, sigma)
    cash = bsm_call(S0, K, T, r, sigma) - delta * S
    for k in range(1, n_steps + 1):
        Z = rng.standard_normal(n_paths)
        S = S * np.exp((mu - 0.5 * true_sigma**2) * dt + true_sigma * np.sqrt(dt) * Z)
        cash = cash * np.exp(r * dt)
        if k < n_steps:
            new_delta = bsm_delta(S, K, T - k * dt, r, sigma)
            cash -= (new_delta - delta) * S
            delta = new_delta
    return cash + delta * S - np.maximum(S - K, 0)


def main():
    rng = np.random.default_rng(seed=0)
    n_paths = 20_000
    price = bsm_call(100.0, 100.0, 1.0, 0.05, 0.20)
    print(f"call price at sigma = 0.20: {price:.4f}\n")

    print("Hedging error at expiration, 20,000 paths (mean +/- its standard error, SD)")
    print(f"{'n':>5} {'mu':>6} {'mean':>8} {'SE':>7} {'SD':>7}")
    for n in (12, 52, 252):
        for mu in (0.05, 0.10, 0.30):
            e = hedging_error(n, mu, n_paths, rng)
            se = e.std(ddof=1) / np.sqrt(n_paths)
            sd = e.std(ddof=1)
            print(f"{n:>5} {mu:>6.2f} {e.mean():>+8.3f} {se:>7.3f} {sd:>7.3f}")

    print("\nExercise 7: priced and hedged at sigma = 0.20, the stock moves at 0.25")
    print(f"{'n':>5} {'mean':>8} {'SE':>7} {'SD':>7}")
    for n in (52, 252):
        e = hedging_error(n, 0.10, n_paths, rng, true_sigma=0.25)
        se = e.std(ddof=1) / np.sqrt(n_paths)
        print(f"{n:>5} {e.mean():>+8.3f} {se:>7.3f} {e.std(ddof=1):>7.3f}")
    high = bsm_call(100.0, 100.0, 1.0, 0.05, 0.25)
    low = bsm_call(100.0, 100.0, 1.0, 0.05, 0.20)
    gap = high - low
    print(f"price at 0.25 minus price at 0.20: {gap:.4f}")
    print(f"  carried to expiration          : {gap * np.exp(0.05):.4f}")

    print("\nThe same, with the stock drifting at r = 0.05 instead of 0.10 (daily)")
    e = hedging_error(252, 0.05, n_paths, rng, true_sigma=0.25)
    se = e.std(ddof=1) / np.sqrt(n_paths)
    print(f"{252:>5} {e.mean():>+8.3f} {se:>7.3f} {e.std(ddof=1):>7.3f}")


if __name__ == "__main__":
    main()

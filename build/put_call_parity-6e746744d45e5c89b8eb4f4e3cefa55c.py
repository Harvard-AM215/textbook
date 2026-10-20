"""Numerical verification of put-call parity:  C - P = S0 - K * exp(-rT).

Solution to Chapter 7 Exercise 4.
"""

import numpy as np
from scipy.stats import norm


def bsm_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def main():
    cases = [
        dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.20),
        dict(S0=80,  K=100, T=0.5, r=0.03, sigma=0.30),
        dict(S0=120, K=110, T=2.0, r=0.06, sigma=0.25),
    ]
    print(f"{'C':>10} {'P':>10} {'C-P':>10} {'S0-K e^-rT':>12} {'gap':>10}")
    for c in cases:
        C = bsm_call(**c)
        P = bs_put(**c)
        rhs = c["S0"] - c["K"] * np.exp(-c["r"] * c["T"])
        print(f"{C:>10.4f} {P:>10.4f} {C-P:>10.4f} {rhs:>12.4f} {abs(C-P-rhs):>10.2e}")


if __name__ == "__main__":
    main()

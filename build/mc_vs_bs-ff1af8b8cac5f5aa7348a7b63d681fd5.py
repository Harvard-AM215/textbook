"""European call: Monte Carlo vs Black-Scholes closed form.

Verifies the MC pricer against the BS formula for several strikes and
reports Z-scores. All Z-scores should land in [-2, +2] modulo bad luck.

Also integrates the diffusion equation's fundamental solution against the payoff, the
route of Chapter 7's "Solving the Black-Scholes equation as a diffusion problem".

Companion to Chapter 7 Worked Example 1 / Exercise 3.
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def bsm_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def mc_call(S0, K, T, r, sigma, n_paths, rng):
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.exp(-r * T) * np.maximum(ST - K, 0)
    return payoffs.mean(), payoffs.std(ddof=1) / np.sqrt(n_paths)


def kernel_call(S, K, tau, r, sigma):
    """Discounted integral: Gaussian kernel, variance sigma^2 tau, times the payoff."""
    y, var = np.log(S) + (r - 0.5 * sigma**2) * tau, sigma**2 * tau

    def integrand(yp):
        G0 = np.exp(-((y - yp) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
        return G0 * (np.exp(yp) - K)

    # Stop 12 SD above the mean: beyond it the integrand is negligible, and e^y'
    # overflows long before infinity.
    U, _ = quad(integrand, np.log(K), y + 12 * np.sqrt(var))
    return np.exp(-r * tau) * U


def main():
    rng = np.random.default_rng(seed=0)
    S0, T, r, sigma = 100.0, 1.0, 0.05, 0.30
    n_paths = 200_000

    print(f"{'K':>6} {'MC':>10} {'2-sigma':>10} {'BS':>10} {'Z':>8}")
    for K in [80, 100, 120]:
        mc_mean, mc_se = mc_call(S0, K, T, r, sigma, n_paths, rng)
        bs = bsm_call(S0, K, T, r, sigma)
        z = (mc_mean - bs) / mc_se
        print(f"{K:>6} {mc_mean:>10.4f} {2*mc_se:>10.4f} {bs:>10.4f} {z:>8.2f}")

    kernel = kernel_call(100.0, 100.0, 1.0, 0.05, 0.20)
    formula = bsm_call(100.0, 100.0, 1.0, 0.05, 0.20)
    print("\nS = K = 100, tau = 1, r = 0.05, sigma = 0.20:")
    print(f"kernel integral {kernel:.4f}, formula {formula:.4f}")


if __name__ == "__main__":
    main()

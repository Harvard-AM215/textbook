"""Monte Carlo estimate of the integral of sin(x) from 0 to pi.

True value is 2. Demonstrates: (a) the change-of-variables trick to move
the integration domain to [0, 1], and (b) computing the standard error
from the same sample as the estimate.

Solution to Chapter 3 Exercise 3.
"""

import numpy as np


def main():
    rng = np.random.default_rng(seed=42)
    N = 1_000_000

    # int_0^pi sin(x) dx = pi * E[sin(pi U)] for U ~ Uniform(0, 1).
    U = rng.uniform(0, 1, size=N)
    g = np.pi * np.sin(np.pi * U)

    I_hat = g.mean()
    se = g.std(ddof=1) / np.sqrt(N)

    print(f"N            = {N}")
    print(f"true value   = 2")
    print(f"estimate     = {I_hat:.6f}")
    print(f"std. error   = {se:.6f}")
    print(f"2-sigma band = [{I_hat - 2*se:.6f}, {I_hat + 2*se:.6f}]")
    print(f"contains 2?  = {I_hat - 2*se <= 2 <= I_hat + 2*se}")


if __name__ == "__main__":
    main()

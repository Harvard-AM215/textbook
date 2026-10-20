"""Reproduce Chapter 3's inline examples and sampling-budget arithmetic."""

import math

import numpy as np
from estimate_pi import estimate_pi


def main():
    rng = np.random.default_rng(seed=0)
    coefficient = 4 * np.sqrt((np.pi / 4) * (1 - np.pi / 4))
    for n in (10_000, 1_000_000):
        estimate = estimate_pi(n, rng)
        print(
            f"pi: N={n}, estimate={estimate}, "
            f"error={abs(estimate - np.pi):.6f}, "
            f"SE={coefficient / np.sqrt(n):.6f}"
        )
    for tolerance in (0.1, 0.01, 0.001):
        print(f"pi SE target {tolerance}: N ~ {(coefficient / tolerance) ** 2:.0f}")
    for tolerance in (1e-3, 1e-6):
        print(
            f"pi two-SE half-width {tolerance}: N ~ "
            f"{(2 * coefficient / tolerance) ** 2:.3g}"
        )

    rng = np.random.default_rng(seed=1)
    n = 100_000
    u = rng.uniform(0, 1, size=n)
    g = np.exp(-(u**2))
    estimate = g.mean()
    se = g.std(ddof=1) / np.sqrt(n)
    exact = np.sqrt(np.pi) * math.erf(1) / 2
    print(f"estimate: {estimate:.4f} +/- {2 * se:.4f}  (2 sigma band)")
    print(
        f"exact integral: {exact:.8f}; inside band: {abs(estimate - exact) <= 2 * se}"
    )

    p = 1e-6
    print(f"rare event, 10% relative SE: N={(1 - p) / (p * 0.1**2):.0f}")
    p = 0.0321
    print(
        f"outbreak SE in percentage points: "
        f"{100 * np.sqrt(p * (1 - p) / 1_000_000):.4f}"
    )


if __name__ == "__main__":
    main()

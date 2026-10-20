"""Fit SIR rates to independent simulated daily infection counts.

Companion to Chapter 10 Worked example 2 and Exercise 5. The observation
model assumes known population size and initial infectious count, complete
reporting, and independent Poisson daily counts conditional on the SIR curve.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


def sir_rhs(y, t, beta, gamma):
    S, I, R = y
    return [-beta * S * I, beta * S * I - gamma * I, gamma * I]


def expected_cases(rates, time_grid, N, I0):
    transmission, gamma = rates  # transmission = beta * N
    sol = odeint(
        sir_rhs, [N - I0, I0, 0], time_grid, args=(transmission / N, gamma)
    )
    return np.maximum(sol[:-1, 0] - sol[1:, 0], 1e-12)


def fit(observed, days, day_count, N, I0):
    grid = days[: day_count + 1]
    counts = observed[:day_count]

    def nll(rates):
        predicted = expected_cases(rates, grid, N, I0)
        return np.sum(predicted - counts * np.log(predicted))

    result = minimize(
        nll,
        [0.3, 0.1],
        method="L-BFGS-B",
        bounds=[(1e-4, 2.0), (1e-4, 2.0)],
    )
    if not result.success:
        raise RuntimeError(result.message)
    transmission, gamma = result.x
    return transmission, gamma, transmission / gamma


def main():
    N, I0 = 100_000, 5
    days = np.arange(101)
    true_rates = (0.35, 0.15)
    rng = np.random.default_rng(0)
    observed = rng.poisson(expected_cases(true_rates, days, N, I0))

    for label, rates in [
        ("true", (*true_rates, true_rates[0] / true_rates[1])),
        ("first 25 days", fit(observed, days, 25, N, I0)),
        ("all 100 days", fit(observed, days, 100, N, I0)),
    ]:
        print(
            f"{label}: beta*N={rates[0]:.3f}, "
            f"gamma={rates[1]:.3f}, R0={rates[2]:.3f}"
        )


if __name__ == "__main__":
    main()

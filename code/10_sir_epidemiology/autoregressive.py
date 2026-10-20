"""Autoregressive forecasts of an SIR epidemic: linear vs exponential.

Treats the cumulative case count C(t) = I(t) + R(t) from the deterministic
SIR of Worked example 1 as if it were observed data, fits a linear and an
exponential model on a 10-day window, and forecasts the next 10 days.
Two windows: early (exponential phase) and around the peak.

Companion to Chapter 10, "Statistical models: autoregressive forecasts".
"""

import numpy as np
from scipy.integrate import odeint


def sir_rhs(y, t, beta, gamma):
    S, I, R = y
    return [-beta * S * I, beta * S * I - gamma * I, gamma * I]


def forecast(days, C, fit_days, horizon):
    """Fit both models on fit_days, forecast the next `horizon` days."""
    t_fit = days[fit_days]
    y_fit = C[fit_days]
    b1, b0 = np.polyfit(t_fit, y_fit, 1)                # linear: C = b0 + b1 t
    e1, e0 = np.polyfit(t_fit, np.log(y_fit), 1)        # exponential: log C = e0 + e1 t
    t_fc = days[fit_days[-1] + 1: fit_days[-1] + 1 + horizon]
    lin = b0 + b1 * t_fc
    exp = np.exp(e0 + e1 * t_fc)
    truth = C[fit_days[-1] + 1: fit_days[-1] + 1 + horizon]
    return t_fc, truth, lin, exp, b1, e1


def main():
    N = 1_000_000
    I0 = 1.0
    gamma = 0.1
    beta = 0.3 / N            # R0 = 3
    days = np.arange(0, 121)
    sol = odeint(sir_rhs, [N - I0, I0, 0.0], days, args=(beta, gamma))
    S, I, R = sol.T
    C = I + R

    for label, start in [("early (days 20-29)", 20), ("near peak (days 65-74)", 65)]:
        fit_days = np.arange(start, start + 10)
        t_fc, truth, lin, exp, b1, e1 = forecast(days, C, fit_days, 10)
        print(f"{label}: fitted daily increment {b1:.0f}, "
              f"fitted daily growth factor {np.exp(e1):.3f}")
        print(f"{'day':>5} {'true C':>10} {'linear':>10} {'exponential':>12}")
        for k in [0, 4, 9]:
            print(f"{t_fc[k]:>5.0f} {truth[k]:>10.0f} {lin[k]:>10.0f} {exp[k]:>12.0f}")
        print()


if __name__ == "__main__":
    main()

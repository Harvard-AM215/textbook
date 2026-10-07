"""Fit a generalized extreme value distribution to block maxima, with standard errors.

Companion to Chapter 8's fitting section and solution to Exercise 5. `fit_gev` returns
(xi, mu, sigma) and their standard errors from the inverse of the observed information,
the Hessian of the negative log-likelihood, as in Chapter 2; SciPy's `genextreme.fit`
gives the estimates but no errors. `return_level` is the level exceeded once per T
blocks on average.

Run as a script, it draws n block maxima from a GEV with xi = 0.15, mu = 50, sigma = 10,
fits them, and repeats 100 times for each n, to show how precisely xi comes back from
30 blocks and from 200: the first fit on its own, the typical standard error, the actual spread of the estimates,
how often the truth lies within two standard errors of the estimate, and how often the
interval also includes zero, so that the data cannot rule out the Gumbel class.
"""

import numpy as np
from scipy.stats import genextreme

TRUE_XI, TRUE_MU, TRUE_SIGMA = 0.15, 50.0, 10.0
SIZES = (30, 50, 100, 200, 500)
REFITS = 100


def negative_log_likelihood(params, m):
    """-log L for block maxima m; SciPy's shape parameter c is -xi."""
    xi, mu, sigma = params
    if sigma <= 0:
        return np.inf
    logpdf = genextreme.logpdf(m, -xi, loc=mu, scale=sigma)
    return -logpdf.sum() if np.all(np.isfinite(logpdf)) else np.inf


def observed_information(params, m, rel_step=1e-4):
    """Hessian of the negative log-likelihood at params, by central differences."""
    h = rel_step * np.maximum(1.0, np.abs(params))
    H = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.zeros(3), np.zeros(3)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = (negative_log_likelihood(params + ei + ej, m)
                       - negative_log_likelihood(params + ei - ej, m)
                       - negative_log_likelihood(params - ei + ej, m)
                       + negative_log_likelihood(params - ei - ej, m)) / (4 * h[i] * h[j])
    return H


def fit_gev(m):
    """Maximum-likelihood (xi, mu, sigma) for block maxima m, with standard errors.

    The optimizer starts from the Gumbel (xi = 0) at the sample's mean and standard
    deviation; SciPy's own starting point can send a small sample's fit astray.
    """
    m = np.asarray(m, dtype=float)
    c, mu, sigma = genextreme.fit(m, 0.0, loc=m.mean(), scale=m.std(ddof=1))
    params = np.array([-c, mu, sigma])
    try:
        se = np.sqrt(np.diag(np.linalg.inv(observed_information(params, m))))
    except np.linalg.LinAlgError:
        se = np.full(3, np.nan)
    return params, se


def return_level(T, xi, mu, sigma):
    """The level exceeded once per T blocks on average: the 1 - 1/T quantile."""
    y = -np.log(1 - 1 / T)
    if abs(xi) < 1e-9:
        return mu - sigma * np.log(y)
    return mu + sigma * (y ** (-xi) - 1) / xi


def refit_shapes(n, refits=REFITS, seed=0):
    """Fitted shapes and their standard errors over `refits` fresh samples of n maxima."""
    rng = np.random.default_rng(seed=seed)
    xis, ses = [], []
    for _ in range(refits):
        m = genextreme.rvs(-TRUE_XI, loc=TRUE_MU, scale=TRUE_SIGMA, size=n, random_state=rng)
        (xi, _, _), se = fit_gev(m)
        xis.append(xi)
        ses.append(se[0])
    return np.array(xis), np.array(ses)


def main():
    print(f"true xi = {TRUE_XI}, mu = {TRUE_MU}, sigma = {TRUE_SIGMA}; {REFITS} refits per n, "
          "seed 0 for each n")
    print(f"{'n':>5} {'first fit':>16} {'mean xi_hat':>12} {'SD of xi_hat':>13} {'mean SE':>8} "
          f"{'truth within 2 SE':>18} {'0 within 2 SE':>14}")
    for n in SIZES:
        xis, ses = refit_shapes(n)
        first = f"{xis[0]:+.3f} +/- {ses[0]:.3f}"
        covered = np.mean(np.abs(xis - TRUE_XI) <= 2 * ses)
        gumbel_allowed = np.mean(np.abs(xis) <= 2 * ses)
        print(f"{n:>5} {first:>16} {xis.mean():>12.3f} {xis.std(ddof=1):>13.3f} "
              f"{np.nanmean(ses):>8.3f} {covered:>18.0%} {gumbel_allowed:>14.0%}")


if __name__ == "__main__":
    main()

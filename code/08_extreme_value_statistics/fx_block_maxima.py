"""Block maxima of the daily USD/EUR move: a GEV fit, a return level with an interval,
and what a Gaussian predicts for the same quantities.

Companion to Chapter 8 Worked Example 3. Reads the committed public-domain file
data/public/fx_usd_eur.csv; the chapter's cell fetches the same file over https. The
quantity is the size of the daily log return in percent, and the blocks are calendar
months. Prints the fit with its standard errors (fit_gev.py), bootstrap intervals for xi and for
the ten-year return level, one resampling single months and one resampling whole years,
then the Gaussian's version of each number, and plots
the return-level curves against the observed maxima at their empirical return periods.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genextreme, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_gev import fit_gev, return_level  # noqa: E402

FX_FILE = Path(__file__).resolve().parents[2] / "data" / "public" / "fx_usd_eur.csv"
MIN_OBS = 15        # a calendar month with fewer trading days is a partial month: dropped
T_YEARS = 10        # the return period quoted in the chapter, in years
N_BOOT = 500


def log_returns(prices):
    """Daily log returns in percent."""
    return 100 * np.log(prices).diff().dropna()


def block_maxima(x, freq="M", min_obs=MIN_OBS):
    """One maximum per calendar block, dropping blocks with fewer than min_obs entries."""
    groups = x.groupby(x.index.to_period(freq))
    return groups.max()[groups.size() >= min_obs]


def bootstrap(m, T, n_boot, rng, blocks=None):
    """Refit resampled maxima n_boot times; returns the refitted shapes and T-block levels.

    With blocks=None each maximum is drawn on its own, with replacement, which assumes the
    maxima are independent. With `blocks` (an array of block labels, one per maximum, such
    as the calendar year) whole blocks are drawn with replacement instead, so that maxima
    which tend to come together, as volatile months do, stay together: a block bootstrap.
    """
    m = np.asarray(m, dtype=float)
    groups = None if blocks is None else [m[blocks == b] for b in np.unique(blocks)]
    xis, levels = [], []
    for _ in range(n_boot):
        if groups is None:
            sample = rng.choice(m, size=m.size, replace=True)
        else:
            picks = rng.integers(len(groups), size=len(groups))
            sample = np.concatenate([groups[i] for i in picks])
        c, mu, sigma = genextreme.fit(sample, 0.0, loc=sample.mean(), scale=sample.std(ddof=1))
        xis.append(-c)
        levels.append(return_level(T, -c, mu, sigma))
    return np.array(xis), np.array(levels)


def lag_correlation(m, lag=1):
    """Pearson correlation between each block maximum and the one `lag` blocks later."""
    m = np.asarray(m, dtype=float)
    return np.corrcoef(m[:-lag], m[lag:])[0, 1]


def simulated_gaussian_fit(sizes, sd, seed):
    """GEV fit to the block maxima of Gaussian days with the observed block sizes."""
    rng = np.random.default_rng(seed=seed)
    simulated = np.array([np.abs(rng.standard_normal(k) * sd).max() for k in sizes])
    return fit_gev(simulated)


def gaussian_block_level(T, block_days, sd):
    """T-block return level of the largest |move| in blocks of block_days Gaussian days.

    P(all block_days moves are within z) = (2 Phi(z / sd) - 1)^block_days; set it to
    1 - 1/T and solve for z.
    """
    p = (1 - 1 / T) ** (1 / block_days)
    return sd * norm.ppf((1 + p) / 2)


def main():
    prices = pd.read_csv(FX_FILE, index_col=0, parse_dates=True)["Close"]
    r = log_returns(prices)
    moves = r.abs()
    sd = r.std(ddof=0)
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    days_per_year = r.size / years

    monthly = block_maxima(moves)
    sizes = moves.groupby(moves.index.to_period("M")).size()[monthly.index]
    T = 12 * T_YEARS
    (xi, mu, sigma), se = fit_gev(monthly.values)
    z_T = return_level(T, xi, mu, sigma)
    z_all = return_level(monthly.size, xi, mu, sigma)

    print(f"{r.size} daily returns, {r.index[0].date()} to {r.index[-1].date()}, "
          f"{years:.1f} years; daily SD {sd:.3f}%")
    print(f"{monthly.size} monthly maxima ({sizes.min()} to {sizes.max()} trading days each); "
          f"median {monthly.median():.2f}%, largest {monthly.max():.2f}% in {monthly.idxmax()}")
    print(f"GEV fit : xi = {xi:+.3f} +/- {se[0]:.3f}   mu = {mu:.3f} +/- {se[1]:.3f}   "
          f"sigma = {sigma:.3f} +/- {se[2]:.3f}")
    print(f"{T_YEARS}-year level (once per {T} months): {z_T:.2f}%;   "
          f"{monthly.size}-month level: {z_all:.2f}%")

    # The likelihood's standard errors assume independent maxima. They are not independent:
    # a volatile month tends to follow a volatile month. Two bootstraps, one that ignores
    # this and one that resamples whole calendar years so that it is kept.
    print(f"lag-1 correlation of the monthly maxima: {lag_correlation(monthly.values):.2f}")
    xis, levels = bootstrap(monthly.values, T, N_BOOT, np.random.default_rng(seed=0))
    print(f"single-month bootstrap, {N_BOOT} refits, 90%: xi in "
          f"[{np.percentile(xis, 5):+.3f}, {np.percentile(xis, 95):+.3f}], "
          f"{T_YEARS}-year level in [{np.percentile(levels, 5):.2f}, {np.percentile(levels, 95):.2f}]%")
    years_of = np.array([per.year for per in monthly.index])
    xis_b, levels_b = bootstrap(monthly.values, T, N_BOOT, np.random.default_rng(seed=0),
                                blocks=years_of)
    xi_sd_b = xis_b.std(ddof=1)
    print(f"whole-year block bootstrap, {N_BOOT} refits, 90%: xi in "
          f"[{np.percentile(xis_b, 5):+.3f}, {np.percentile(xis_b, 95):+.3f}] (SD {xi_sd_b:.3f}), "
          f"{T_YEARS}-year level in [{np.percentile(levels_b, 5):.2f}, {np.percentile(levels_b, 95):.2f}]%")

    # The Gaussian's version of each number, with the same daily standard deviation.
    z_T_gauss = gaussian_block_level(T, round(sizes.mean()), sd)
    expected_above = (1 - (2 * norm.cdf(2.0 / sd) - 1) ** sizes).sum()
    observed_above = int((monthly > 2.0).sum())
    z_big = moves.max() / sd
    once_per_years = 1 / (2 * norm.sf(z_big)) / days_per_year
    print(f"Gaussian, same daily SD: {T_YEARS}-year level {z_T_gauss:.2f}%; "
          f"months with a move above 2%: {expected_above:.1f} expected, {observed_above} observed")
    print(f"largest move {moves.max():.2f}% = {z_big:.1f} daily SDs; a Gaussian gives one such "
          f"day per {once_per_years:.1e} years")

    # If the days were Gaussian, what shape would the monthly maxima fit to? One simulated
    # history (seed 0) in full, then the spread of the fitted shape over forty more.
    (xi_g, mu_g, sigma_g), se_g = simulated_gaussian_fit(sizes, sd, seed=0)
    print(f"GEV fit to simulated Gaussian months: xi = {xi_g:+.3f} +/- {se_g[0]:.3f}, "
          f"{T_YEARS}-year level {return_level(T, xi_g, mu_g, sigma_g):.2f}%")
    shapes = np.array([simulated_gaussian_fit(sizes, sd, seed=s)[0][0] for s in range(1, 41)])
    print(f"forty more simulated histories: xi averages {shapes.mean():+.3f}, SD {shapes.std(ddof=1):.3f}, "
          f"range {shapes.min():+.3f} to {shapes.max():+.3f}")
    print(f"observed minus simulated shape: {xi - xi_g:.3f}; combined spread with the "
          f"likelihood SE {np.hypot(se[0], se_g[0]):.3f} (ratio {(xi - xi_g) / np.hypot(se[0], se_g[0]):.1f}), "
          f"with the block-bootstrap SD {np.hypot(xi_sd_b, se_g[0]):.3f} "
          f"(ratio {(xi - xi_g) / np.hypot(xi_sd_b, se_g[0]):.1f})")

    # Return-level plot: fitted curves against the observed maxima at their empirical
    # return periods, (N + 1) / rank.
    ordered = np.sort(monthly.values)[::-1]
    periods = (monthly.size + 1) / np.arange(1, monthly.size + 1)
    Ts = np.logspace(np.log10(1.05), 3, 300)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(periods, ordered, "k.", ms=4, label="observed monthly maxima")
    ax.semilogx(Ts, [return_level(t, xi, mu, sigma) for t in Ts], "C0-", lw=2,
                label=rf"GEV fit, $\xi = {xi:+.2f}$")
    ax.semilogx(Ts, [return_level(t, 0.0, *genextreme.fit(monthly.values, 0.0, loc=mu,
                                                          scale=sigma, f0=0.0)[1:])
                     for t in Ts], "C0--", lw=1.5, label="Gumbel fit")
    ax.semilogx(Ts, [gaussian_block_level(t, round(sizes.mean()), sd) for t in Ts],
                "C3-", lw=2, label="Gaussian days, same SD")
    ax.set(xlabel="return period (months)", ylabel="largest daily move in the month (%)",
           title="USD/EUR: return levels of the monthly maximum")
    ax.legend()
    ax.grid(which="both", alpha=0.2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

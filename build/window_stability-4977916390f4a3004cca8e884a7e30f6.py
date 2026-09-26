"""Rolling five-year GBM fits, with the standard error of a single window for scale.

Solution to Chapter 6 Exercise 5. Prints the range of each estimate and plots both as
time series, with the two-standard-error band a single five-year window would have if
mu and sigma were constant. Uses the book's loader: the US stock market from 1926 where
the network allows, otherwise the USD/EUR exchange rate. Pass --fx to use the committed
USD/EUR file.

Windows are five calendar years, not a fixed number of returns: before May 1952 the US
market also traded on Saturdays, so five years then held about 1,450 returns rather than
1,260. Each window's time step is its span divided by its number of returns.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from equity_data import price_series  # noqa: E402

FX_FILE = Path(__file__).resolve().parents[2] / "data" / "public" / "fx_usd_eur.csv"


def load_prices():
    """The book's loader, or the committed USD/EUR file with --fx."""
    if "--fx" in sys.argv:
        prices = pd.read_csv(FX_FILE, index_col=0, parse_dates=True)["Close"]
        return prices, "USD/EUR (committed file)"
    return price_series()


def main():
    prices, source = load_prices()
    log_r = np.log(prices).diff().dropna()

    years_per_window = 5
    roll = log_r.rolling(f"{round(365.25 * years_per_window)}D")
    n = roll.count()
    dt = years_per_window / n                   # average spacing inside each window
    rolling_sigma = roll.std(ddof=0) / np.sqrt(dt)
    rolling_mu = roll.mean() / dt + rolling_sigma**2 / 2

    # Keep only complete windows.
    complete = log_r.index >= log_r.index[0] + pd.DateOffset(years=years_per_window)
    rolling_mu, rolling_sigma, n = rolling_mu[complete], rolling_sigma[complete], n[complete]

    # Full-sample fit, with time measured from the dates.
    years = (log_r.index[-1] - log_r.index[0]).days / 365.25
    dt_all = years / log_r.size
    sigma_all = log_r.std(ddof=0) / np.sqrt(dt_all)
    mu_all = log_r.mean() / dt_all + sigma_all**2 / 2

    # Standard errors of one window's estimates, at the full-sample volatility. The
    # sigma SE is given twice: for Gaussian returns, and corrected for the data's
    # excess kurtosis k (Var of the sample variance is sigma^4 (k + 2) / n).
    k = kurtosis(log_r)
    n_typ = n.median()
    se_mu = sigma_all / np.sqrt(years_per_window)
    se_sigma_gauss = sigma_all / np.sqrt(2 * n_typ)
    se_sigma_fat = sigma_all * np.sqrt((k + 2) / (4 * n_typ))

    print(f"source = {source}")
    print(f"{len(rolling_mu)} windows of {years_per_window} years, "
          f"median {n_typ:.0f} returns each; full-sample sigma_hat = {sigma_all:.4f}")
    print(f"mu_hat    ranges {rolling_mu.min():+.3f} to {rolling_mu.max():+.3f}"
          f"   (one window's SE: {se_mu:.3f})")
    print(f"sigma_hat ranges {rolling_sigma.min():.3f} to {rolling_sigma.max():.3f}"
          f"   (one window's SE: {se_sigma_gauss:.4f} if Gaussian, "
          f"{se_sigma_fat:.4f} at excess kurtosis {k:.1f})")
    print(f"largest sigma_hat in the window ending {rolling_sigma.idxmax().date()}, "
          f"smallest in the window ending {rolling_sigma.idxmin().date()}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax = axes[0]
    ax.plot(rolling_mu.index, rolling_mu.values, "C0-")
    ax.axhline(0, color="k", lw=0.5)
    for y in (mu_all - 2 * se_mu, mu_all + 2 * se_mu):
        ax.axhline(y, color="C0", ls="--", lw=1)
    ax.set(ylabel=r"$\hat\mu$ (annualized)",
           title=r"5-year rolling drift; dashed: $\pm 2$ SE if $\mu$ constant")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(rolling_sigma.index, rolling_sigma.values, "C3-")
    for y in (sigma_all - 2 * se_sigma_fat, sigma_all + 2 * se_sigma_fat):
        ax.axhline(y, color="C3", ls="--", lw=1)
    ax.set(xlabel="window end date", ylabel=r"$\hat\sigma$ (annualized)",
           title=r"5-year rolling volatility; dashed: $\pm 2$ SE if $\sigma$ constant")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

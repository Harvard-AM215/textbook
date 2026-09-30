"""Fit GBM to a daily price series and check the Gaussian assumption.

Companion to Chapter 6 Worked Example 2 and solution to Exercise 4. Uses the book's
loader: the US stock market from 1926 where the network allows, otherwise the USD/EUR
exchange rate. Pass --fx to use the committed USD/EUR file, which is what the in-browser
kernel gets. To fit a single stock instead, replace `prices` with its daily closes.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm

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
    r = np.log(prices).diff().dropna().values   # daily log returns
    # Time from the dates, not 252 returns a year: before May 1952 the US market also
    # traded on Saturdays, and 1/252 would stretch that century to 104 years.
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    dt = years / r.size

    # Maximum likelihood: variance with denominator n, and mu = m/dt + sigma^2/2.
    sigma_hat = r.std() / np.sqrt(dt)
    mu_hat = r.mean() / dt + sigma_hat**2 / 2
    se_mu = sigma_hat / np.sqrt(years)
    z = (r - r.mean()) / r.std()

    print(f"source      = {source}")
    print(f"{r.size} daily returns over {years:.1f} years")
    print(f"mean(r)/dt  = {r.mean() / dt:.4f}  (estimates mu - sigma^2/2)")
    print(f"mu_hat      = {mu_hat:.4f} +/- {se_mu:.4f}")
    print(f"sigma_hat   = {sigma_hat:.4f}")
    print(f"beyond 4 SD : {(np.abs(z) > 4).sum()}  (Gaussian: {r.size * 2 * norm.sf(4):.1f})")
    print(f"excess kurtosis of daily log returns = {kurtosis(r):.1f}  (Gaussian: 0)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) Histogram of standardized log returns against N(0, 1), log density axis so
    # the tails are visible.
    ax = axes[0]
    ax.hist(z, bins=80, density=True, alpha=0.6, label="data")
    xs = np.linspace(-6, 6, 300)
    ax.plot(xs, norm.pdf(xs), "C3-", lw=2, label=r"$\mathcal{N}(0, 1)$")
    ax.set(xlabel="standardized log return", ylabel="density (log scale)",
           title="Standardized daily log returns", yscale="log", xlim=(-6, 6))
    ax.legend()
    ax.grid(which="both", alpha=0.2)

    # (b) Q-Q plot: data quantiles against Gaussian quantiles.
    ax = axes[1]
    z_sorted = np.sort(z)
    p = (np.arange(z_sorted.size) + 0.5) / z_sorted.size
    theory_q = norm.ppf(p)
    ax.plot(theory_q, z_sorted, "k.", ms=3, alpha=0.5)
    lo, hi = theory_q.min(), theory_q.max()
    ax.plot([lo, hi], [lo, hi], "C3-", lw=2, label="y = x (Gaussian)")
    ax.set(xlabel="Gaussian quantile", ylabel="data quantile",
           title="Q-Q plot against the Gaussian")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

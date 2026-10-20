"""Is GBM's volatility constant? Fit sigma separately in each calendar year.

Companion to Chapter 6 Worked Example 3. Compares the scatter of the yearly estimates
with the two-standard-error band that sampling noise alone would allow if sigma were
constant: once with the Gaussian formula SE = sigma / sqrt(2N), and once widened for
the measured excess kurtosis k of the daily log returns,
SE = sigma * sqrt((k + 2) / (4N)). Pass --fx to use the committed USD/EUR file.

Time is measured from the dates. A calendar year's N returns span one year, so that
year's spacing is 1/N; before May 1952 the US market also traded on Saturdays and N was
280 to 300, so a fixed 1/252 would understate those years' sigma by 5 to 8%.
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
    r = np.log(prices).diff().dropna()
    years = (r.index[-1] - prices.index[0]).days / 365.25
    sigma_all = r.std(ddof=0) / np.sqrt(years / r.size)
    k = kurtosis(r)

    by_year = r.groupby(r.index.year)
    n = by_year.size()
    full = n >= 200                            # drop the partial first and last years
    sigma_year = (by_year.std(ddof=0) * np.sqrt(n))[full]   # spacing 1/n in year units
    drift_year = (by_year.mean() * n)[full]

    band = 2 * sigma_all / np.sqrt(2 * n[full])
    band_k = 2 * sigma_all * np.sqrt((k + 2) / (4 * n[full]))
    dev = abs(sigma_year - sigma_all)

    print(f"source: {source}")
    print(f"full years: {len(sigma_year)}; "
          f"excess kurtosis of daily log returns: {k:.1f}")
    print(f"yearly sigma_hat: {sigma_year.min():.3f} ({sigma_year.idxmin()}) "
          f"to {sigma_year.max():.3f} ({sigma_year.idxmax()})")
    print(f"Gaussian band  {sigma_all:.3f} +/- {band.mean():.3f}: "
          f"{int((dev > band).sum())} years outside")
    print(f"fat-tail band  {sigma_all:.3f} +/- {band_k.mean():.3f}: "
          f"{int((dev > band_k).sum())} years outside")
    print(f"SD of yearly m/dt: {drift_year.std(ddof=1):.3f}  "
          f"(noise alone: {sigma_all:.3f})")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sigma_year.index, sigma_year.values, "C3o-", ms=3,
            label=r"yearly $\hat\sigma$")
    ax.fill_between(sigma_year.index, sigma_all - band_k, sigma_all + band_k,
                    color="C0", alpha=0.3,
                    label=r"$\pm 2$ SE if $\sigma$ were constant")
    ax.set(xlabel="year", ylabel=r"$\hat\sigma$ (annualized)",
           title="Volatility estimated one calendar year at a time")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

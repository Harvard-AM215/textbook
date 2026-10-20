"""Pull a real options chain (default: SPY); plot implied-vol smile.

Solution to Chapter 7 Exercise 5.
"""

from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm


def bsm_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol_call(C_market, S0, K, T, r):
    def f(sigma):
        return bsm_call(S0, K, T, r, sigma) - C_market

    try:
        return brentq(f, 1e-4, 5.0)
    except ValueError:
        return np.nan


def main(ticker_symbol="SPY", r=0.04):
    ticker = yf.Ticker(ticker_symbol)
    # The first expiration at least two weeks out: an expiration today gives T = 0,
    # and with a day or two left the implied vol is very sensitive to quote noise.
    today = date.today()
    expiries = [e for e in ticker.options
                if (date.fromisoformat(e) - today).days >= 14]
    if not expiries:
        raise SystemExit(f"No options at least two weeks out for {ticker_symbol}")
    expiry = expiries[0]

    chain = ticker.option_chain(expiry)
    calls = chain.calls.copy()
    S0 = ticker.history(period="1d")["Close"].iloc[-1]
    T = (date.fromisoformat(expiry) - today).days / 365

    calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)].copy()
    calls["mid"] = 0.5 * (calls["bid"] + calls["ask"])
    calls["iv"] = [implied_vol_call(m, S0, K, T, r)
                   for m, K in zip(calls["mid"], calls["strike"])]

    keep = calls.dropna(subset=["iv"])
    print(f"Ticker: {ticker_symbol}  S0={S0:.2f}  expiry={expiry}  T={T:.4f}y")
    print(f"Strikes used: {len(keep)}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(keep["strike"], keep["iv"], "o-")
    ax.axvline(S0, color="k", ls="--", lw=0.7, label=f"$S_0$ = {S0:.2f}")
    ax.set(xlabel="strike K", ylabel="implied volatility",
           title=f"{ticker_symbol} call implied-vol smile (expiry {expiry})")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

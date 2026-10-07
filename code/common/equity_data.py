"""Load a financial price series, adapting to wherever the code is running.

The short version for readers: `price_series()` returns something usable wherever you are,
and degrades rather than erroring when the equity source is unreachable. It is not
unconditionally exception-free -- if the committed fallback file is *also* unreachable (no
network at all, or the site is down) the underlying HTTP error propagates, because at that
point there is nothing honest left to return. What you get depends on the environment, and the
function tells you which it gave you.

    Colab / local Python   real US equity market data, 1926-present (Ken French)
    Browser (Pyodide)      USD/EUR exchange rate, 1999-present (FRED, public domain)

WHY THE BROWSER GETS SOMETHING DIFFERENT
----------------------------------------
Not a technical limitation -- a licensing one. A stock index is a commercial product, not a
public fact: S&P Dow Jones Indices, NASDAQ and CBOE compute, brand and sell their index
data. So a free source may let you *download* equity data while forbidding you to
*republish* it, and those are different permissions. Checked 2026-08-17:

    FRED SP500, DJIA      "Reproduction ... prohibited except with prior written permission"
    FRED NASDAQCOM        copyright NASDAQ OMX Group
    Yahoo (via yfinance)  terms restrict redistribution
    Ken French library    freely downloadable, but a copyright line and no grant

This textbook therefore commits none of it. That is why there is no equity CSV to load: we
are not permitted to serve one. Downloading it yourself is fine -- that is what every reader
of the French library does -- so the code below fetches at runtime instead.

The browser kernel cannot do that fetch. Pyodide runs inside the page's origin and the
browser blocks cross-origin requests to dartmouth.edu, so the download fails there no matter
what. Rather than error, we fall back to the exchange-rate series, which is US government
data, genuinely public domain, and therefore something we *can* commit and serve.

The fallback is not a consolation prize: FX log-returns are fat-tailed and
volatility-clustered too, so every point these chapters make still lands. The equity series
just makes it more starkly -- excess kurtosis around 17 against about 2.5 for USD/EUR.

**To get the real equity data, open the chapter in Colab** using the badge at the top of the
page. Colab has a full Python and an unrestricted network, so the fetch succeeds there.

Source, when the fetch runs: Kenneth R. French, Data Library, Tuck School of Business,
Dartmouth College, built from CRSP. Cite it if you use it.
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile

import pandas as pd

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FACTORS_DAILY = "F-F_Research_Data_Factors_daily_CSV.zip"
INDUSTRY_DAILY = "10_Industry_Portfolios_daily_CSV.zip"
FX_URL = "https://harvard-am215.github.io/textbook/data/public/fx_usd_eur.csv"
FX_BASKET_URL = "https://harvard-am215.github.io/textbook/data/public/fx_basket.csv"


def environment() -> str:
    """One of 'browser', 'colab', 'local'.

    `sys.platform == "emscripten"` is the canonical Pyodide check and is what the browser
    kernel reports; the module probe is a belt-and-braces second signal.
    """
    if "google.colab" in sys.modules:
        return "colab"
    if sys.platform == "emscripten" or "pyodide" in sys.modules:
        return "browser"
    return "local"


def can_fetch_external() -> bool:
    """Whether a cross-origin download can succeed here. False in the browser kernel."""
    return environment() != "browser"


def _read_csv_url(url: str, **kw) -> pd.DataFrame:
    """Read a CSV over http, by whichever route this environment supports."""
    if environment() == "browser":
        from pyodide.http import open_url  # noqa: PLC0415 -- browser-only import

        return pd.read_csv(open_url(url), **kw)
    return pd.read_csv(url, **kw)


def _french_table(name: str) -> pd.DataFrame:
    """First table out of a French CSV zip.

    These are not plain CSVs: notes come first, and some files stack two tables
    (value-weighted, then equal-weighted) separated by a blank line and a fresh header.
    Stopping at the first break after data begins gives the value-weighted one.
    """
    with urllib.request.urlopen(FRENCH + name, timeout=120) as r:
        blob = r.read()
    z = zipfile.ZipFile(io.BytesIO(blob))
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()

    header = start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(","):
            header, start = [c.strip() for c in line.split(",")], i + 1
            break
    if header is None:
        raise ValueError("no header row in French CSV")

    rows = []
    for line in lines[start:]:
        tok = line.split(",")[0].strip()
        if not tok.isdigit():
            if rows:
                break
            continue
        rows.append([tok] + [float(p.strip()) for p in line.split(",")[1:]])

    df = pd.DataFrame(rows, columns=["Date"] + header[1:])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    return df.set_index("Date")


def market_returns() -> pd.DataFrame:
    """Daily US market return and risk-free rate, 1926-present, as decimals.

    Raises if the network is unavailable -- use `price_series()` if you want the
    environment-aware version that cannot fail.
    """
    df = _french_table(FACTORS_DAILY) / 100.0
    df = df.rename(columns={"Mkt-RF": "MktRF"})
    df["Mkt"] = df["MktRF"] + df["RF"]
    return df


def industry_portfolios() -> pd.DataFrame:
    """Daily returns for 10 industry portfolios, as decimals. Raises without network."""
    return _french_table(INDUSTRY_DAILY) / 100.0


def price_series(quiet: bool = False) -> tuple[pd.Series, str]:
    """A daily price-like series, whatever this environment can reach.

    Falls back to the committed FX file when the equity fetch fails, so an unreachable
    Dartmouth is not an error. Raises only if the fallback is unreachable too.

    Returns ``(series, label)``. The series is named ``Close`` either way, so downstream
    code -- log-returns, a GBM fit, a kurtosis check -- is identical for both.
    """
    if can_fetch_external():
        try:
            mkt = market_returns()["Mkt"]
            series = (100.0 * (1.0 + mkt).cumprod()).rename("Close")
            label = "US equity market, 1926-present (Ken French / CRSP)"
            if not quiet:
                print(f"Loaded {label}.")
            return series, label
        except Exception as exc:  # network refused, offline, source moved
            if not quiet:
                print(f"Could not reach the equity source ({type(exc).__name__}); "
                      "falling back to the committed exchange-rate series.")

    df = _read_csv_url(FX_URL, index_col=0, parse_dates=True)
    series = df["Close"].rename("Close")
    label = "USD/EUR exchange rate, 1999-present (FRED, public domain)"
    if not quiet:
        if environment() == "browser":
            print(
                f"Loaded {label}.\n"
                "The browser kernel cannot fetch the equity data -- that is a licensing\n"
                "constraint, not a bug: we are not permitted to serve an equity CSV, and\n"
                "the browser cannot reach the original source. Open this chapter in Colab\n"
                "(badge at the top of the page) to work with real US market data instead.\n"
                "Everything below works either way; the FX series is fat-tailed too."
            )
        else:
            print(f"Loaded {label}.")
    return series, label


def asset_returns(quiet: bool = False) -> tuple[pd.DataFrame, str]:
    """Daily returns for several assets, for portfolio work.

    Falls back to the committed currency basket when the equity fetch fails. Raises only
    if that fallback is unreachable too.

    Returns ``(frame, label)``: ten US industry portfolios where the network allows,
    otherwise five USD exchange-rate pairs from the committed public-domain file. Both are
    plain daily returns with a DatetimeIndex and one column per asset, so a covariance
    matrix, a frontier or a max-Sharpe solve is written once and works with either.

    Worth knowing which you have, because they illustrate diversification differently: the
    ten industries are all positively correlated (roughly 0.55-0.89), as equities inside one
    market are, while the currency pairs span negative to positive correlation. The FX
    frontier therefore shows a *more* dramatic diversification benefit than the equity one --
    the opposite of the usual expectation, and worth a sentence if you use it in class.
    """
    if can_fetch_external():
        try:
            df = industry_portfolios()
            label = "10 US industry portfolios (Ken French / CRSP)"
            if not quiet:
                print(f"Loaded {label}.")
            return df, label
        except Exception as exc:
            if not quiet:
                print(f"Could not reach the equity source ({type(exc).__name__}); "
                      "falling back to the committed currency basket.")

    px = _read_csv_url(FX_BASKET_URL, index_col=0, parse_dates=True)
    df = px.apply(pd.to_numeric, errors="coerce").pct_change().dropna()
    label = "5 USD currency pairs (FRED, public domain)"
    if not quiet:
        if environment() == "browser":
            print(
                f"Loaded {label}.\n"
                "The browser kernel cannot fetch the equity portfolios -- a licensing\n"
                "constraint, not a bug. Open this chapter in Colab (badge at the top) to\n"
                "build the frontier from real US industry portfolios instead."
            )
        else:
            print(f"Loaded {label}.")
    return df, label


if __name__ == "__main__":
    import numpy as np

    print(f"environment: {environment()}  external fetch: {can_fetch_external()}")
    s, lab = price_series()
    r = np.log(s).diff().dropna()
    print(f"  {len(s)} obs, {s.index.min().date()}..{s.index.max().date()}")
    print(f"  ann. vol {r.std() * np.sqrt(252):.2%}   excess kurtosis {r.kurtosis():.1f}")

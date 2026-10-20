---
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 8: Extreme Value Statistics

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harvard-AM215/textbook/blob/gh-pages/notebooks/08_extreme_value_statistics.ipynb)
*Runs in the browser -- click the power icon above to start the kernel -- or in Colab, which has a full Python and a live network.*

> The largest of $n$ independent draws has a distribution of its own, the single-draw distribution raised to the $n$-th power; as $n$ grows it settles into one of only three shapes, chosen by how fast the tail of the single-draw distribution falls off, so a model fitted to typical values says little about the worst case, and the worst case has to be fitted from the largest values in the data themselves.

:::{note} Running the code
The Python cells on this page run **in your browser**. Click the **power icon** at the top of the page to activate the kernel, then run — or edit — any cell. Packages (NumPy, SciPy, …) load automatically the first time you import them (a few seconds).
:::

## Motivation

A sewer is not designed for the average day of rain but for the heaviest day it will ever have to carry. A beam holds every load it meets until the largest one. A bank that survives ordinary trading days can fail on the one day the market moves eight times its typical daily move, as the dollar price of a euro did in March 2009 (Worked Example 3). Each of these is a question about the *largest* of many draws, not about their average. The central limit theorem describes the average and says nothing about the largest draw: two distributions with the same mean and variance can have maxima that differ many times over.

Earlier chapters met the problem in data. Chapter 5's Worked Example 3 found that an exchange rate makes far more large daily moves than a Gaussian allows, and Chapter 7 found the options market charging for those moves through the volatility smile. Both chapters recorded these **fat tails** and deferred them. This chapter models them. It derives the distribution of the maximum, and finds that as the number of draws grows this distribution takes one of only three shapes, and that which shape appears depends only on how fast the tail of the underlying distribution falls off. That last fact is what makes extremes modelable: you do not need the whole distribution, only the class of its tail.

The worked examples follow the modeling loop. Example 1 **verifies** the growth law of the Gaussian maximum on simulated data. Example 2 adds a power-law tail, where the maximum grows as a power of $n$ and the average of the maxima has no ordinary error bar. Example 3 fits real monthly maxima, puts an interval on a ten-year level, and **validates** the Gaussian's prediction for the same quantity against the data.

## Setup & notation

Let $X_1, \dots, X_n$ be independent draws from one distribution with CDF $F(x) = \mathbb{P}(X \le x)$ and density $f$. The **tail** is $1 - F(x) = \mathbb{P}(X > x)$. Worked Example 3 returns to data that are dependent or change over time.

New symbols in this chapter:

- $M_n = \max(X_1, \dots, X_n)$: the **block maximum**, the largest of $n$ draws. A **block** is a group of $n$ consecutive draws, such as the trading days of one month or the days of one year. Minima need no separate theory, since $\min X_i = -\max(-X_i)$.
- $b_n$, $a_n$: the **centering** and **scale** of the maximum, chosen so that $(M_n - b_n)/a_n$ settles to a fixed distribution as $n$ grows, as the central limit theorem standardizes a mean.
- $G$: the limiting distribution of the standardized maximum.
- $\alpha$: the **tail index** of a power-law tail, $\mathbb{P}(X > x) \propto x^{-\alpha}$. Smaller $\alpha$ means a heavier tail.
- $\mu$, $\sigma$, $\xi$: the **location**, **scale** and **shape** of the generalized extreme value distribution fitted to block maxima. Here $\sigma$ is a scale in the units of the data and $\mu$ a location, not the volatility and drift of Chapters 6 and 7.
- $\phi$, $\Phi$: the standard Gaussian density and CDF, as in Chapter 7.

## Core ideas

### The maximum and the tail rule

The maximum of $n$ draws is at most $x$ exactly when every draw is at most $x$. Independence therefore gives
$$
\begin{aligned}
\mathbb{P}(M_n\le x)
&=\mathbb{P}(X_1\le x,\ldots,X_n\le x) \\
&=\mathbb{P}(X_1\le x)\cdots\mathbb{P}(X_n\le x) \\
&=F(x)^n.
\end{aligned}
$$

Now let $p=1-F(x)$ be the chance that one draw exceeds $x$. The expected number of exceedances among $n$ draws is $np$, and
$$
\mathbb{P}(M_n\le x)=(1-p)^n\approx e^{-np}
$$
when $p$ is small. If $np\gg1$, at least one exceedance is very likely. If $np\ll1$, an exceedance is unlikely. The transition occurs near $np=1$. Thus a typical maximum $m_n$ satisfies
$$
\boxed{\ n\bigl[1-F(m_n)\bigr]\approx1.\ }
$$
In words: **the typical maximum is near the level exceeded by one draw in $n$.** This one-exceedance rule gives the main growth laws.

### Deriving the growth laws

**Exponential tail.** If $1-F(x)=e^{-x}$, then
$$
n e^{-m_n}=1
\quad\Longrightarrow\quad
m_n=\log n.
$$
The exact distribution also shows the size of the fluctuations. Set $x=\log n+z$:
$$
\mathbb{P}(M_n-\log n\le z)
=\left(1-\frac{e^{-z}}{n}\right)^n
\longrightarrow \exp(-e^{-z}).
$$
This limit is the **Gumbel** distribution. Its standard deviation is $\pi/\sqrt{6}\approx1.28$, so the maximum is $\log n$ plus a fluctuation of roughly constant size.

**Gaussian tail.** For a standard Gaussian,
$$
1-\Phi(x)\approx\frac{\phi(x)}{x}
=\frac{e^{-x^2/2}}{x\sqrt{2\pi}}.
$$
Substituting this into the tail rule and taking logarithms gives
$$
\frac{m_n^2}{2}+\log m_n+\frac12\log(2\pi)\approx\log n.
$$
The first term grows fastest, so to leading order
$$
m_n\approx\sqrt{2\log n}.
$$
Exercise 2 derives the next correction. It matters for the sample sizes used in practice.

**Power-law tail.** If $1-F(x)=x^{-\alpha}$ for $x\ge1$, then
$$
n m_n^{-\alpha}=1
\quad\Longrightarrow\quad
m_n=n^{1/\alpha}.
$$
Here the maximum grows as a power of $n$, much faster than a Gaussian maximum. Scaling by this growth gives
$$
\mathbb{P}\!\left(\frac{M_n}{n^{1/\alpha}}\le z\right)
=\left(1-\frac{z^{-\alpha}}{n}\right)^n
\longrightarrow \exp(-z^{-\alpha}),
$$
for $z>0$. This is the **Fréchet** distribution. For $\alpha=2$, the maximum grows like $\sqrt n$; for $\alpha=1$, it grows like $n$.

**Bounded tail.** Suppose the distribution ends at $x^\ast$ and, near that endpoint,
$$
1-F(x^\ast-y)\approx C y^\alpha.
$$
Applying the tail rule to the gap $y_n=x^\ast-m_n$ gives
$$
nC y_n^\alpha\approx1
\quad\Longrightarrow\quad
x^\ast-m_n\approx(Cn)^{-1/\alpha}.
$$
The maximum approaches the endpoint instead of growing without bound. The standardized maximum has the **Weibull** extreme-value limit.

These rates are the main modeling result:

| tail | typical maximum |
|---|---|
| bounded above | approaches the endpoint |
| exponential-like | grows like $\log n$ |
| Gaussian | grows like $\sqrt{2\log n}$ |
| power law $x^{-\alpha}$ | grows like $n^{1/\alpha}$ |

The differences are large. A Gaussian maximum reaches only about four standard deviations in ten thousand draws, while a power-law maximum can be many times larger.

### Three limits and the GEV model

The examples above illustrate a general theorem: after suitable centering and scaling, a nontrivial limit for the maximum must be Gumbel, Fréchet, or Weibull. The tail selects the limit:

| tail of $F$ | limit | examples |
|---|---|---|
| bounded above | Weibull | uniform; quantities with a hard upper limit |
| unbounded and thinner than any power law | Gumbel | exponential, Gaussian, lognormal, gamma |
| power law | Fréchet | Pareto, Student's $t$, Cauchy |

For fitting data, the three limits are written as one **generalized extreme value** (GEV) distribution:
$$
G(z;\mu,\sigma,\xi)
=\exp\!\left[-\left(1+\xi\frac{z-\mu}{\sigma}\right)^{-1/\xi}\right],
\qquad
1+\xi\frac{z-\mu}{\sigma}>0.
$$
Here $\mu$ shifts the distribution and $\sigma>0$ sets its scale. The shape $\xi$ selects the tail:

- $\xi<0$: Weibull, with an upper endpoint;
- $\xi=0$: Gumbel, understood as the limit as $\xi\to0$;
- $\xi>0$: Fréchet, with power-law tail index $\alpha=1/\xi$.

The shape $\xi$ controls how quickly return levels grow, so it is the key parameter for extrapolation. It is also hard to estimate from a small number of maxima.

One caution matters for finite samples. A Gaussian is in the Gumbel class, but it approaches the limit slowly. Over blocks of only a few dozen draws, Gaussian maxima can look bounded and a GEV fit can return $\hat\xi<0$. Worked Example 3 shows this effect. **Always interpret $\hat\xi$ together with its uncertainty.**

### Records

A **record** occurs when a draw exceeds every earlier draw. Among the first $k$ i.i.d. draws, each is equally likely to be the largest. The chance that draw $k$ sets a record is therefore $1/k$. Adding these probabilities gives
$$
\mathbb{E}[\text{records in } n \text{ draws}] = 1 + \tfrac12 + \tfrac13 + \dots + \tfrac1n = H_n \approx \log n + 0.577 .
$$
The distribution does not appear in this calculation. Thus Gaussian and power-law samples set the same number of records on average, even though their record values grow at very different rates. For example, $1000$ i.i.d. draws produce about $H_{1000}=7.49$ records. A handful of records is therefore not evidence of a trend by itself (Exercise 7 and the [companion script](./code/08_extreme_value_statistics/records.py)).


## Worked example 1: the maximum of $n$ Gaussian samples

How well does the leading law $m_n\approx\sqrt{2\log n}$ describe Gaussian maxima at practical sample sizes? For each $n$, the cell simulates $2000$ independent blocks and records their maxima.

```{code-cell} python
import numpy as np

rng = np.random.default_rng(0)
M = 2000
for n in (10, 100, 1000, 10_000):
    maxima = rng.standard_normal((M, n)).max(axis=1)
    print(f"n = {n:>6}:  mean of max = {maxima.mean():.3f}   SD of max = {maxima.std(ddof=1):.3f}"
          f"   sqrt(2 ln n) = {np.sqrt(2 * np.log(n)):.3f}")

# n =     10:  mean of max = 1.535   SD of max = 0.566   sqrt(2 ln n) = 2.146
# n =    100:  mean of max = 2.515   SD of max = 0.430   sqrt(2 ln n) = 3.035
# n =   1000:  mean of max = 3.229   SD of max = 0.348   sqrt(2 ln n) = 3.717
# n =  10000:  mean of max = 3.857   SD of max = 0.304   sqrt(2 ln n) = 4.292
```

The maximum grows slowly: increasing $n$ from $10$ to $10{,}000$ raises its mean from $1.54$ to only $3.86$. Its standard deviation shrinks from $0.57$ to $0.30$. The leading law captures the slow growth but overshoots each mean by about half a unit. Exercise 2 derives the correction.

We can also compare the simulation with an exact finite-$n$ calculation. Since the maximum has CDF $\Phi(x)^n$, its density is
$$
\frac{d}{dx}\Phi(x)^n=n\phi(x)\Phi(x)^{n-1}.
$$
Integrating $x$ against this density gives exact means of $1.539$, $2.508$, $3.241$, and $3.852$. The simulated means differ by at most $0.013$.

The [companion script](./code/08_extreme_value_statistics/gaussian_extremes.py) extends the calculation to $n=100{,}000$. It finds a simulated mean of $4.379$, an exact mean of $4.384$, and a leading-order prediction of $4.799$. After the correction from Exercise 2, the standardized maxima are close to Gumbel but not yet at the limit. This is **verification** of the Gaussian model, not evidence that real data are Gaussian.

## Worked example 2: the maximum of $n$ power-law samples

Now take a Pareto tail,
$$
\mathbb{P}(X>x)=x^{-2},\qquad x\ge1.
$$
For $n=10{,}000$, the natural scale of the maximum is $n^{1/2}=100$. Moreover,
$$
\frac{M_n}{100}\longrightarrow Z,
\qquad
\mathbb{P}(Z\le z)=e^{-z^{-2}}.
$$
Setting this CDF to $1/2$ gives $\operatorname{median}(Z)=(\log2)^{-1/2}=1.201$, so the predicted median maximum is $120.1$. The limiting mean is $\mathbb{E}[Z]=\sqrt\pi=1.772$, giving a predicted mean of $177.2$. The cell compares both predictions with $1000$ simulated blocks.

```{code-cell} python
import numpy as np

rng = np.random.default_rng(0)
n, M, alpha = 10_000, 1000, 2.0
samples = rng.pareto(alpha, size=(M, n)) + 1          # P(X > x) = x**(-alpha) for x >= 1
maxima = samples.max(axis=1)
scale = n ** (1 / alpha)
print(f"median of max : {np.median(maxima):6.1f}   (Frechet: {scale * np.log(2) ** (-1 / alpha):.1f})")
print(f"mean of max   : {maxima.mean():6.1f}   (Frechet: {scale * np.sqrt(np.pi):.1f})")
print(f"largest max   : {maxima.max():6.0f}")

# median of max :  115.8   (Frechet: 120.1)
# mean of max   :  167.6   (Frechet: 177.2)
# largest max   :   3565
```

The median, $115.8$, is close to the prediction $120.1$. The mean is less stable. Although the Fréchet mean is finite for $\alpha=2$, its variance is infinite, so the sample mean has no ordinary $1/\sqrt M$ error bar. Across one hundred seeds, the [companion script](./code/08_extreme_value_statistics/pareto_extremes.py) finds means from $157$ to $202$, while the medians stay between $114$ and $126$.

The instability comes from a few very large maxima. Here the largest is $3565$, about thirty times the median, and by itself raises the mean by $3.4$. **For a tail this heavy, the median is the more stable summary.** Exercise 4 explores other values of $\alpha$.

The contrast with the Gaussian is large. For $10{,}000$ Gaussian draws, the median maximum is about four standard deviations above the mean. A Pareto distribution with $\alpha=3$, which has finite variance, gives a median maximum about $26$ standard deviations above its mean. The tail, not the center of the distribution, controls this difference.

## Worked example 3: the largest daily move of an exchange rate

How large a daily move in the dollar price of a euro should occur once in ten years? We use the daily series from Chapters 5 and 6 and measure the absolute daily log return in percent. Each calendar month is one block. Months contain between $18$ and $23$ trading days; we ignore this small difference in block size.

```{code-cell} python
import sys
import numpy as np
import pandas as pd
from scipy.stats import genextreme

url = "https://harvard-am215.github.io/textbook/data/public/fx_usd_eur.csv"
if sys.platform == "emscripten":                           # the in-browser kernel cannot
    from pyodide.http import open_url                      # open a URL directly
    url = open_url(url)
prices = pd.read_csv(url, index_col="Date", parse_dates=["Date"])["Close"]
moves = (100 * np.log(prices).diff().dropna()).abs()        # size of the daily move, in %
months = moves.groupby(moves.index.to_period("M"))
monthly = months.max()[months.size() >= 15]                # drop the partial last month

c, mu, sigma = genextreme.fit(monthly.values)
xi = -c                                                    # SciPy's shape parameter is -xi
z_120 = genextreme.isf(1 / 120, c, loc=mu, scale=sigma)    # exceeded once per 120 months

print(f"{monthly.size} monthly maxima; median {monthly.median():.2f}%, largest {monthly.max():.2f}% ({monthly.idxmax()})")
print(f"xi = {xi:+.3f}   mu = {mu:.3f}%   sigma = {sigma:.3f}%")
print(f"10-year level = {z_120:.2f}%")

# 331 monthly maxima; median 1.19%, largest 4.62% (2009-03)
# xi = +0.029   mu = 1.024%   sigma = 0.397%
# 10-year level = 3.06%
```

The fitted shape is $\hat\xi=0.029$, close to the Gumbel value $0$. Successive monthly maxima have correlation $0.49$, so an independence-based error bar would be too small.

The [companion script](./code/08_extreme_value_statistics/fx_block_maxima.py) therefore uses a **bootstrap**. It creates many artificial histories by drawing years at random from the observed record; a year may be chosen more than once, while another may be left out. It keeps all months from each chosen year together, then refits the GEV to each artificial history. The spread of these fits measures the uncertainty. The resulting $90\%$ interval for $\xi$ is $-0.062$ to $0.107$, which does not distinguish a Gumbel tail from a mild Fréchet tail.

A ten-year return level $z_{120}$ satisfies
$$
1-G(z_{120})=\frac{1}{120}.
$$
The fitted value is $3.06\%$, with a $90\%$ bootstrap interval from $2.61\%$ to $3.55\%$. The interval is wide because only a few observations determine the tail shape.

For comparison, the Gaussian model from Chapter 6 has
$$
\sigma_{\mathrm{day}}=100\frac{0.0912}{\sqrt{251}}=0.576\%.
$$
For $21$ Gaussian days, the ten-year level solves
$$
\left[2\Phi\!\left(\frac{z}{0.576}\right)-1\right]^{21}
=1-\frac1{120},
$$
which gives $z=2.04\%$, well below the GEV estimate.

Out of $331$ months, the Gaussian predicts only $3.5$ with a move above $2\%$; the data contain $23$. The largest move, $4.62\%$, is eight daily standard deviations and is essentially impossible under the Gaussian model. This is **validation**: the Gaussian underpredicts all three measures of the extremes.

One subtlety remains: with blocks of only $21$ days, simulated Gaussian data give a fitted shape near $-0.12$, not $0$. The observed value $+0.029$ still separates the data from that finite-block Gaussian benchmark, but it does not identify the precise tail class.

The return level is a forecast only if the future resembles 1999 to 2026. **Under that assumption, the ten-year daily move is about $3.1\%$, with a $90\%$ interval from $2.61\%$ to $3.55\%$, compared with $2.0\%$ under the Gaussian model.**

## Intuition

> **Why are there only three limiting shapes?** The maximum of $nk$ draws is also the maximum of $k$ block maxima. A limiting distribution must therefore keep the same shape when we take maxima again, apart from a shift and rescaling. Only the Gumbel, Fréchet, and Weibull families have this property.

> **Why does the Gaussian maximum grow so slowly?** Its tail falls roughly like $e^{-x^2/2}$. Setting this equal to $1/n$ gives $x\approx\sqrt{2\log n}$. Each factor-of-ten increase in $n$ therefore adds less than the previous one.

> **Why does the shape parameter matter most?** The location $\mu$ shifts the distribution and the scale $\sigma$ stretches it. The shape $\xi$ controls how return levels grow as the return period increases. Small uncertainty in $\xi$ can therefore produce large uncertainty in a hundred- or thousand-year level.

> **Why fit block maxima instead of all daily values?** Extreme-value theory supplies a model for the block maxima, not for the full daily distribution. Short-range dependence between days can still give a GEV limit, although it reduces the effective block size. The cost is data: $6920$ daily values become only $331$ monthly maxima.

> **Why can't thirty blocks determine the shape?** The largest few maxima carry most of the information about $\xi$. In Exercise 5, a sample of $30$ gives a standard error near $0.16$ for a true shape of $0.15$. Most such fits cannot even determine the sign of $\xi$.

## Connections

- **Builds on:** CDFs and the central limit theorem (Appendix A), maximum likelihood (Chapter 2), Monte Carlo error bars (Chapter 3), fat tails and volatility clustering (Chapters 5 and 6), and the volatility smile (Chapter 7).
- **Used in:** Appendix B, where Gaussian tail assumptions affect portfolio risk.
- **Further directions:** peaks over a threshold and the generalized Pareto distribution; the Hill estimator for $\alpha$; record statistics; and models of athletic records.

## Exercises

1. **Conceptual.** A friend says, "The average of many draws is Gaussian, so the maximum should be Gaussian too." Explain the error. What is true instead about the limiting shape, centering, and scale of a maximum?

2. **Derivation.** For standard Gaussian draws, use $1-\Phi(x)\approx\phi(x)/x$ to show that
$$
b_n = \sqrt{2\log n} - \frac{\log\log n + \log 4\pi}{2\sqrt{2\log n}}, \qquad a_n = \frac{1}{\sqrt{2\log n}}
$$
make $\mathbb{P}(M_n\le b_n+a_nz)$ tend to $\exp(-e^{-z})$. Start from $n\phi(b_n)/b_n=1$, then expand $\phi(b_n+a_nz)$ to first order in the exponent.

3. **Computational.** Extend Worked Example 1 to $n=100{,}000$, drawing in chunks to limit memory use. Compare the simulated mean and standard deviation with the exact expectation $\int x\,n\phi(x)\Phi(x)^{n-1}\,dx$ and with $\sqrt{2\log n}$. For the largest $n$, standardize with Exercise 2's constants and compare the histogram with the Gumbel density.

4. **Computational.** Repeat Worked Example 2 for $\alpha\in\{0.5,1,2\}$ and $n=10{,}000$. Compare each median maximum with $(\log2)^{-1/\alpha}n^{1/\alpha}$, and find the median fraction of the block sum contributed by its largest draw. Explain why the mean maximum has no ordinary standard error for $\alpha\le2$, and why the draws have no finite mean for $\alpha\le1$.

5. **Computational.** Draw $200$ maxima from a GEV with $\xi=0.15$, $\mu=50$, and $\sigma=10$. (`scipy.stats.genextreme` uses shape $-\xi$.) Fit a GEV and estimate the standard error of $\hat\xi$ from the Hessian, as in Chapter 2. Repeat $100$ times. Does the reported standard error match the spread of $\hat\xi$? How often does $\hat\xi\pm2\,\mathrm{SE}$ include zero? Repeat with only $30$ maxima.

6. **Modeling judgment.** A Gumbel fit to fifty annual rainfall maxima gives $\hat\mu=110$ mm and $\hat\sigma=20$ mm, so the hundred-year level is $202$ mm. A GEV fit gives $\hat\xi=0.15\pm0.10$, with nearly the same location and scale. Holding $\mu$ and $\sigma$ fixed, compute the hundred- and thousand-year levels at $\xi=0.15$ and at $\hat\xi\pm2\,\mathrm{SE}$. What should guide the design, and how could the interval be narrowed?

7. **Computational.** Count records in sequences of $1000$ draws from three distributions and compare the mean with $H_{1000}$. Then use the $1/k$ rule to find the mean and standard deviation of the record count among $40$ independent annual maxima. Use $\operatorname{Var}(R)=\sum_k(1/k)(1-1/k)$. Are six record years evidence of a trend?

:::{admonition} Solutions
:class: dropdown

**1.** The central limit theorem concerns sums, while a maximum depends on the tail. After suitable centering and scaling, a maximum approaches a Gumbel, Fréchet, or Weibull distribution. The tail determines which one. Exponential maxima are centered by $\log n$; Gaussian maxima by about $\sqrt{2\log n}$; power-law maxima are scaled by $n^{1/\alpha}$. A Gaussian fit to the center of the data does not determine the extremes.

**2.** For a small Gaussian tail probability,
$$
\begin{aligned}
\mathbb{P}(M_n\le b_n+a_nz)
&=\Phi(b_n+a_nz)^n \\
&\approx\exp\!\left[-n\bigl(1-\Phi(b_n+a_nz)\bigr)\right].
\end{aligned}
$$
Using $1-\Phi(x)\approx\phi(x)/x$ and choosing $a_n=1/b_n$ gives
$$
n\bigl(1-\Phi(b_n+a_nz)\bigr)
\approx \frac{n\phi(b_n)}{b_n}e^{-z}.
$$
Set $n\phi(b_n)/b_n=1$. The probability then tends to $\exp(-e^{-z})$.

Taking logarithms of the condition for $b_n$ gives
$$
\log n=\frac{b_n^2}{2}+\log b_n+\frac12\log(2\pi).
$$
The leading solution is $\sqrt{2\log n}$. Substituting it into the smaller terms gives
$$
b_n=\sqrt{2\log n}-\frac{\log\log n+\log4\pi}{2\sqrt{2\log n}},
\qquad
a_n=\frac1{\sqrt{2\log n}}.
$$
At $n=10^5$, the correction lowers $b_n$ from $4.80$ to $4.28$. Adding the Gumbel mean, $0.577a_n=0.12$, predicts an expected maximum of $4.40$; the exact value is $4.38$.

**3.** With $10{,}000$ blocks, the simulated means are $1.539$, $2.508$, $3.241$, $3.854$, and $4.379$. The exact means are $1.539$, $2.508$, $3.241$, $3.852$, and $4.384$. The standard deviation falls from $0.587$ to $0.273$, while $\sqrt{2\log n}$ ranges from $2.146$ to $4.799$ and consistently overshoots.

At $n=10^5$, the standardized maxima have mean $0.475$ and standard deviation $1.311$; the Gumbel values are $0.577$ and $1.283$. The histogram is close to Gumbel, but the finite-$n$ correction still matters. See the [companion script](./code/08_extreme_value_statistics/gaussian_extremes.py).

**4.** The [companion script](./code/08_extreme_value_statistics/pareto_extremes.py) gives:

| $\alpha$ | simulated median | predicted median | median largest-share |
|---:|---:|---:|---:|
| $0.5$ | $1.80\times10^8$ | $2.08\times10^8$ | $0.61$ |
| $1$ | $1.34\times10^4$ | $1.44\times10^4$ | $0.13$ |
| $2$ | $115.8$ | $120.1$ | $0.01$ |

For $1<\alpha\le2$, the maxima have a finite mean but infinite variance, so their sample mean has no ordinary $1/\sqrt M$ error bar. For $\alpha\le1$, even the draws have no finite mean, so their sample average does not settle. Medians are more stable for these heavy tails.

**5.** One sample of $200$ maxima gives $\hat\xi=0.147\pm0.060$; one sample of $30$ gives $0.153\pm0.148$. Across $100$ repetitions:

| number of maxima | spread of $\hat\xi$ | mean reported SE | intervals including $0$ |
|---:|---:|---:|---:|
| $200$ | $0.068$ | $0.056$ | $29\%$ |
| $30$ | $0.162$ | $0.164$ | $90\%$ |

The reported standard error is close to the observed spread. With only $30$ maxima, however, nine fits in ten cannot determine whether $\xi$ is positive. The [companion script](./code/08_extreme_value_statistics/fit_gev.py) starts the optimizer at $\xi=0$ because SciPy's default starting point can fail for small samples.

**6.** Let $y_T=-\log(1-1/T)$. The GEV return level is
$$
z_T=\mu+\frac{\sigma}{\xi}\left(y_T^{-\xi}-1\right),
$$
with the Gumbel limit used at $\xi=0$. The results are:

| $\xi$ | hundred-year level | thousand-year level |
|---:|---:|---:|
| $-0.05$ | $192$ mm | $227$ mm |
| $0$ (Gumbel) | $202$ mm | $248$ mm |
| $0.15$ | $243$ mm | $352$ mm |
| $0.35$ | $339$ mm | $694$ mm |

The uncertainty grows sharply with the return period because $\xi$ controls extrapolation. The design should use the full range and weigh the cost of extra capacity against the cost of failure. More years of data, threshold exceedances, comparable sites, or physical knowledge of the tail could narrow the range.

**7.** The [companion script](./code/08_extreme_value_statistics/records.py) finds mean record counts of $7.468$, $7.551$, and $7.465$ for Gaussian, exponential, and Pareto samples. All agree with $H_{1000}=7.485$.

For $40$ annual maxima,
$$
\mathbb{E}[R]=H_{40}=4.28,
\qquad
\operatorname{Var}(R)=4.28-1.62=2.66,
$$
so $\operatorname{SD}(R)=1.63$. Six records are only $(6-4.28)/1.63=1.1$ standard deviations above the mean. That is not strong evidence of a trend.

:::

## Further reading

- Fisher and Tippett (1928), *Math. Proc. Camb. Phil. Soc.* 24(2):180–190 — the original, readable account of the three limits and the Gaussian's slow convergence. [PDF](https://digital.library.adelaide.edu.au/server/api/core/bitstreams/d9d47fc7-41db-48cd-b8f7-41542c5e6397/content) · [doi](https://doi.org/10.1017/S0305004100015681).
- Coles, *An Introduction to Statistical Modeling of Extreme Values* (Springer, 2001) — block maxima, GEV fitting, return levels, and threshold methods. ISBN 9781852334598 · [HOLLIS](https://hollis.harvard.edu/discovery/search?query=any,contains,Coles%20introduction%20statistical%20modeling%20extreme%20values&tab=LibraryCatalog&search_scope=MyInstitution&vid=01HVD_INST:HVD2&offset=0).
- Embrechts, Klüppelberg and Mikosch, *Modelling Extremal Events: for Insurance and Finance* (Springer, 1997) — a mathematical treatment with finance applications and record theory. ISBN 9783540609315 · [HOLLIS](https://hollis.harvard.edu/discovery/search?query=any,contains,Modelling%20Extremal%20Events%20Embrechts&tab=LibraryCatalog&search_scope=MyInstitution&vid=01HVD_INST:HVD2&offset=0).
- [Generalized extreme value distribution](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution) — a quick plot of the three shapes (CC BY-SA).

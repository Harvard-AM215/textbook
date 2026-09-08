---
kernelspec:
  name: python3
  display_name: Python 3
---

# Chapter 4: Tournaments & Stochastic Simulation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harvard-AM215/textbook/blob/gh-pages/notebooks/04_tournaments.ipynb)
*Runs in the browser -- click the power icon above to start the kernel -- or in Colab, which has a full Python and a live network.*

> When a system is built from many random binary choices arranged in a tree (single-elimination tournaments, decision processes, branching cascades), Monte Carlo simulation is almost always faster than enumeration and lets you put numbers on questions like "how unfair is the bracket draw?"

:::{note} Running the code
The Python cells on this page run **in your browser**. Click the **power icon** at the top of the page to activate the kernel, then run — or edit — any cell. Packages (NumPy, SciPy, …) load automatically the first time you import them (a few seconds).
:::

## Motivation

Take an NCAA basketball tournament: 64 teams, six rounds, 63 games. You have a model for each pairwise matchup ("Duke beats Kentucky 65% of the time"). What is the probability each team wins the whole thing?

You could enumerate all $2^{63} \approx 10^{19}$ possible bracket outcomes and assign each one a probability. That would give an exact answer, but it is computationally infeasible. With Monte Carlo, you simulate a bracket in a millisecond, repeat the simulation a million times, and count championship frequencies. The calculation finishes in under a minute.

This is the first substantial application of the Monte Carlo method from Ch. 3. It is also the book's simplest *path-dependent* stochastic process: the matchups in round 3 depend on the winners from rounds 1 and 2. The chapters on random walks, GBM, and option pricing extend the same pattern: follow a sequence of random choices, then ask about an outcome at the end.

## Setup & notation

A single-elimination tournament with $n$ teams (we'll assume $n$ is a power of 2) takes $\log_2 n$ rounds. Each round halves the field. The structure has two ingredients:

- **The draw** — the bracket structure, telling you which teams play in round 1 and how winners are paired thereafter. We treat the draw as fixed for any one tournament.
- **The pairwise probability matrix** $P \in \mathbb{R}^{n \times n}$ where $P_{ij}$ is the probability that team $i$ beats team $j$ in a single match. We assume $P_{ji} = 1 - P_{ij}$ (no home-court advantage; only "who plays whom" matters), which makes $P$ a function of $\binom{n}{2}$ free parameters.

Each game is a Bernoulli trial with success probability $P_{ij}$. The chain of trials is structured by the bracket.

New symbols:
- $P_{ij}$ — probability team $i$ defeats team $j$ in a single match.
- $w_i$ — the (unknown) probability that team $i$ wins the championship.

## Core ideas

### Why simulation, not enumeration

For 4 teams, you could enumerate: 3 games, $2^3 = 8$ outcomes, each with a closed-form probability. For 64 teams, $2^{63} \approx 10^{19}$ outcomes — you cannot enumerate. Monte Carlo's $1/\sqrt N$ error scaling (Ch. 3) is dimension-independent: 1 million simulations gives standard error $\sim 0.001$ regardless of bracket size. The trade is precision for tractability, and for any practical question ("what's the probability team X wins?") the precision is more than enough.

### The simulation loop

Conceptually:

```
def simulate_tournament(draw, P, rng):
    survivors = list(initial pairings from draw)
    while len(survivors) > 1:
        next_round = []
        for (i, j) in pair_up(survivors):
            winner = i if rng.uniform() < P[i, j] else j
            next_round.append(winner)
        survivors = next_round
    return survivors[0]   # champion
```

Run this $N$ times, count championships. Team $i$'s estimated win probability is $\hat w_i = (\text{championships})_i / N$, with standard error $\sqrt{\hat w_i (1 - \hat w_i) / N}$ from the Bernoulli formula in Ch. 3.

The full version with a 4-team example and clean bracket-pairing logic is in [`code/04_tournaments/simulate_4team.py`](./code/04_tournaments/simulate_4team.py).

### The "effect of the draw" question

A subtle point that surfaces immediately when you simulate: with the same matrix $P$, *different draws produce different championship probabilities*. A team that gets paired against the strongest competitor in round 1 has a very different path to victory than a team that draws byes and weaker opponents. To quantify this, simulate many tournaments per draw, then sample many draws. The variance across draws (for a single team) measures how much the bracket structure matters relative to the underlying skill.

For NCAA-style 64-team brackets where the draw is constructed by a seeding committee rather than at random, this analysis is a useful check on whether the seeding process is producing a fair playing field — or systematically advantaging the top seeds beyond what their skill would predict.

### Estimating $P$ from data: a forward pointer to Ch. 5

In practice, nobody hands you $P$. You estimate it from past matchup data, and the natural model is a Bradley–Terry / logistic regression form:
$$
P_{ij} = \sigma(R_i - R_j) = \frac{1}{1 + e^{-(R_i - R_j)}},
$$
where each team has a single "rating" $R_i$ and the win probability depends only on the rating gap. Fit by MLE over the entire history of matches. Update online as new matches happen and you have **ELO ratings** — which Ch. 5 revisits, showing the update rule is a random walk in rating space (Exercise 5 there). The connection is: tournament simulation is what you do *after* you have ratings; ELO is what you do *before*, to get them.

## Worked example: 4-team tournament

Take 4 teams with a randomly-generated $P$ matrix and a fixed bracket: $(0\,\text{vs}\,1)$ and $(2\,\text{vs}\,3)$ in round 1, winners meet in the final.

```{code-cell} python
import numpy as np

def random_P(n, rng):
    """Random pairwise probability matrix with P_ji = 1 - P_ij."""
    upper = rng.uniform(size=(n, n))
    P = np.triu(upper, k=1) + (1 - np.triu(upper, k=1)).T
    np.fill_diagonal(P, 0.5)
    return P

def simulate(draw, P, rng):
    """Single-elimination simulation; returns champion index."""
    survivors = list(draw)
    while len(survivors) > 1:
        next_round = []
        for i, j in zip(survivors[::2], survivors[1::2]):
            next_round.append(i if rng.uniform() < P[i, j] else j)
        survivors = next_round
    return survivors[0]

rng = np.random.default_rng(0)
n, N = 4, 100_000
P = random_P(n, rng)
champs = np.array([simulate([0, 1, 2, 3], P, rng) for _ in range(N)])

for i in range(n):
    p = (champs == i).mean()
    se = np.sqrt(p * (1 - p) / N)
    print(f"team {i}: win prob = {p:.4f} +/- {2*se:.4f}")
```

The four win probabilities sum to 1 — a verification check in the Ch. 3 sense: it can catch a bug in the simulation loop, but says nothing about whether $P$ describes the real teams. The $2\sigma$ error bars are $\sim 0.003$ at $N = 10^5$; to halve them, use $4 \times 10^5$ simulations — the same $1/\sqrt N$ scaling we have seen since Ch. 3.

The full code includes a 16-team bracket and a "draw effect" analysis in [`code/04_tournaments/simulate_4team.py`](./code/04_tournaments/simulate_4team.py) and [`code/04_tournaments/draw_effect.py`](./code/04_tournaments/draw_effect.py).

## Intuition

> **Why does the upset rate fall as the bracket progresses?** Because the field gets stronger each round as weak teams are eliminated. Later matchups have stronger teams on both sides, and those teams tend to win their head-to-heads more reliably. Most surprises therefore occur early. This is consistent with the famous Cinderella runs in March Madness: a 12-seed almost never wins it all, but it routinely beats a 5-seed in round 1.

> **Why is the variance of championship probability across draws often big?** Because tournaments are *path-dependent*: who you play in round 1 affects who you might play in round 2. If the draw puts two strong teams in the same half of the bracket, only one of them can make the final — guaranteeing a "weaker" finalist from that half. The seeding committee tries to undo this with deliberate bracket construction, but in random draws (e.g., World Cup group-stage assignments by drawing from pots), the variance is real and routinely shifts a team's championship odds by factors of 2 or more.

> **Why is "model the matchups, simulate the bracket" the right factorization?** Because (a) the matchup model can be calibrated against far more data than the bracket itself (every regular-season game contributes), and (b) the bracket-level question is then a pure stochastic-process simulation that doesn't need any new data. Compare this to the alternative — fit one giant model directly on past tournament *outcomes* — which throws away most of the information and overfits to the small sample of past tournaments.

## Connections

- **Builds on:** Bernoulli trials (Ch. 2); Monte Carlo for probabilities (Ch. 3); the $1/\sqrt N$ standard-error formula.
- **Used in:** Ch. 5, Exercise 5 (ELO ratings — the natural way to estimate $P$ before simulating with it — reinterpreted as a random walk); broadly, anywhere a stochastic process is structured as a tree of binary choices.
- **Adjacent ideas not pursued here:** sequential Monte Carlo / particle filters (more efficient when the tree is huge but most branches are unlikely); reinforcement learning (useful when the "matchup probabilities" depend on actions you can choose, not just on the players).

## Exercises

1. **Conceptual.** A friend insists "the seeding system is rigged because the 1-seed wins more often than its skill alone would predict." Sketch how you would test this claim quantitatively using nothing but historical bracket-fill data.

2. **Computational.** Simulate a 16-team bracket with $P$ generated randomly (with $P_{ij} \sim$ Uniform on $[0, 1]$, lower triangle filled in). Run $10^5$ tournaments per draw and report the championship probabilities for the four highest-skill teams (define "highest skill" as the team with the largest sum of pairwise win probabilities, $\sum_j P_{ij}$).

3. **Convergence.** Repeat Exercise 2 with $N = 10^2, 10^3, 10^4, 10^5$. Plot the standard error of one team's win probability vs $N$ on log-log axes. Does the slope match the predicted $-1/2$?

4. **Effect of the draw.** Pick one specific team and compute its championship probability under 100 randomly-permuted draws. Plot a histogram of the resulting probabilities. How wide is it relative to the team's mean win probability? What does this say about whether the draw is a meaningful source of variance?

5. **Connection to ELO.** Suppose teams have ELO ratings $R_i$ and the matchup probability is $P_{ij} = \sigma(R_i - R_j)$ where $\sigma$ is the logistic function. Generate 8 teams with ratings $R \in \{0, 100, 200, \dots, 700\}$, build $P$, and simulate $10^5$ tournaments. Plot championship probability vs ELO rating. Is the relationship linear, exponential, or something else?

6. **Modeling judgment.** I propose to predict the World Cup champion by training a neural network directly on past World Cup outcomes (one tournament per data point, $\sim 20$ data points total). Sketch three things that go wrong with this approach and propose a better alternative grounded in the chapter's "matchup model + bracket simulation" framing.

:::{admonition} Solutions
:class: dropdown

**1.** Build a matchup-probability model $P$ from regular-season games (lots of data, no tournament-specific selection bias). Then for each historical tournament, simulate the bracket many times under that $P$ and record the predicted distribution over winners. Compare the simulated 1-seed win rate to the actual 1-seed win rate. If actuals are systematically higher, the deviation is a quantitative measure of "seeding advantage beyond skill." If they match, the seeding system is just doing its job — making the bracket reflect skill.

**2.** See [`code/04_tournaments/simulate_4team.py`](./code/04_tournaments/simulate_4team.py) (the same file generalizes to 16 teams).

**3.** Standard error follows $\sqrt{p(1-p)/N}$, which on log-log gives slope $-1/2$. Confirm in [`code/04_tournaments/convergence.py`](./code/04_tournaments/convergence.py).

**4.** See [`code/04_tournaments/draw_effect.py`](./code/04_tournaments/draw_effect.py). For a randomly generated $P$ on 8 teams, the variance can be large: with the script's default seed, the "strongest" team's championship probability ranges from $0\%$ to $100\%$ across draws (SD comparable to the mean). A "strongest" team in a random $P$ is often only marginally stronger. Small per-match advantages compound over $\log_2 n$ rounds, so a draw that pairs the team with the second- and third-strongest teams can sharply reduce its odds. The draw effect is much smaller in real tournaments where seeding deliberately separates top teams, but this experiment shows that bracket structure can still matter substantially.

**5.** See [`code/04_tournaments/elo_to_winprob.py`](./code/04_tournaments/elo_to_winprob.py). The relationship is approximately *exponential* in ELO rating: the top-rated team's championship probability is dramatically higher than the second-highest, which is dramatically higher than the third, and so on. This is because winning a tournament requires winning *several* matches in a row, and small per-match advantages compound multiplicatively.

**6.** Three failures: (i) **Sample size**: $\sim 20$ tournaments is far too few to fit any meaningful model — the network will memorize noise. (ii) **Heterogeneous teams**: most tournaments include teams that weren't in any past tournament; a team-level model trained on past winners can't generalize. (iii) **Selection bias**: past winners are a biased sample (tournaments held over many years with changing FIFA rankings; rule changes; format changes). Better approach: factorize. Train a *match-level* model on the much larger dataset of all international matches (thousands per year), getting a $P$ that depends on team attributes (ELO, FIFA ranking, recent form). Then simulate the World Cup bracket structure with this $P$. The match model has lots of data; the simulation needs no extra data; the two pieces together give you tournament predictions with quantified uncertainty.

:::

## Further reading

- Massey, *Statistical Models Applied to the Rating of Sports Teams* (Bluefield College
  honors thesis, 1997) — the first widely-known method for fitting team strengths from
  win-loss data. **Free in full** from the author:
  [masseyratings.com](https://masseyratings.com/theory/massey97.pdf).
- For the deeper view of paired-comparison models: Stern, "A Brownian Motion Model for the
  Progress of Sports Scores", *JASA* 89(427):1128–1134 (1994) — connects directly to Chs. 5
  and 6 of this book.
  [doi:10.1080/01621459.1994.10476851](https://doi.org/10.1080/01621459.1994.10476851) ·
  [HOLLIS](https://hollis.harvard.edu/discovery/search?query=any,contains,10.1080/01621459.1994.10476851&tab=Everything&search_scope=MyInst_and_CI&vid=01HVD_INST:HVD2&offset=0). No free version; the Berkeley
  course-site PDF is an unsanctioned scan.
- Silver, *The Signal and the Noise* (2012), Ch. 3 (on baseball) — popular treatment of the
  "probabilistic forecast → tournament simulation" pipeline. ISBN 9781594204111 ·
  [HOLLIS](https://hollis.harvard.edu/discovery/search?query=any,contains,Signal%20and%20the%20Noise%20Silver&tab=LibraryCatalog&search_scope=MyInstitution&vid=01HVD_INST:HVD2&offset=0).

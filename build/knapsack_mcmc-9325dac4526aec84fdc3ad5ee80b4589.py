"""Random-walk and Metropolis MCMC for the 0/1 knapsack problem.

Two search rules, both proposing one random item flip per step and both
rejecting infeasible proposals:
- `walk_blind`: accept every feasible proposal, regardless of value. With
  positive weights every feasible subset is reachable, but the walk ignores
  value and wanders.
- `walk_metropolis`: accept a feasible proposal that raises the value; accept
  one that lowers it by dv with probability exp(-dv / T). The simplest
  value-aware rule.

An exact dynamic-programming solution provides the optimum for comparison.

Solution to Chapter 3 Exercise 6.
"""

import numpy as np
import matplotlib.pyplot as plt


def random_instance(n, rng, w_max=50, v_max=100, W=100):
    """Generate a random knapsack instance."""
    w = rng.integers(1, w_max, size=n)
    v = rng.integers(1, v_max, size=n)
    return w, v, W


def value(x, v):
    return int(v @ x)


def feasible(x, w, W):
    return int(w @ x) <= W


def walk_blind(w, v, W, n_steps, rng):
    """Accept every feasible proposal; ignore value."""
    n = len(w)
    x = np.zeros(n, dtype=int)
    best_x, best_v = x.copy(), 0
    history = [0]

    for _ in range(n_steps):
        i = rng.integers(n)
        x_new = x.copy()
        x_new[i] = 1 - x_new[i]

        if feasible(x_new, w, W):
            x = x_new
            if value(x, v) > best_v:
                best_v = value(x, v)
                best_x = x.copy()
        history.append(best_v)

    return best_x, best_v, history


def walk_metropolis(w, v, W, n_steps, rng, T=20.0):
    """Metropolis rule: accept feasible improvements always, feasible
    decreases of size dv with probability exp(-dv / T)."""
    n = len(w)
    x = np.zeros(n, dtype=int)
    cur_v = 0
    best_x, best_v = x.copy(), 0
    history = [0]

    for _ in range(n_steps):
        i = rng.integers(n)
        x_new = x.copy()
        x_new[i] = 1 - x_new[i]

        if feasible(x_new, w, W):
            new_v = value(x_new, v)
            dv = new_v - cur_v
            if dv >= 0 or rng.random() < np.exp(dv / T):
                x, cur_v = x_new, new_v
                if cur_v > best_v:
                    best_v = cur_v
                    best_x = x.copy()
        history.append(best_v)

    return best_x, best_v, history


def knapsack_exact(w, v, W):
    """Exact optimum by dynamic programming over capacities 0..W."""
    best = np.zeros(W + 1, dtype=int)
    for wi, vi in zip(w, v):
        for c in range(W, wi - 1, -1):
            best[c] = max(best[c], best[c - wi] + vi)
    return int(best[W])


def main():
    rng = np.random.default_rng(seed=0)
    n = 50
    w, v, W = random_instance(n, rng, W=200)
    print(f"Instance: n={n}, W={W}")
    print(f"  weights: {w}")
    print(f"  values : {v}")

    n_steps = 10_000
    T = 20.0
    _, best_a, hist_a = walk_blind(w, v, W, n_steps, rng)
    _, best_b, hist_b = walk_metropolis(w, v, W, n_steps, rng, T=T)
    opt = knapsack_exact(w, v, W)

    print(f"\nBlind walk best value:            {best_a}")
    print(f"Metropolis (T={T:g}) best value:   {best_b}")
    print(f"Exact optimum (dynamic program):  {opt}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist_a, label="blind walk")
    ax.plot(hist_b, label=f"Metropolis, T={T:g}")
    ax.axhline(opt, color="k", ls="--", lw=1, label="exact optimum")
    ax.set(
        xlabel="step",
        ylabel="best value found so far",
        title="Knapsack search: two acceptance rules",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

"""Check Keller's three-arc solution without assuming it: discretize and optimize.

Companion to Chapter 9 Worked Example 1. The three-arc form (sprint, cruise, fade) was an
assumption Keller then showed to be stationary. Here the control f(t) is instead a free
number at each of N time nodes, the dynamics are stepped forward from those numbers, and a
constrained optimizer maximises the distance covered in a fixed time T subject to
0 <= f_k <= F and E_k >= 0 at every node. Nothing tells it about arcs. If Keller is right,
the arcs should appear on their own.

Discretisation: f is piecewise constant on each step, so the velocity update
v_{k+1} = v_k e^{-dt/tau} + f_k tau (1 - e^{-dt/tau}) is exact; the energy update uses the
mid-step speed. The optimizer is SciPy's SLSQP, with the energy constraints imposed
directly rather than by a penalty. Finite-difference gradients are affordable because
the rollout is vectorised (a linear recurrence) and N is a few hundred.

The run prints result.success, the iteration count, the distance against the three-arc
value, the smallest energy reached, and the switch times read off the profile, at N and
at 2N so the discretisation error is visible.

    python3 direct_transcription.py          # the 400 m, N = 200 and 400
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keller import KELLER_SI, T_of_D, solve_race, velocity_profile  # noqa: E402


def rollout(f, T, tau, F, sigma, E0):
    """Step the dynamics under the piecewise-constant control f: returns (D, v, E).

    v has N + 1 entries (v[0] = 0) and so has E (E[0] = E0).
    """
    N = f.size
    dt = T / N
    a = np.exp(-dt / tau)
    v = np.concatenate([[0.0], lfilter([(1.0 - a) * tau], [1.0, -a], f)])
    v_mid = 0.5 * (v[:-1] + v[1:])
    E = E0 + dt * np.cumsum(sigma - f * v_mid)
    E = np.concatenate([[E0], E])
    D = float(np.sum(v_mid) * dt)
    return D, v, E


def solve_direct(T, tau, F, sigma, E0, N=200, f0=None, maxiter=5000):
    """Maximise D over the N control values with 0 <= f <= F and E >= 0 at every node."""
    x0 = np.full(N, 0.8 * F) if f0 is None else np.interp(
        np.linspace(0, 1, N), np.linspace(0, 1, f0.size), f0)
    res = minimize(lambda f: -rollout(f, T, tau, F, sigma, E0)[0], x0, method="SLSQP",
                   bounds=[(0.0, F)] * N,
                   constraints=[dict(type="ineq",
                                     fun=lambda f: rollout(f, T, tau, F, sigma, E0)[2])],
                   options=dict(maxiter=maxiter, ftol=1e-10))
    D, v, E = rollout(res.x, T, tau, F, sigma, E0)
    return dict(f=res.x, v=v, E=E, D=D, success=bool(res.success), nit=int(res.nit),
                message=res.message, t=np.linspace(0.0, T, N + 1))


def switch_times(sol, F, tol_f=0.02, tol_E=1.0):
    """Read t1 (force first leaves the bound) and t2 (store first empties) off a solution."""
    t_mid = 0.5 * (sol["t"][:-1] + sol["t"][1:])
    off_bound = np.nonzero(sol["f"] < F * (1 - tol_f))[0]
    t1 = t_mid[off_bound[0]] if off_bound.size else sol["t"][-1]
    empty = np.nonzero(sol["E"] < tol_E)[0]
    t2 = sol["t"][empty[0]] if empty.size else sol["t"][-1]
    return t1, t2


def main(params=KELLER_SI, D_race=400.0, N=200, show=True):
    T = T_of_D(D_race, **params)
    D_ref, t1_ref, t2_ref = solve_race(T, **params)
    print(f"{D_race:.0f} m with Keller's constants: T = {T:.2f} s; three-arc solution "
          f"D = {D_ref:.2f} m, t1 = {t1_ref:.2f} s, t2 = {t2_ref:.2f} s")
    sols = {}
    f0 = None
    for n in (N, 2 * N):
        sol = solve_direct(T, N=n, f0=f0, **params)
        sols[n] = sol
        f0 = sol["f"]
        t1, t2 = switch_times(sol, params["F"])
        print(f"N = {n:4d}: success = {sol['success']}, {sol['nit']} iterations, "
              f"D = {sol['D']:.2f} m ({sol['D'] - D_ref:+.3f}), min E = {sol['E'].min():.3f}, "
              f"force leaves F at t = {t1:.2f} s, store empties at t = {t2:.2f} s"
              f"  [{sol['message']}]")

    if show:
        sol = sols[2 * N]
        t_a, v_a, _, _ = velocity_profile(T, **params)
        t_a = np.asarray(t_a)
        f_a = np.gradient(v_a, t_a) + v_a / params["tau"]
        E_a = params["E0"] + np.concatenate(
            [[0.0], np.cumsum(np.diff(t_a) * (params["sigma"] - (f_a * v_a)[:-1]))])
        fig, axes = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True)
        axes[0].plot(sol["t"], sol["v"], color="C3", lw=3, alpha=0.5, label="free control")
        axes[0].plot(t_a, v_a, color="k", lw=1.2, label="three-arc solution")
        axes[0].set_ylabel("v (m/s)")
        axes[0].legend(frameon=False)
        t_mid = 0.5 * (sol["t"][:-1] + sol["t"][1:])
        axes[1].plot(t_mid, sol["f"], color="C3", lw=3, alpha=0.5)
        axes[1].plot(t_a, f_a, color="k", lw=1.2)
        axes[1].axhline(params["F"], color="0.6", lw=0.8, ls=":")
        axes[1].set_ylabel("f (m/s^2)")
        axes[2].plot(sol["t"], sol["E"], color="C3", lw=3, alpha=0.5)
        axes[2].plot(t_a, E_a, color="k", lw=1.2)
        axes[2].axhline(0, color="0.6", lw=0.8, ls=":")
        axes[2].set_ylabel("E (J/kg)")
        axes[2].set_xlabel("t (s)")
        axes[0].set_title(f"The {D_race:.0f} m two ways (T = {T:.2f} s)", loc="left")
        fig.tight_layout()
        plt.show()
    return sols


if __name__ == "__main__":
    main()

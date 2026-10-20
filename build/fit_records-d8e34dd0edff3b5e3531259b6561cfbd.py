"""Fit Keller's four constants to the 1972 records, then test them on later data.

Companion to Chapter 9 Worked Examples 2 and 3.

Stage 1 fits (tau, F) to the eight dashes by relative least squares, using the closed-form
dash relation. Stage 2 holds those and fits (sigma, E0) to the fourteen longer records,
with the full three-arc solve inside every residual. The consistency check recomputes Dc
and confirms the eight races treated as dashes really are shorter than it. The loss valley
is then mapped on a grid around the stage-2 minimum: the range of sigma, of E0 and of the
ratio E0/sigma over which the loss stays within a chosen RMS-error tolerance. This describes sensitivity to the tolerance,
not a statistical confidence interval.

    python3 fit_records.py            # the 1972 fit and the valley
    python3 fit_records.py --records  # Keller's published constants against today's records
    python3 fit_records.py --olympic  # ... and against the Olympic finals 2008-2024, with
                                      #     the unconstrained and fixed-tau modern refits
    python3 fit_records.py --quick    # stages 1 and 2 only (the tests use this)

Every comparison with later data uses Keller's PUBLISHED constants, not the refit, so that
the chapter's tables all refer to one set of numbers.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keller import (  # noqa: E402
    CAL,
    KELLER_PUBLISHED,
    KELLER_RECORDS,
    KELLER_SI,
    N_DASHES,
    WORLD_RECORDS,
    D_dash,
    Dc_of,
    T_dash_of_D,
    T_of_D,
    Tc_of,
    fmt_time,
    olympic_bests,
    olympic_flat,
    rel_error,
)

D_ALL = np.array([r[1] for r in KELLER_RECORDS])
T_ALL = np.array([r[2] for r in KELLER_RECORDS])
REACTION_TIME = 0.15   # s, in official electronic sprint times, absent from the model


# --------------------------------------------------------------------------------------
# The two stages
# --------------------------------------------------------------------------------------

def fit_dashes(D, T_rec, tau0=1.0, F0=10.0):
    """Stage 1: (tau, F) from the dashes by relative least squares on log-parameters."""
    def resid(p):
        tau, F = np.exp(p)
        return (T_rec - np.array([T_dash_of_D(d, tau, F) for d in D])) / T_rec
    r = least_squares(resid, np.log([tau0, F0]))
    tau, F = np.exp(r.x)
    return tau, F, float(np.sum(r.fun**2))


def stage2_loss(sigma, E0, tau, F, D, T_rec):
    """Sum of squared relative errors over the longer records, full solve inside."""
    T_th = np.array([T_of_D(d, tau, F, sigma, E0, xtol=1e-4) for d in D])
    return float(np.sum(((T_rec - T_th) / T_rec) ** 2))


def fit_aerobic(D, T_rec, tau, F, sigma0=40.0, E00=2000.0):
    """Stage 2: (sigma, E0) from the longer records, tau and F held fixed."""
    def resid(q):
        sigma, E0 = np.exp(q)
        T_th = np.array([T_of_D(d, tau, F, sigma, E0, xtol=1e-4) for d in D])
        return (T_rec - T_th) / T_rec
    r = least_squares(resid, np.log([sigma0, E00]), diff_step=1e-4)
    sigma, E0 = np.exp(r.x)
    return sigma, E0, float(np.sum(r.fun**2))


def constants_table(label_a, a, label_b, b):
    """Print two sets of constants side by side, in Keller's published units."""
    rows = [("tau (s)", a["tau"], b["tau"], "{:.3f}"),
            ("F (m/s^2)", a["F"], b["F"], "{:.2f}"),
            ("sigma (cal/kg s)", a["sigma"] / CAL, b["sigma"] / CAL, "{:.2f}"),
            ("E0 (cal/kg)", a["E0"] / CAL, b["E0"] / CAL, "{:.0f}"),
            ("Dc (m)", Dc_of(**a), Dc_of(**b), "{:.0f}"),
            ("E0/sigma (s)", a["E0"] / a["sigma"], b["E0"] / b["sigma"], "{:.1f}")]
    print(f"{'constant':>18} {label_a:>14} {label_b:>14}")
    for name, x, y, fmt in rows:
        print(f"{name:>18} {fmt.format(x):>14} {fmt.format(y):>14}")


def fit_1972(verbose=True):
    """Both stages on Keller's table; returns the fitted constants in SI."""
    tau, F, l1 = fit_dashes(D_ALL[:N_DASHES], T_ALL[:N_DASHES])
    sigma, E0, l2 = fit_aerobic(D_ALL[N_DASHES:], T_ALL[N_DASHES:], tau, F)
    fit = dict(tau=tau, F=F, sigma=sigma, E0=E0)
    if verbose:
        print(f"stage 1 (8 dashes):      tau = {tau:.3f} s   F = {F:.2f} m/s^2   "
              f"loss = {l1:.2e}")
        print(f"stage 2 (14 longer):     sigma = {sigma / CAL:.2f} cal/(kg s)   "
              f"E0 = {E0 / CAL:.0f} cal/kg   loss = {l2:.2e}")
        Dc = Dc_of(**fit)
        print(f"consistency: Dc = {Dc:.0f} m; longest dash fitted = "
              f"{D_ALL[N_DASHES - 1]:.0f} m, shortest longer race = {D_ALL[N_DASHES]:.0f} m"
              f" -> {'consistent' if D_ALL[N_DASHES - 1] < Dc < D_ALL[N_DASHES] else 'NOT consistent'}")
        print()
        constants_table("Keller 1973", KELLER_SI, "this fit", fit)
    return fit, l2


# --------------------------------------------------------------------------------------
# The loss valley
# --------------------------------------------------------------------------------------

def loss_valley(fit, loss_min, n=21, span=0.06):
    """Map the stage-2 loss on a grid around the minimum and report how wide the valley is.

    The loss is a sum of 14 squared relative errors, so sqrt(loss / 14) is the root-mean-
    square relative error of the fit. The report gives, for RMS errors up to 0.1, 0.2 and
    0.5 percentage points above the minimum, the ranges of sigma, E0 and E0/sigma that stay
    inside, and where Keller's own published pair sits.
    """
    tau, F = fit["tau"], fit["F"]
    D, T_rec = D_ALL[N_DASHES:], T_ALL[N_DASHES:]
    sig = fit["sigma"] * np.linspace(1 - span, 1 + span, n)
    e0 = fit["E0"] * np.linspace(1 - 2 * span, 1 + 2 * span, n)
    L = np.array([[stage2_loss(s, e, tau, F, D, T_rec) for s in sig] for e in e0])
    S, E = np.meshgrid(sig, e0)
    rms = np.sqrt(L / len(D))
    rms_min = np.sqrt(loss_min / len(D))
    print(f"loss grid {n}x{n}, sigma +/-{100 * span:.0f}%, E0 +/-{200 * span:.0f}%: "
          f"RMS relative error at the minimum {100 * rms_min:.2f}%")
    for extra in (0.001, 0.002, 0.005):
        inside = rms <= rms_min + extra
        if not inside.any():
            continue
        print(f"RMS error within {100 * extra:.1f} points of the minimum: sigma "
              f"{S[inside].min() / CAL:.2f} to {S[inside].max() / CAL:.2f} cal/(kg s), E0 "
              f"{E[inside].min() / CAL:.0f} to {E[inside].max() / CAL:.0f} cal/kg, E0/sigma "
              f"{(E / S)[inside].min():.1f} to {(E / S)[inside].max():.1f} s"
              + (f"; corr(sigma, E0) inside = "
                 f"{np.corrcoef(S[inside], E[inside])[0, 1]:+.2f}" if inside.sum() > 2
                 else ""))
    keller_rms = np.sqrt(stage2_loss(KELLER_SI["sigma"], KELLER_SI["E0"], tau, F, D, T_rec)
                         / len(D))
    print(f"Keller's published (sigma, E0) = (9.93, 575) with this tau and F: RMS error "
          f"{100 * keller_rms:.2f}%, i.e. {100 * (keller_rms - rms_min):.2f} points above "
          f"the minimum")
    return sig, e0, L


# --------------------------------------------------------------------------------------
# Later data
# --------------------------------------------------------------------------------------

def error_table(rows, params, header):
    """rows: (label, D, T_data, who). Prints the theory time and error; returns errors."""
    print(header)
    print(f"{'race':>14} {'data':>8} {'theory':>8} {'error':>7}   who")
    errs = []
    for label, D, T_data, who in rows:
        T_th = T_of_D(D, **params)
        e = rel_error(T_th, T_data)
        errs.append((e, label, who))
        print(f"{label:>14} {fmt_time(T_data):>8} {fmt_time(T_th):>8} {e:+6.1f}%   {who}")
    e = np.array([x[0] for x in errs])
    lo, hi = min(errs), max(errs)
    print(f"median |error| {np.median(np.abs(e)):.1f}%; most negative {lo[0]:+.1f}% "
          f"({lo[1]}, {lo[2]}); most positive {hi[0]:+.1f}% ({hi[1]}, {hi[2]})")
    return errs


def records_comparison(params=KELLER_SI):
    rows = [(f"{int(d)} m", d, t, f"{who} {yr}") for d, t, who, yr in WORLD_RECORDS]
    return error_table(rows, params, "Keller's 1973 constants against today's world records "
                       "(error = theory minus record, as % of record)")


def olympic_comparison(params=KELLER_SI):
    rows = [(f"{y} {int(d)} m", d, t, who) for y, d, t, who in olympic_flat()]
    return error_table(rows, params, "Keller's 1973 constants against Olympic winning times "
                       "2008-2024 (error = theory minus winning time, as % of it)")


def modern_refit(verbose=True, reaction_time=REACTION_TIME):
    """Compare an unconstrained dash fit with a fit that holds tau fixed."""
    bests = olympic_bests()
    D = np.array([b[0] for b in bests])
    T = np.array([b[1] for b in bests])
    dash = D < 300.0
    tau_bad, F_bad, l_bad = fit_dashes(D[dash], T[dash])
    if verbose:
        print(f"naive stage 1 on the two Olympic dash distances (100 m, 200 m): "
              f"tau = {tau_bad:.2e} s, F = {F_bad:.2e} m/s^2, loss = {l_bad:.1e}")
        print(f"  the fit runs away: tau -> 0 and F -> infinity with F tau = "
              f"{tau_bad * F_bad:.1f} m/s, a runner who reaches top speed instantly. Two "
              f"times inconsistent with continued acceleration cannot both be matched "
              f"by this common dash model")
    tau = KELLER_SI["tau"]
    Ts = T[dash] - reaction_time

    def loss_F(logF):
        F = np.exp(logF)
        return float(np.sum(((Ts - np.array([T_dash_of_D(d, tau, F) for d in D[dash]]))
                             / Ts) ** 2))
    rF = minimize_scalar(loss_F, bounds=(np.log(8.0), np.log(25.0)), method="bounded")
    F = float(np.exp(rF.x))
    sigma, E0, l2 = fit_aerobic(D[~dash], T[~dash] - reaction_time, tau, F)
    fit = dict(tau=tau, F=F, sigma=sigma, E0=E0)
    if verbose:
        print(f"fixed-tau refit: tau held at {tau} s, {reaction_time} s reaction time removed, "
              f"F from the two dashes, (sigma, E0) from 400-10000 m Olympic bests")
        constants_table("Keller 1973", KELLER_SI, "Olympic bests", fit)
    return fit, (tau_bad, F_bad)


def plot_speed_curve(params=KELLER_SI, refit=None):
    """Average speed D/T against distance: theory, 1972 records, today's records, finals."""
    Dgrid = np.concatenate([np.linspace(5, 295, 80), np.geomspace(300, 11000, 100)])
    Tgrid = np.array([T_of_D(d, **params) for d in Dgrid])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.semilogx(Dgrid, Dgrid / Tgrid, color="k", lw=1.5, label="theory, Keller's 1973 constants")
    if refit is not None:
        Tr = np.array([T_of_D(d, **refit) for d in Dgrid])
        ax.semilogx(Dgrid, Dgrid / (Tr + REACTION_TIME), color="C0", lw=1.2, ls="--",
                    label="theory, refit to Olympic bests")
    ax.plot(D_ALL, D_ALL / T_ALL, "o", mfc="white", mec="k", label="world records 1972")
    wr_d = np.array([r[0] for r in WORLD_RECORDS])
    wr_t = np.array([r[1] for r in WORLD_RECORDS])
    ax.plot(wr_d, wr_d / wr_t, "s", color="C3", label="world records 2026")
    ol = olympic_flat()
    ax.plot([d for _, d, _, _ in ol], [d / t for _, d, t, _ in ol], "D", ms=4, color="C1",
            alpha=0.7, label="Olympic champions 2008-24")
    ax.axvline(Dc_of(**params), color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("race distance D (m)")
    ax.set_ylabel("average speed D/T (m/s)")
    ax.set_xlim(40, 11000)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Keller's curve, fitted to 1972, against fifty years of later running",
                 loc="left")
    fig.tight_layout()
    plt.show()


def main(argv=None, show=True):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", action="store_true")
    ap.add_argument("--olympic", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.records:
        records_comparison()
        return
    if args.olympic:
        olympic_comparison()
        print()
        refit, _ = modern_refit()
        if show:
            plot_speed_curve(refit=refit)
        return
    fit, loss_min = fit_1972()
    if not args.quick:
        print()
        loss_valley(fit, loss_min)
        if show:
            plot_speed_curve()


if __name__ == "__main__":
    main()

"""Keller's model of optimal racing: the solver, the physiological constants and the data.

Companion module for Chapter 9. The chapter's browser cells carry their own copies of the
functions they need (the in-browser kernel cannot import this file), so this module exists
for the companion scripts and for tests/test_published_numbers.py, which also checks that
the copies embedded in the chapter match the data here.

The model, per unit mass of runner, with the propulsive force f(t) as the control:

    D = int_0^T v dt,   dv/dt + v/tau = f,  v(0) = 0,   0 <= f <= F,
    dE/dt = sigma - f v,  E(0) = E0,  E >= 0.

Given (tau, F, sigma, E0) and D, choose f to minimise T; equivalently, given T, maximise D.
The optimal race is a sprint at f = F until t1, a cruise at constant speed until t2, and a
fade on E = 0 to the line (Keller 1974). The solver here reduces the whole problem to one
unknown, t1: the cruise speed follows by continuity, t2 from E(t2) = 0 (linear in t2 because
the cruise speed is constant), and the fade-arc distance has a closed form, so D(t1) is a
cheap function maximised over one variable.

Units: metres, seconds, and energy per unit mass in J/kg (= m^2/s^2). Keller published sigma
and E0 in cal/kg; KELLER_SI converts them with 1 cal = 4.184 J.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar

CAL = 4.184  # joules per (small) calorie

# Keller (1973), Table 2, as published: tau in s, F in m/s^2, sigma in cal/(kg s), E0 in
# cal/kg. Dc = 291 m.
KELLER_PUBLISHED = dict(tau=0.892, F=12.2, sigma=9.93, E0=575.0)
KELLER_SI = dict(tau=0.892, F=12.2, sigma=9.93 * CAL, E0=575.0 * CAL)

YD, MILE = 0.9144, 1609.344


def mmss(m, s):
    """Seconds from minutes and seconds."""
    return 60.0 * m + s


# Keller (1973), Table 1: the 22 world records as of 1972, (label, distance m, time s).
# The first N_DASHES are the "short sprints" fitted in stage 1.
KELLER_RECORDS = [
    ("50 yd", 50 * YD, 5.1), ("50 m", 50.0, 5.5),
    ("60 yd", 60 * YD, 5.9), ("60 m", 60.0, 6.5),
    ("100 yd", 100 * YD, 9.1), ("100 m", 100.0, 9.9),
    ("200 m", 200.0, 19.5), ("220 yd", 220 * YD, 19.5),
    ("400 m", 400.0, 44.5), ("440 yd", 440 * YD, 44.9),
    ("800 m", 800.0, mmss(1, 44.3)), ("880 yd", 880 * YD, mmss(1, 44.9)),
    ("1000 m", 1000.0, mmss(2, 16.2)), ("1500 m", 1500.0, mmss(3, 33.1)),
    ("1 mile", MILE, mmss(3, 51.1)), ("2000 m", 2000.0, mmss(4, 56.2)),
    ("3000 m", 3000.0, mmss(7, 39.6)), ("2 miles", 2 * MILE, mmss(8, 19.8)),
    ("3 miles", 3 * MILE, mmss(12, 50.4)), ("5000 m", 5000.0, mmss(13, 16.6)),
    ("6 miles", 6 * MILE, mmss(26, 47.0)), ("10000 m", 10000.0, mmss(27, 39.4)),
]
N_DASHES = 8

# Men's Olympic champions, 2008-2024 (the 2020 Games were held in 2021), official winning
# times in seconds: {year: {distance m: (time, athlete)}}.
OLYMPIC_RESULTS = {
    2008: {100: (9.69, "Usain Bolt"), 200: (19.30, "Usain Bolt"),
           400: (43.75, "LaShawn Merritt"), 800: (mmss(1, 44.65), "Wilfred Bungei"),
           1500: (mmss(3, 33.11), "Asbel Kiprop"),
           5000: (mmss(12, 57.82), "Kenenisa Bekele"),
           10000: (mmss(27, 1.17), "Kenenisa Bekele")},
    2012: {100: (9.63, "Usain Bolt"), 200: (19.32, "Usain Bolt"),
           400: (43.94, "Kirani James"), 800: (mmss(1, 40.91), "David Rudisha"),
           1500: (mmss(3, 34.08), "Taoufik Makhloufi"),
           5000: (mmss(13, 41.66), "Mo Farah"), 10000: (mmss(27, 30.42), "Mo Farah")},
    2016: {100: (9.81, "Usain Bolt"), 200: (19.78, "Usain Bolt"),
           400: (43.03, "Wayde van Niekerk"), 800: (mmss(1, 42.15), "David Rudisha"),
           1500: (mmss(3, 50.00), "Matthew Centrowitz"),
           5000: (mmss(13, 3.30), "Mo Farah"), 10000: (mmss(27, 5.17), "Mo Farah")},
    2020: {100: (9.80, "Marcell Jacobs"), 200: (19.62, "Andre De Grasse"),
           400: (43.85, "Steven Gardiner"), 800: (mmss(1, 45.06), "Emmanuel Korir"),
           1500: (mmss(3, 28.32), "Jakob Ingebrigtsen"),
           5000: (mmss(12, 58.15), "Joshua Cheptegei"),
           10000: (mmss(27, 43.22), "Selemon Barega")},
    2024: {100: (9.784, "Noah Lyles"), 200: (19.46, "Letsile Tebogo"),
           400: (43.40, "Quincy Hall"), 800: (mmss(1, 41.19), "Emmanuel Wanyonyi"),
           1500: (mmss(3, 27.65), "Cole Hocker"),
           5000: (mmss(13, 13.66), "Jakob Ingebrigtsen"),
           10000: (mmss(26, 43.14), "Joshua Cheptegei")},
}

# Men's outdoor world records at the seven Olympic distances: a snapshot of the ratified
# list checked on 2026-10-04 (World Athletics; none pending ratification on that date).
# (distance m, time s, athlete, year). Records are set under the model's own objective,
# minimum time, which is what makes them the chapter's validation data.
WORLD_RECORDS = [
    (100.0, 9.58, "Usain Bolt", 2009),
    (200.0, 19.19, "Usain Bolt", 2009),
    (400.0, 43.03, "Wayde van Niekerk", 2016),
    (800.0, mmss(1, 40.91), "David Rudisha", 2012),
    (1500.0, mmss(3, 26.00), "Hicham El Guerrouj", 1998),
    (5000.0, mmss(12, 35.36), "Joshua Cheptegei", 2020),
    (10000.0, mmss(26, 11.00), "Joshua Cheptegei", 2020),
]


def olympic_flat():
    """Every Olympic final as (year, distance, time, athlete), sorted."""
    return [(y, float(d), float(t), who)
            for y in sorted(OLYMPIC_RESULTS)
            for d, (t, who) in sorted(OLYMPIC_RESULTS[y].items())]


def olympic_bests():
    """The fastest winning time at each distance, as (distance, time, year, athlete)."""
    best = {}
    for y, d, t, who in olympic_flat():
        if d not in best or t < best[d][0]:
            best[d] = (t, y, who)
    return [(d, *best[d]) for d in sorted(best)]


def fmt_time(sec):
    """Seconds as 'm:ss.ss' past a minute, else 'ss.ss'."""
    if sec < 60:
        return f"{sec:.2f}"
    m = int(sec // 60)
    return f"{m}:{sec - 60 * m:05.2f}"


# --------------------------------------------------------------------------------------
# The sprint (f = F from rest) and the critical time
# --------------------------------------------------------------------------------------

def v_dash(t, tau, F):
    """Speed at time t under maximal force from rest, Keller (3.2)."""
    return F * tau * (1.0 - np.exp(-t / tau))


def D_dash(T, tau, F):
    """Distance covered by time T under maximal force from rest, Keller (3.4)."""
    return F * tau**2 * (T / tau + np.exp(-T / tau) - 1.0)


def int_v2_dash(t, tau, F):
    """The integral of v_dash^2 from 0 to t, in closed form."""
    e1 = np.exp(-t / tau)
    return F**2 * tau**2 * (t - 2.0 * tau * (1.0 - e1) + 0.5 * tau * (1.0 - e1**2))


def E_dash(t, tau, F, sigma, E0):
    """Oxygen store at time t while sprinting flat out, Keller (3.3)."""
    return E0 + sigma * t - v_dash(t, tau, F)**2 / 2.0 - int_v2_dash(t, tau, F) / tau


def Tc_of(tau, F, sigma, E0):
    """The critical time: the store runs out while sprinting flat out (E_dash = 0)."""
    return brentq(E_dash, 1e-3, 2000.0, args=(tau, F, sigma, E0))


def Dc_of(tau, F, sigma, E0):
    """The critical distance: the longest race run flat out from start to finish."""
    return D_dash(Tc_of(tau, F, sigma, E0), tau, F)


def Tstar_of(tau, F, sigma, E0):
    """The race time above which all three arcs are present, Keller (3.16).

    For Tc <= T <= T* the optimum has no cruise: the runner sprints until the store is empty
    at t1 = t2 = Tc and fades to the line.
    """
    Tc = Tc_of(tau, F, sigma, E0)
    inner = 1.0 - sigma / (F**2 * tau) * (1.0 - np.exp(-Tc / tau)) ** -2
    return Tc + tau * (np.log(2.0) - 0.5 * np.log(inner))


# --------------------------------------------------------------------------------------
# Long races: sprint, cruise, fade, reduced to one unknown
# --------------------------------------------------------------------------------------

def t2_of_t1(t1, T, tau, F, sigma, E0):
    """The time the store runs out, given the sprint ends at t1 and the cruise is constant.

    E(t2) = 0 is linear in t2 because v is constant on the cruise, so t2 has a closed form.
    Valid when the cruise speed exceeds sqrt(sigma tau), i.e. the store is being drained.
    """
    v1 = v_dash(t1, tau, F)
    I1 = int_v2_dash(t1, tau, F)
    num = v1**2 / 2.0 + I1 / tau - E0 - v1**2 * t1 / tau
    den = sigma - v1**2 / tau
    return num / den


def D_arc3(t1, t2, T, tau, F, sigma):
    """Distance covered on the fade, t2 <= t <= T, in closed form.

    On the fade v^2 = a + b exp(-2 (t - t2) / tau) with a = sigma tau and b = v1^2 - a
    (Keller 3.7). The antiderivative of u(s) = sqrt(a + b e^{-2s/tau}) is
    tau [ -u + sqrt(a) artanh(sqrt(a)/u) ], which is where Keller's inverse hyperbolic
    tangents come from; written out, the integral from 0 to s is
        tau (u(0) - u(s)) + sqrt(a) [ tau log((sqrt(a) + u(s)) / (sqrt(a) + u(0))) + s ],
    a form that stays accurate when u is close to sqrt(a) and also covers b <= 0.
    Accepts arrays.
    """
    t1, t2 = np.asarray(t1, dtype=float), np.asarray(t2, dtype=float)
    a = sigma * tau
    b = v_dash(t1, tau, F) ** 2 - a
    s = T - t2
    u0 = np.sqrt(a + b)
    us = np.sqrt(a + b * np.exp(-2.0 * s / tau))
    root_a = np.sqrt(a)
    return tau * (u0 - us) + root_a * (tau * np.log((root_a + us) / (root_a + u0)) + s)


def D_of_t1(t1, T, tau, F, sigma, E0):
    """Total distance in time T if the sprint ends at t1 (vectorised over t1).

    When the cruise speed is below sqrt(sigma tau) the store never empties, so t2 = T and
    there is no fade; t2 is also clipped into [t1, T] so the function is defined
    everywhere. The result is piecewise smooth: the branch at v1^2 = sigma tau and the clip
    at t2 = T are the boundaries, which is why solve_race scans a grid before polishing.
    """
    t1 = np.asarray(t1, dtype=float)
    v1 = v_dash(t1, tau, F)
    draining = v1**2 > sigma * tau * (1.0 + 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        t2 = np.where(draining, t2_of_t1(t1, T, tau, F, sigma, E0), T)
    t2 = np.clip(t2, t1, T)
    return D_dash(t1, tau, F) + v1 * (t2 - t1) + D_arc3(t1, t2, T, tau, F, sigma)


def solve_race(T, tau, F, sigma, E0, ngrid=60):
    """The optimal race of duration T: returns (D, t1, t2).

    For T <= Tc the whole race is a sprint (t1 = t2 = T). Otherwise D(t1) is scanned on a
    grid, since its maximum is a narrow peak beside a long flat plateau, and the best cell
    is polished with a bounded one-dimensional search. For Tc <= T <= T* the corner
    t1 = t2 = Tc (sprint until empty, then fade) wins and is returned.
    """
    Tc = Tc_of(tau, F, sigma, E0)
    if T <= Tc:
        return D_dash(T, tau, F), T, T
    grid = np.linspace(1e-3, Tc - 1e-6, ngrid)
    Dg = D_of_t1(grid, T, tau, F, sigma, E0)
    i = int(np.argmax(Dg))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, ngrid - 1)]
    res = minimize_scalar(lambda a: -float(D_of_t1(a, T, tau, F, sigma, E0)),
                          bounds=(lo, hi), method="bounded",
                          options=dict(xatol=1e-9))
    t1, D3 = float(res.x), -float(res.fun)
    t2 = float(t2_of_t1(t1, T, tau, F, sigma, E0))
    D_pin = float(D_dash(Tc, tau, F) + D_arc3(Tc, Tc, T, tau, F, sigma))
    # inside the two-arc window the plateau is flat to a millimetre, so an interior point
    # can tie the corner to rounding; the corner is then the answer
    if t2 < t1 or D_pin >= D3 - 1e-6:
        return D_pin, Tc, Tc
    return D3, t1, t2


def D_of_T(T, tau, F, sigma, E0):
    """Distance of the optimal race of duration T."""
    return solve_race(T, tau, F, sigma, E0)[0]


def T_of_D(D, tau, F, sigma, E0, xtol=1e-5):
    """Time of the optimal race over distance D: inverts the monotone map D(T)."""
    return brentq(lambda T: D_of_T(T, tau, F, sigma, E0) - D, 0.1, 1.2 * D / 3.0 + 20.0,
                  xtol=xtol)


def T_dash_of_D(D, tau, F):
    """Time to cover D flat out from rest: inverts Keller (3.4). Valid for D <= Dc."""
    return brentq(lambda T: D_dash(T, tau, F) - D, 1e-3, 1.2 * D / 3.0 + 20.0)


def velocity_profile(T, tau, F, sigma, E0, n=2000):
    """The optimal v(t) sampled at n points: returns (t, v, t1, t2)."""
    _, t1, t2 = solve_race(T, tau, F, sigma, E0)
    t = np.linspace(0.0, T, n)
    v1 = v_dash(t1, tau, F)
    v3 = np.sqrt(sigma * tau + (v1**2 - sigma * tau) * np.exp(-2.0 * (t - t2) / tau))
    v = np.where(t <= t1, v_dash(t, tau, F), np.where(t <= t2, v1, v3))
    return t, v, t1, t2


def fade_deficit(T, tau, F, sigma, E0):
    """How much slower the runner crosses the line than the cruise speed, v(t2) - v(T)."""
    _, t1, t2 = solve_race(T, tau, F, sigma, E0)
    v1 = v_dash(t1, tau, F)
    vT = np.sqrt(sigma * tau + (v1**2 - sigma * tau) * np.exp(-2.0 * (T - t2) / tau))
    return v1 - vT


def rel_error(T_theory, T_data):
    """Keller's error convention, (T_theory - T_data) / T_data, in percent.

    Positive means the runner beat the theory.
    """
    return 100.0 * (T_theory - T_data) / T_data

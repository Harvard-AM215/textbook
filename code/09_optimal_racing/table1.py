"""Reproduce Keller's Table 1 with his published constants.

Companion to Chapter 9 Worked Example 1. For each of the 22 records in Keller (1973),
Table 1, prints the record time, the time the model gives for that distance, the error in
Keller's convention (theory minus record, as a percentage of the record; positive means
the runner beat the theory), the model's average speed, the length of the opening sprint
t1 and the length of the closing fade T - t2. Dashes (D < Dc) have t1 = T and no fade.

    python3 table1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keller import (  # noqa: E402
    KELLER_PUBLISHED,
    KELLER_RECORDS,
    KELLER_SI,
    Dc_of,
    T_of_D,
    Tc_of,
    Tstar_of,
    fmt_time,
    rel_error,
    solve_race,
)


def table_rows(params=KELLER_SI):
    """One dict per record: label, distance, record, theory, error, speed, t1, fade."""
    Dc = Dc_of(**params)
    rows = []
    for label, D, T_rec in KELLER_RECORDS:
        T_th = T_of_D(D, **params)
        _, t1, t2 = solve_race(T_th, **params)
        rows.append(dict(label=label, D=D, record=T_rec, theory=T_th,
                         error=rel_error(T_th, T_rec), speed=D / T_th, t1=t1,
                         fade=(T_th - t2) if D > Dc else None))
    return rows


def main(params=KELLER_SI):
    p = KELLER_PUBLISHED
    print(f"Keller's constants: tau = {p['tau']} s, F = {p['F']} m/s^2, "
          f"sigma = {p['sigma']} cal/(kg s) = {params['sigma']:.1f} W/kg, "
          f"E0 = {p['E0']:.0f} cal/kg = {params['E0']:.0f} J/kg")
    print(f"Tc = {Tc_of(**params):.2f} s   Dc = {Dc_of(**params):.1f} m   "
          f"T* = {Tstar_of(**params):.2f} s")
    print()
    print(f"{'race':>8} {'record':>8} {'theory':>8} {'error':>7} {'D/T':>6} "
          f"{'t1':>6} {'T-t2':>6}")
    rows = table_rows(params)
    for r in rows:
        fade = f"{r['fade']:6.2f}" if r["fade"] is not None else "     -"
        print(f"{r['label']:>8} {fmt_time(r['record']):>8} {fmt_time(r['theory']):>8} "
              f"{r['error']:+6.1f}% {r['speed']:6.2f} {r['t1']:6.2f} {fade}")
    worst = max(rows, key=lambda r: abs(r["error"]))
    print()
    print(f"largest |error| = {abs(worst['error']):.1f}% ({worst['label']}); "
          f"Keller reports 3.1% at 6 miles")
    return rows


if __name__ == "__main__":
    main()

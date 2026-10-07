"""Reproduce Chapter 9's numerical answers to Exercises 4 and 5.

Run with: python exercise_checks.py
"""

from fit_records import modern_refit
from keller import (
    KELLER_SI, D_arc3, Dc_of, T_of_D, fade_deficit, solve_race, v_dash,
)


def main():
    p = KELLER_SI
    print("Exercise 4: distance, speed deficit (m/s), extra distance (m)")
    for distance in (400, 1500, 10000):
        time = T_of_D(distance, **p)
        _, t1, t2 = solve_race(time, **p)
        speed = v_dash(t1, p["tau"], p["F"])
        fade_distance = D_arc3(t1, t2, time, p["tau"], p["F"], p["sigma"])
        extra = speed * (time - t2) - fade_distance
        print(f"{distance:5d}  {fade_deficit(time, **p):.2f}  {extra:.2f}")

    print("\nExercise 5: reaction (s), F (m/s^2), increase (%), Dc (m), E0/sigma (s)")
    for reaction in (0.10, 0.15, 0.20):
        fit, _ = modern_refit(verbose=False, reaction_time=reaction)
        increase = 100 * (fit["F"] / p["F"] - 1)
        print(f"{reaction:.2f}  {fit['F']:.2f}  {increase:.1f}  "
              f"{Dc_of(**fit):.1f}  {fit['E0'] / fit['sigma']:.1f}")


if __name__ == "__main__":
    main()

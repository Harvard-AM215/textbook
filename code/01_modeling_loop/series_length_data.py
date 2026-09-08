"""Derive the historical seven-game rate quoted in Chapter 1, from committed data.

Chapter 1 validates the IID best-of-7 model against one external number: the share of
World Series that needed all seven decisive games. That number has been wrong twice --
first a denominator counting *seasons* rather than *series* (119, corrected to 115), then
a 37% share that was never sourced and is not what the record shows. This script exists so
the third version is checkable rather than asserted: it recomputes every figure the chapter
quotes, from `data/public/world_series.csv`, and prints the exclusions it applied.

Run it:

    python3 series_length_data.py

Provenance of the CSV: Wikipedia, "List of World Series champions" (retrieved 2026-08-18),
parsed from the article's raw wikitext, cross-checked against Baseball-Reference's
postseason series table. Series results are uncopyrightable facts, which is why this file
may live under `data/public/` while the market-data CSVs beside it may not -- see
`data/README.md`.

The parse is validated against a primary source: restricted to 1905-1951 it must reproduce
Mosteller (1952, JASA 47(259):355-380) Table 5 exactly -- 44 series splitting 9/13/11/11.
That assertion is the reason to trust the rest of the output.
"""

import csv
import pathlib
from collections import Counter

# Best-of-nine, so their lengths are not comparable to a best-of-7 model.
BEST_OF_NINE = {1903, 1919, 1920, 1921}
# 1904: the NL champion declined to play. 1994: the players' strike cancelled the postseason.
NOT_PLAYED = {1904, 1994}

DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "public" / "world_series.csv"


def load(path=DATA):
    """Year -> series result string ('4-3', '4-0-1', ...), omitting years with no series."""
    with path.open() as fh:
        return {
            int(row["year"]): row["result"]
            for row in csv.DictReader(fh)
            if row["result"]
        }


def decisive_games(result):
    """Games that produced a decision.

    A tied game adds a game *played* without adding a decision, so 1912's '4-3-1' is seven
    decisive games across eight games played. That distinction is why the chapter says 40
    series needed a seventh decisive game while only 39 were literally seven games long.
    """
    parts = [int(x) for x in result.split("-")]
    return parts[0] + parts[1]


def lengths(results, lo, hi):
    """Counter of decisive-game counts over best-of-7 series in [lo, hi]."""
    return Counter(
        decisive_games(r)
        for y, r in results.items()
        if lo <= y <= hi and y not in BEST_OF_NINE
    )


def main():
    results = load()

    # --- validate the parse against Mosteller's published table, before trusting it ---
    m = lengths(results, 1905, 1951)
    got = (sum(m.values()), m[4], m[5], m[6], m[7])
    assert got == (44, 9, 13, 11, 11), (
        f"parse does not reproduce Mosteller (1952) Table 5: got {got}, expected "
        "(44, 9, 13, 11, 11). The CSV or the parser has drifted -- do not trust the "
        "numbers below until this passes."
    )
    print(f"Mosteller (1952) Table 5, 1905-1951: n=44, {m[4]}/{m[5]}/{m[6]}/{m[7]}  [reproduced]")

    # --- the window Chapter 1 quotes ---
    lo, hi = 1905, 2023
    c = lengths(results, lo, hi)
    n = sum(c.values())
    sevens = c[7]

    print(f"\nBest-of-7 World Series, {lo}-{hi}")
    print(f"  excluded, best-of-nine : {sorted(BEST_OF_NINE)}")
    print(f"  excluded, not played   : {sorted(NOT_PLAYED)}")
    print(f"  series counted         : {n}")
    print(f"  length distribution    : 4:{c[4]}  5:{c[5]}  6:{c[6]}  7:{c[7]}")
    print(f"  seven decisive games   : {sevens}/{n} = {sevens / n:.4f} = {sevens / n:.1%}")

    tied = sorted(y for y, r in results.items() if r.count("-") == 2 and lo <= y <= hi)
    eight = [y for y in tied if decisive_games(results[y]) == 7]
    print(f"  series with a tie game : {tied}")
    print(f"  of the {sevens}, played across eight games: {eight} "
          f"(so {sevens - len(eight)} were literally seven games)")

    # --- the era split, for the question of whether 1969 changed anything ---
    print("\nEra split (divisional play and the LCS began in 1969)")
    for a, b in [(1905, 1968), (1969, 2023)]:
        e = lengths(results, a, b)
        ne = sum(e.values())
        print(f"  {a}-{b}: n={ne:<4} {e[4]}/{e[5]}/{e[6]}/{e[7]}"
              f"   seven-game rate {e[7] / ne:.1%}")


if __name__ == "__main__":
    main()

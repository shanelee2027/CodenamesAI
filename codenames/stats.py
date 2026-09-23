"""Small statistical helpers shared by the analysis scripts.

Standard library only, so a script that reports a p-value does not pull in
scipy for four formulas. Each one used to be copied into the script that
needed it.
"""

from __future__ import annotations

import math
import random


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs: list[float]) -> float:
    """Sample standard deviation (n - 1)."""
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval on a proportion. Wilson rather than the normal
    approximation, because several runs are only 40 games."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test against p = 0.5 -- the sign test, when
    `k` of `n` decisive pairs went one way."""
    if n == 0:
        return 1.0

    def pmf(i: int) -> float:
        return math.comb(n, i) * 0.5 ** n

    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12))


def fisher_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on [[a, b], [c, d]]."""
    n = a + b + c + d

    def p(x: int) -> float:
        return (math.comb(a + b, x) * math.comb(c + d, a + c - x)) / math.comb(n, a + c)

    lo = max(0, a + c - (c + d))
    hi = min(a + b, a + c)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs + 1e-12))


def boot_diff(a: list[float], b: list[float], reps: int, rng: random.Random) -> tuple[float, float]:
    """Bootstrap 95% interval on mean(b) - mean(a), resampling each group."""
    ds = sorted(mean([rng.choice(b) for _ in b]) - mean([rng.choice(a) for _ in a])
                for _ in range(reps))
    return ds[int(0.025 * reps)], ds[int(0.975 * reps)]


def perm_p(a: list[float], b: list[float], reps: int, rng: random.Random) -> float:
    """Two-sided permutation p-value for a difference in means."""
    obs = abs(mean(b) - mean(a))
    pool, na = a + b, len(a)
    hits = 0
    for _ in range(reps):
        rng.shuffle(pool)
        hits += abs(mean(pool[na:]) - mean(pool[:na])) >= obs - 1e-12
    return (hits + 1) / (reps + 1)


def n_for_power(delta: float, s: float) -> float:
    """Samples per arm for 80% power at alpha 0.05, two-sided:
    2 (z_a + z_b)^2 s^2 / delta^2."""
    if not delta or not s or math.isnan(s):
        return float("inf")
    return 2 * (1.96 + 0.8416) ** 2 * s ** 2 / delta ** 2

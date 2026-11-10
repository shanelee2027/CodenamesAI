"""Fit the naive Gaussian model's noise sigma to gpt-oss rankings by maximum likelihood.

The model (docs/clue-selection-theory.tex, thesis Section 3): the guesser sees
board word x as zhat_x = z_x + eps_x with eps_x ~ N(0, sigma^2) independent, and
ranks the unrevealed words by zhat. Every cached gpt-oss ranking of a generated
position is one data point: (clue, unrevealed candidates, ranking). Its
log-likelihood under sigma is the log-probability that the model puts the
observed top m words first, in that order:

    P(x_1 > x_2 > ... > x_m > every other candidate)

computed exactly as a one-dimensional integral. Writing f_j, F_j for the
density and CDF of zhat_{x_j}, and Q_j(s) = P(zhat_{x_1} > ... > zhat_{x_j} > s),

    Q_0(s) = 1,   Q_j(s) = int_s^inf f_j(u) Q_{j-1}(u) du,
    P      = int f_m(s) Q_{m-1}(s) prod_{y not in top m} F_y(s) ds,

so on a grid in s each Q_j is a reverse cumulative sum.

**How deep each ranking is used.** A request only had to name at least
max(k + 2, 3) words (codenames/guessers/llm.py::required_named, k = the number
in the prompt, 1-4 here); any word it left out was appended in board order and
cached as part of the ranking. So only the first k + 2 positions are certainly
gpt-oss's own picks, and a data point is used to depth at most k + 2. Deeper
than that would fit the backfill order; the far tail (how twenty unrelated
words are ordered) is also where equal noise for every word is least
plausible, so the model is not asked to explain it.

Three fits are reported: (1) every point at its own guaranteed depth; (2) for
each depth m = 1..5, the points deep enough for it; (3) the same on the k = 4
points only, the one subset deep enough for every m, so the trend in sigma
with m is not confounded with which points are included.

Positions are re-planned exactly as scripts/data/collect_listener_data.py
plans them (same generator, same seeds), so the data is the `--vocab
online_train` collection: boards dealt from the online game's 550 training
words. z is standardized per clue over the online game's 800 words
(cache/clue_stats_online.npz).

Free: reads cached responses only, never calls an API.

Usage:
    python scripts/pipeline/fit_gaussian_sigma.py --n 17000 --space numberbatch
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, ndtr

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from codenames.board import load_online_split  # noqa: E402
from codenames.clue_stats import ClueStats  # noqa: E402
from codenames.listener_training import ONLINE_TRAIN_BASE  # noqa: E402
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor  # noqa: E402


def load_collector():
    """The collection script as a module, so positions are planned by the very
    same code that bought them (it lives under scripts/, not a package)."""
    path = PROJECT_ROOT / "scripts" / "data" / "collect_listener_data.py"
    spec = importlib.util.spec_from_file_location("collect_listener_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def top_m_loglik(z: np.ndarray, top: list[int], sigma: float, m: int, steps_per_sigma: int = 40) -> float:
    """log P(the candidates at indices top[0..m-1] are the model's first m, in order).

    z: (n,) standardized similarities of the unrevealed candidates."""
    lo, hi = z.min() - 7 * sigma, z.max() + 7 * sigma
    n_grid = int(np.ceil((hi - lo) / sigma * steps_per_sigma))
    s = np.linspace(lo, hi, n_grid + 1)
    ds = s[1] - s[0]
    u = (s[None, :] - z[:, None]) / sigma                      # (n, grid)
    dens = np.exp(-0.5 * u**2) / (np.sqrt(2 * np.pi) * sigma)  # f_x(s)
    head = top[:m]
    rest = np.ones(len(z), dtype=bool)
    rest[head] = False
    log_rest = log_ndtr(u[rest]).sum(axis=0)                   # log prod_{y in rest} F_y(s)
    # Q_{j}(s) for j = 1..m-1, by reverse cumulative sums of f_j * Q_{j-1}.
    q = np.ones_like(s)
    for j in head[:-1]:
        g = dens[j] * q * ds
        q = np.cumsum(g[::-1])[::-1] - 0.5 * g                 # integral over u > s (half the own cell)
    integrand = dens[head[-1]] * q * np.exp(log_rest) * ds
    p = integrand.sum()
    return float(np.log(max(p, 1e-300)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=17000, help="positions to plan, as in the collection run")
    ap.add_argument("--space", default="numberbatch")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--provider", default="deepinfra")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--sigmas", default="0.4:3.0:0.1", help="grid lo:hi:step")
    ap.add_argument("--max-m", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples over positions")
    ap.add_argument("--out", default=None, help="write the results as JSON here")
    args = ap.parse_args()

    col = load_collector()
    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    plan_stats = ClueStats.load(DEFAULT_CACHE_DIR)                       # what the collector planned with
    z_stats = ClueStats.load(DEFAULT_CACHE_DIR, stem="clue_stats_online")  # z over the 800 online words
    pool = col.build_clue_pool(plan_stats, 10.0)
    vocab = load_online_split()[0]
    plan_space = sims.spaces.index("numberbatch")                        # the collector's default --space

    from codenames.guessers.openai_compat import OpenAICompatGuesser
    g = OpenAICompatGuesser(model=args.model, provider=args.provider, reasoning_effort=args.effort,
                            cache_path=col.DB)

    si = z_stats.space_index(args.space)
    clue_index = {w: i for i, w in enumerate(z_stats.clue_words)}
    data, skipped = [], {"uncached": 0, "nan": 0}
    for i in range(args.n):
        p = col.make_position(i, sims, plan_stats, pool, plan_space, col.DEFAULT_MIX, vocab, ONLINE_TRAIN_BASE)
        if p is None:
            continue
        ranking = g._disk_cache.get(g.cache_model_id, p["clue"], tuple(p["candidates"]), p["k"])
        if ranking is None:
            skipped["uncached"] += 1
            continue
        cands = p["candidates"]
        ci = clue_index[p["clue"]]
        idx = [sims.board_index[w.lower()] for w in cands]
        raw = np.asarray(sims.tensor[ci, idx, si], dtype=np.float64)
        z = (raw - z_stats.mean[ci, si]) / z_stats.std[ci, si]
        if not np.all(np.isfinite(z)):
            skipped["nan"] += 1
            continue
        pos = {w: j for j, w in enumerate(cands)}
        depth = min(max(p["k"] + 2, 3), args.max_m, len(cands) - 1)
        top = [pos[w] for w in ranking[:depth]]
        data.append((z, top, p["k"]))
    print(f"data points: {len(data)}   skipped: {skipped}", flush=True)

    lo, hi, step = (float(x) for x in args.sigmas.split(":"))
    sigmas = np.round(np.arange(lo, hi + step / 2, step), 6)
    results = {"n": len(data), "space": args.space, "sigmas": sigmas.tolist(), "fits": {}}
    rng = np.random.default_rng(0)
    # ll[m][i] = log-lik of point i at depth m over the sigma grid (None if too shallow).
    ll = {m: [None] * len(data) for m in range(1, args.max_m + 1)}
    for i, (z, top, _k) in enumerate(data):
        for m in range(1, len(top) + 1):
            ll[m][i] = np.array([top_m_loglik(z, top, s, m) for s in sigmas])
        if (i + 1) % 2000 == 0:
            print(f"  likelihoods {i + 1}/{len(data)}", flush=True)

    def fit(name: str, rows: np.ndarray) -> None:
        total = rows.sum(axis=0)
        best = int(np.argmax(total))
        boots = [sigmas[np.argmax(rows[rng.integers(0, len(rows), len(rows))].sum(axis=0))]
                 for _ in range(args.boot)]
        ci95 = np.percentile(boots, [2.5, 97.5]).tolist()
        results["fits"][name] = {"n": len(rows), "sigma": float(sigmas[best]), "ci95": ci95,
                                 "mean_loglik": float(total[best] / len(rows)),
                                 "mean_loglik_by_sigma": (total / len(rows)).tolist()}
        print(f"{name:>22}: n={len(rows):6d}  sigma={sigmas[best]:.2f}  "
              f"95% bootstrap [{ci95[0]:.2f}, {ci95[1]:.2f}]  mean log-lik {total[best] / len(rows):.4f}",
              flush=True)

    fit("own depth", np.array([ll[len(top)][i] for i, (_z, top, _k) in enumerate(data)]))
    for m in range(1, args.max_m + 1):
        rows = [r for r in ll[m] if r is not None]
        if rows:
            fit(f"depth {m}", np.array(rows))
    for m in range(1, args.max_m + 1):
        rows = [ll[m][i] for i, (_z, _t, k) in enumerate(data) if k == 4 and ll[m][i] is not None]
        if rows:
            fit(f"k=4 points, depth {m}", np.array(rows))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

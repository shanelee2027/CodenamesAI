"""Estimate a listener's `sigma` from rankings it has already produced.

`expected_words` models the guesser as perceiving each board word as
`z_w + eps_w`, `eps ~ N(0, sigma)`, and then working down its own perceived
order (see codenames/spymasters/expected_words.py). `sigma` is the single
parameter of that model, in z units. Until now it has never been measured:
it was picked from the announced-number distribution on fresh boards, and a
per-turn reward proxy later turned out to rank candidate values *backwards*
against real game outcomes (docs/log.md).

But every LLM ranking this project has ever paid for is cached in
cache/llm_store.db, and a ranking is exactly what that model predicts. So
sigma can be estimated directly from data already bought, with no new API
calls at all.

**The estimator.** Treat an observed ranking as a draw from the model above.
That is Thurstone's Case V with *known* utilities -- the z-scores are not
free parameters, they come from the similarity tensor -- which leaves sigma
as the only unknown. What is scored is the listener's top pick: the
probability that the word it chose beat every other word on the board.
Conditioning on that word's own noise makes the rest independent, leaving a
one-dimensional Gaussian integral,

    P(a beats all of S) = INT phi(u) prod_{b in S} Phi(u + (z_a - z_b)/sigma) du

(substituting `eps_a = sigma * u`), evaluated on a fixed grid in `u`. The
log-likelihood is summed over positions and maximised over sigma.

**Only the top pick is scored, and that is deliberate.** Conditioning on
the winner's own noise makes every other word independent, so the top-1
probability above is exact. Extending it down the ranking is not: the
familiar "probability this is the max of what remains, multiplied along the
ranking" factorisation is exact only under Gumbel noise (that is Luce's
axiom), and this model is Gaussian. Both cheap extensions were implemented
and measured against planted sigmas on simulated listeners:

    true sigma   exact top-1   sequential   pairwise (top 3)
          0.50         0.493        --            0.415
          1.00         1.003        --            0.783
          2.00         2.004       ~2.9           1.547
          4.00         4.121        --            2.927

The sequential product drifts high and the pairwise composite drifts low --
each pair is conditioned on the winner having already won, which is a
selection effect, not noise. Only the exact top-1 likelihood is used here.
It costs information per position, and that is paid for with positions:
there are several thousand cached.

The top pick is also the part of a ranking worth trusting. Nothing in the
game ever reads the tail of a 25-word ordering, and a model asked to order
words it considers equally irrelevant is answering a question it was not
really posed.

**What this does NOT measure.** Two cautions, both of which matter for how
the number may honestly be used:

1. *Noise versus mismatch.* The model says deviations from numberbatch's
   z-order are independent Gaussian noise. Where an LLM instead disagrees
   with numberbatch *systematically* -- it knows a sense of a word the
   embedding does not -- that disagreement is not noise, but the fit has
   nowhere else to put it and inflates sigma to absorb it. `calibration`
   below is what exposes this: a well-specified sigma reproduces the
   observed spread of the listener's choices, an inflated one does not.

2. *Descriptive is not optimal.* This estimates the sigma that best
   *describes* a listener. The sigma a spymaster should *play* is a separate
   question -- being deliberately pessimistic about a listener can score
   better than modelling it accurately -- and docs/log.md already records
   one proxy that ordered sigma backwards against real outcomes. These two
   numbers should be reported side by side, never substituted for one
   another.

Conditional on the clue distribution, too: the cached rankings are responses
to clues that particular spymasters chose, not a random sample of the
vocabulary. That is arguably the right conditioning -- it is the distribution
the system actually produces -- but it is a conditioning, and it moves if the
spymaster does.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

# Grid for the standard-normal integral over the winner's own noise. +-8 sd
# covers the mass to ~1e-15; 129 points is where the fitted sigma stops
# moving in the 4th decimal on this data.
_U_LO, _U_HI, _U_N = -8.0, 8.0, 129


@dataclass(frozen=True)
class Observation:
    """One cached ranking, already resolved to z-scores.

    `z` holds the z-score of each candidate word IN THE ORDER THE LISTENER
    RANKED THEM, so the observed permutation is simply `0, 1, 2, ...` and the
    likelihood never has to carry a separate index array.
    """

    clue: str
    z: np.ndarray  # (n_candidates,) z-scores, in the listener's ranked order
    number: int | None


def _pack(observations: list[Observation]) -> tuple[np.ndarray, np.ndarray]:
    """Ragged observations -> a padded `(P, n_max)` array plus a validity
    mask, so every position is scored in one vectorised pass instead of a
    Python loop over several thousand of them."""
    n_max = max(len(o.z) for o in observations)
    z = np.zeros((len(observations), n_max), dtype=np.float64)
    mask = np.zeros((len(observations), n_max), dtype=bool)
    for i, o in enumerate(observations):
        z[i, : len(o.z)] = o.z
        mask[i, : len(o.z)] = True
    return z, mask


def log_likelihood(z: np.ndarray, mask: np.ndarray, sigma: float) -> float:
    """Exact total log-likelihood of the observed top picks at this `sigma`.

    `z` is `(P, n_max)` with column 0 the word the listener chose and the
    rest the field it beat; `mask` marks real candidates.

        P(a wins) = INT phi(u) prod_b Phi(u + (z_a - z_b)/sigma) du

    Masked-out columns contribute `log 1 = 0`, so ragged boards need no
    special case.
    """
    if sigma <= 0:
        return -np.inf
    u = np.linspace(_U_LO, _U_HI, _U_N)
    log_w = -0.5 * u**2 - 0.5 * np.log(2 * np.pi) + np.log((_U_HI - _U_LO) / (_U_N - 1))

    alive = mask[:, 0] & mask[:, 1:].any(axis=1)
    if not alive.any():
        return -np.inf
    d = (z[alive, :1] - z[alive, 1:]) / sigma  # (P, m)
    terms = log_ndtr(u[None, :, None] + d[:, None, :])  # (P, U, m)
    terms = np.where(mask[alive, 1:][:, None, :], terms, 0.0)
    inner = terms.sum(axis=2) + log_w[None, :]  # (P, U)
    m = inner.max(axis=1, keepdims=True)
    return float((m[:, 0] + np.log(np.exp(inner - m).sum(axis=1))).sum())


def fit_sigma(
    observations: list[Observation],
    lo: float = 0.05,
    hi: float = 20.0,
    tol: float = 1e-3,
) -> tuple[float, float]:
    """Maximum-likelihood `sigma` and its log-likelihood.

    Golden-section on a unimodal 1-D problem: as `sigma -> 0` the model
    insists the listener follows z-order exactly, as `sigma -> inf` it
    predicts a uniform random pick, and the data pin a single interior
    optimum between those.
    """
    z, mask = _pack(observations)
    phi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = log_likelihood(z, mask, c), log_likelihood(z, mask, d)
    while b - a > tol:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = log_likelihood(z, mask, c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = log_likelihood(z, mask, d)
    best = (a + b) / 2
    return best, log_likelihood(z, mask, best)


def profile(observations: list[Observation], sigmas: np.ndarray) -> np.ndarray:
    """Log-likelihood at each `sigma` -- for a curve, and for a likelihood
    interval (a drop of 1.92 from the peak is the 95% chi-square cutoff on
    one parameter)."""
    z, mask = _pack(observations)
    return np.array([log_likelihood(z, mask, s) for s in sigmas])


def likelihood_interval(
    observations: list[Observation], best: float, drop: float = 1.92, span: float = 4.0
) -> tuple[float, float]:
    """95% likelihood interval: the sigmas where the log-likelihood falls
    `drop` below its peak. Reported instead of a standard error because the
    likelihood in sigma is visibly asymmetric -- large sigma is close to
    unidentifiable, small sigma is sharply excluded."""
    z, mask = _pack(observations)
    peak = log_likelihood(z, mask, best)
    target = peak - drop

    def edge(lo: float, hi: float) -> float:
        for _ in range(60):
            mid = (lo + hi) / 2
            if log_likelihood(z, mask, mid) < target:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    return edge(best, max(best / span, 1e-3)), edge(best, best * span)


def calibration(observations: list[Observation], sigma: float, rng_seed: int = 0) -> dict[str, float]:
    """Does this `sigma` reproduce what the listener actually did?

    The fit can always return *some* sigma; that does not make the model
    right. The sharpest check that costs nothing: where did the listener's
    top pick sit in numberbatch's own z-order? Under the model, simulating
    `z + N(0, sigma)` and taking the argmax should land on the same z-rank
    distribution. If the listener is systematically choosing words the
    embedding ranks low -- knowledge the embedding lacks, not noise -- the
    simulated mean rank will come out lower than the observed one however
    sigma is tuned, and that gap is the model's misspecification made
    visible.
    """
    rng = np.random.default_rng(rng_seed)
    obs_rank, sim_rank, obs_top1, sim_top1 = [], [], [], []
    for o in observations:
        n = len(o.z)
        if n < 2:
            continue
        order = np.argsort(-o.z)  # z-order; position 0 in o.z is the listener's pick
        rank_of_pick = int(np.where(order == 0)[0][0])
        obs_rank.append(rank_of_pick)
        obs_top1.append(rank_of_pick == 0)
        noisy = o.z + rng.normal(0.0, sigma, size=n)
        pick = int(np.argmax(noisy))
        sim_rank.append(int(np.where(order == pick)[0][0]))
        sim_top1.append(pick == int(order[0]))
    return {
        "n": float(len(obs_rank)),
        "observed_mean_zrank": float(np.mean(obs_rank)),
        "simulated_mean_zrank": float(np.mean(sim_rank)),
        "observed_top1_rate": float(np.mean(obs_top1)),
        "simulated_top1_rate": float(np.mean(sim_top1)),
    }


def load_observations(
    db_path: Path,
    model: str,
    sims,
    clue_stats,
    space: str = "numberbatch",
    require_number: bool = True,
    min_candidates: int = 4,
) -> list[Observation]:
    """Pull one model's cached rankings and resolve them to z-scores.

    `require_number` keeps only rankings produced for a real turn. A
    `number=None` row came from `score_candidates`, whose prompt omits the
    count -- a different question, so pooling the two would blend two
    listeners.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sql = "SELECT clue, candidates, number, ranking FROM responses WHERE model = ?"
    if require_number:
        sql += " AND number IS NOT NULL"
    rows = conn.execute(sql, (model,)).fetchall()
    conn.close()

    clue_index = {w: i for i, w in enumerate(clue_stats.clue_words)}
    si = clue_stats.space_index(space)
    out: list[Observation] = []
    for clue, candidates_json, number, ranking_json in rows:
        ci = clue_index.get(clue.lower())
        if ci is None:
            continue  # a clue outside the tensor's vocabulary has no z at all
        candidates = json.loads(candidates_json)
        ranking = json.loads(ranking_json)
        if len(candidates) < min_candidates:
            continue
        # Trust the ranking only as far as it is a permutation of the board.
        if sorted(w.lower() for w in ranking) != sorted(w.lower() for w in candidates):
            continue
        try:
            idxs = [sims.board_index[w.lower()] for w in ranking]
        except KeyError:
            continue
        sim = np.asarray(sims.tensor[ci, idxs, si], dtype=np.float64)
        mean = float(clue_stats.mean[ci, si])
        std = float(clue_stats.std[ci, si])
        if not np.isfinite(std) or std <= 0 or not np.all(np.isfinite(sim)):
            continue
        out.append(Observation(clue=clue, z=(sim - mean) / std, number=number))
    return out

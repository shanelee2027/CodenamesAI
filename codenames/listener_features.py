"""Features for a distilled listener: one row per candidate word.

The model this feeds is a conditional logit (McFadden): a shared function
scores every word on the board, a softmax over the board turns those scores
into "which word does the listener pick", and the observed teacher ranking is
the response. So a feature describes a *word in the context of a board*, never
a word alone -- what matters is how w compares to what it is up against.

**Used by training and by inference, and it must be the same code for both.**
If the two paths differ by so much as a normalisation constant, the arena
silently measures a different model than the one that was fitted, and nothing
crashes. Hence one module, one function.

**No feature may see a word's role.** The listener is shown a clue and a list
of words; it does not know which are own, opponent, neutral or assassin. A
role-derived feature would be silent cheating -- the distilled guesser would
look excellent in the arena while being worthless as a model of a listener.
`tests/test_listener_features.py` asserts the signature cannot see roles.

**Tier 1 (this file) is 21 features.** Deliberately small: there are only
~9k choice events available, and each further block should be added as a named
hypothesis and measured, not poured in at the start. See docs/log.md.

Two of them are worth explaining.

`P_MAX_SIGMA` -- the probability that w is the listener's pick under the
Gaussian model `expected_words` already assumes, at three noise levels. This
is the analytic baseline (and sigma~2.06 was measured to be roughly right for
real listeners, see codenames/listener_fit.py), so including it turns the
learned model into a *residual* learner: the Gaussian model supplies a
board-aware prior and the trees only have to capture where it is wrong. That
is a far easier problem than rediscovering a softmax from 9k examples.

`RIVAL_MIN` -- `max over rivals of (min over spaces)`, the strongest rival
that is strong in *every* embedding space. Per-space order statistics cannot
distinguish one rival scoring (3,3,3) from three specialists scoring (3,0,0),
(0,3,0), (0,0,3): both give a top competitor of 3 in every space, but the
first is a genuine threat and the second is not. Taking the min across spaces
before maximising over rivals separates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

# Noise levels for the Gaussian argmax feature. 2.0 brackets the measured
# listener sigma (2.06 for Sonnet, 2.20 for gpt-oss); 1 and 3 give the trees
# a spread to interpolate between rather than one point estimate.
P_MAX_SIGMAS = (1.0, 2.0, 3.0)

# Quadrature grid for that integral. Coarser than listener_fit's 129 points
# because this runs once per turn inside games, and the feature only has to be
# a useful covariate, not a maximum-likelihood estimate.
_U_LO, _U_HI, _U_N = -6.0, 6.0, 49

FEATURE_NAMES: list[str] = [
    # per-word signal, one per embedding space
    "z_glove", "z_numberbatch", "z_wiki2vec",
    "rank_glove", "rank_numberbatch", "rank_wiki2vec",
    "gaptop_glove", "gaptop_numberbatch", "gaptop_wiki2vec",
    # board context (constant within a board; earns its place via tree
    # interactions -- a constant cannot change a softmax over the board, but it
    # can gate a split on a per-word feature)
    "k", "n_candidates", "peak_z", "lead_margin",
    # the analytic listener model's own prediction
    "p_max_sigma1", "p_max_sigma2", "p_max_sigma3",
    # multi-space competition
    "own_min_space", "rival_min_space", "gap_vs_rival_min",
    # per-word column statistics: z normalises per CLUE (row), not per WORD.
    # "Bank" is moderately close to everything, "Platypus" to almost nothing.
    "word_mean_sim", "word_sd_sim",
]

N_FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True)
class WordStats:
    """Per-board-word similarity statistics across the whole clue vocabulary.

    Column-wise counterpart to `ClueStats` (which is row-wise). Computed once
    over the full tensor and cached, because it needs every clue.
    """

    mean: np.ndarray  # (n_board_words, n_spaces)
    sd: np.ndarray
    board_words: list[str]

    @classmethod
    def build(cls, sims) -> "WordStats":
        t = np.asarray(sims.tensor, dtype=np.float32)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(t, axis=0)
            sd = np.nanstd(t, axis=0)
        order = sorted(sims.board_index, key=lambda w: sims.board_index[w])
        return cls(mean=mean, sd=sd, board_words=order)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, sd=self.sd, board_words=np.array(self.board_words))

    @classmethod
    def load(cls, path: Path) -> "WordStats":
        d = np.load(path, allow_pickle=False)
        return cls(mean=d["mean"], sd=d["sd"], board_words=[str(w) for w in d["board_words"]])


def p_is_max(z: np.ndarray, sigma: float) -> np.ndarray:
    """P(word i is the listener's pick) under `perceived = z + N(0, sigma)`.

    Conditioning on the winner's own noise makes every other word independent,
    which collapses an otherwise intractable orthant probability to a
    one-dimensional integral -- the same derivation as
    codenames/listener_fit.py, reused here as a feature rather than a
    likelihood.
    """
    n = len(z)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    u = np.linspace(_U_LO, _U_HI, _U_N)
    log_w = -0.5 * u**2 - 0.5 * np.log(2 * np.pi) + np.log((_U_HI - _U_LO) / (_U_N - 1))
    d = (z[:, None] - z[None, :]) / sigma  # (i, j) = (z_i - z_j)/sigma
    terms = log_ndtr(u[None, :, None] + d[:, None, :])  # (i, U, j)
    # Drop the self term: Phi(u + 0) would otherwise damp every row equally.
    idx = np.arange(n)
    terms[idx, :, idx] = 0.0
    inner = terms.sum(axis=2) + log_w[None, :]
    m = inner.max(axis=1, keepdims=True)
    return np.exp(m[:, 0] + np.log(np.exp(inner - m).sum(axis=1)))


def _ranks(values: np.ndarray) -> np.ndarray:
    """Descending rank, normalised to [0, 1]; 0 is the board's best word."""
    n = len(values)
    if n == 1:
        return np.zeros(1)
    order = np.argsort(-values)
    r = np.empty(n)
    r[order] = np.arange(n)
    return r / (n - 1)


def extract(
    clue: str,
    candidates: list[str],
    number: int | None,
    sims,
    clue_stats,
    word_stats: WordStats,
    clue_index: dict[str, int],
) -> np.ndarray | None:
    """`(len(candidates), N_FEATURES)` in the order `candidates` is given, or
    None when the clue is outside the tensor's vocabulary or a candidate has no
    vector -- the caller decides what to do with an unusable position rather
    than getting silently imputed rows."""
    ci = clue_index.get(clue.lower())
    if ci is None:
        return None
    try:
        idxs = [sims.board_index[w.lower()] for w in candidates]
    except KeyError:
        return None

    sim = np.asarray(sims.tensor[ci, idxs, :], dtype=np.float64)  # (n, 3)
    mu = np.asarray(clue_stats.mean[ci], dtype=np.float64)
    sd = np.asarray(clue_stats.std[ci], dtype=np.float64)
    if not np.all(np.isfinite(sd)) or np.any(sd <= 0) or not np.all(np.isfinite(sim)):
        return None
    z = (sim - mu) / sd  # (n, 3)
    n = len(candidates)

    cols: list[np.ndarray] = []
    cols.extend(z[:, s] for s in range(3))
    cols.extend(_ranks(z[:, s]) for s in range(3))
    cols.extend(z[:, s] - z[:, s].max() for s in range(3))

    k = float(number) if number is not None else -1.0
    z_nb = z[:, 1]
    top2 = np.sort(z_nb)[::-1][:2]
    lead = float(top2[0] - top2[1]) if n > 1 else 0.0
    cols.append(np.full(n, k))
    cols.append(np.full(n, float(n)))
    cols.append(np.full(n, float(z_nb.max())))
    cols.append(np.full(n, lead))

    for s in P_MAX_SIGMAS:
        cols.append(p_is_max(z_nb, s))

    own_min = z.min(axis=1)  # strong in EVERY space
    if n > 1:
        # For each word, the best rival's own_min -- max over the others.
        order = np.argsort(-own_min)
        best, second = own_min[order[0]], own_min[order[1]]
        rival = np.where(np.arange(n) == order[0], second, best)
    else:
        rival = np.full(n, -np.inf)
    rival = np.where(np.isfinite(rival), rival, 0.0)
    cols.append(own_min)
    cols.append(rival)
    cols.append(own_min - rival)

    wi = np.asarray(idxs)
    cols.append(np.asarray(word_stats.mean[wi, 1], dtype=np.float64))
    cols.append(np.asarray(word_stats.sd[wi, 1], dtype=np.float64))

    out = np.column_stack(cols)
    assert out.shape == (n, N_FEATURES), (out.shape, N_FEATURES)
    return out

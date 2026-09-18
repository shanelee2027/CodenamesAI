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

**32 features, added as four named blocks.** Each block went in as a stated
hypothesis and was measured on its own rather than poured in at the start; the
per-block ablation is `scripts/pipeline/train_listener.py --blocks` and the
numbers are in docs/log.md. Tier 1 -- the numberbatch views plus everything
derived from them -- was the original 21; glove/wiki2vec, SWOW and the entity
vectors came after.

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

# How many of the clue's strongest candidates define "the cluster" for the
# cohesion features. Small on purpose: a clue for 4 points at a handful of
# words, and averaging over the whole board would drown the signal in words
# the clue never meant.
COHESION_TOP = 5


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
    # tier 2: thematic grouping. Is w part of the cluster the clue points at,
    # as opposed to merely close to the clue on its own? This is what a
    # listener does at k>=3, and the regime where pairwise clue-word
    # similarity collapses (0.36 top-1 at k=4).
    "cohesion", "cohesion_rank", "cohesion_minus_own",
    # tier 2: measured HUMAN association (SWOW). Embeddings measure similarity
    # -- words used in similar contexts. Association measures relatedness --
    # words that come to mind together. "nikon" and "Olympus" are not similar,
    # they are associated, and that is the kind of link a listener follows.
    # Two hops because the raw graph is sparse (0.57 of a 25-word board direct,
    # 15.6 at two hops); see scripts/data/build_swow_tables.py.
    "swow1", "swow2", "swow2_rank", "swow2_share", "swow_has",
    # tier 2: Wikipedia2Vec ENTITY vectors, which the tensor build discards.
    # A word vector for "nikon" encodes how the token is used; the ENTITY
    # vector for the company sits near Olympus and Canon because the articles
    # do. Aimed at the encyclopedic residue (schmidt -> Scorpion).
    "ent_sim", "ent_rank", "ent_has",
]

N_FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True)
class SwowTables:
    """Human-association strengths, clue row -> board-word columns.

    Absent rows (a clue SWOW never cued -- 41% of our pool) and absent entries
    both come back as NaN rather than 0. Zero would claim "these words are
    unrelated"; NaN says "no evidence", and LightGBM learns a split direction
    for it. Conflating the two is the main way a sparse source poisons a dense
    feature set.
    """

    one: object
    two: object
    board_pos: dict[str, int]

    @classmethod
    def load(cls, path: Path) -> "SwowTables":
        import scipy.sparse as sp

        d = np.load(path, allow_pickle=False)
        one = sp.csr_matrix((d["one_data"], d["one_indices"], d["one_indptr"]), shape=tuple(d["one_shape"]))
        two = sp.csr_matrix((d["two_data"], d["two_indices"], d["two_indptr"]), shape=tuple(d["two_shape"]))
        bw = [str(w) for w in d["board_words"]]
        return cls(one=one, two=two, board_pos={w: i for i, w in enumerate(bw)})

    def row(self, which, clue_i: int, cols: np.ndarray) -> np.ndarray:
        m = self.one if which == 1 else self.two
        lo, hi = m.indptr[clue_i], m.indptr[clue_i + 1]
        if lo == hi:
            return np.full(len(cols), np.nan)
        idx, val = m.indices[lo:hi], m.data[lo:hi]
        lookup = dict(zip(idx.tolist(), val.tolist()))
        return np.array([lookup.get(int(c), np.nan) for c in cols], dtype=np.float64)


@dataclass(frozen=True)
class EntitySims:
    """Cosine similarity between same-named Wikipedia entities, NaN where
    either side has no entity vector (29% clue coverage, 62% board)."""

    sims: np.ndarray                 # (n_clue_pool, n_board_words)
    row_of: dict[int, int]           # tensor clue index -> matrix row
    board_pos: dict[str, int]

    @classmethod
    def load(cls, path: Path) -> "EntitySims":
        d = np.load(path, allow_pickle=False)
        rows = {int(c): i for i, c in enumerate(d["clue_rows"])}
        bw = [str(w) for w in d["board_words"]]
        return cls(sims=d["sims"], row_of=rows, board_pos={w: i for i, w in enumerate(bw)})

    def row(self, clue_i: int, cols: np.ndarray) -> np.ndarray:
        r = self.row_of.get(clue_i)
        if r is None:
            return np.full(len(cols), np.nan)
        out = np.full(len(cols), np.nan)
        ok = cols >= 0
        out[ok] = self.sims[r, cols[ok]]
        return out


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
    """Descending rank in [0, 1], NaN-safe and tie-safe.

    Both properties are load-bearing, and getting them wrong leaked the
    answer once already (docs/log.md). A column that is entirely NaN -- a clue
    no association or entity source covers -- must come back all-NaN, NOT
    ranked; and ties must get their average rank rather than being broken by
    array position. `argsort` on equal values returns index order, so either
    slip turns the feature into "where does this word sit in the list", which
    is information the model must never see.
    """
    finite = np.isfinite(values)
    out = np.full(len(values), np.nan)
    if not finite.any():
        return out
    v = values[finite]
    order = np.argsort(-v, kind="stable")
    r = np.empty(len(v), dtype=np.float64)
    r[order] = np.arange(len(v), dtype=np.float64)
    # Average rank within each tied group, so equal values are indistinguishable.
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    out[finite] = r / max(1, len(v) - 1)
    return out


def _cohesion(
    candidates: list[str], idxs: list[int], z_nb: np.ndarray, sims, clue_stats, clue_index: dict[str, int]
) -> np.ndarray:
    """Mean z of w against the clue's top candidates, excluding w itself.

    "Is w in the group?" rather than "is w near the clue?". A word can sit
    close to the clue by accident; a word that is also close to the OTHER
    words the clue selects is part of a theme, which is what a listener is
    looking for when told to find four of something.
    """
    n = len(candidates)
    out = np.full(n, np.nan)
    top = np.argsort(-z_nb)[: COHESION_TOP + 1]
    rows = []
    for t in top:
        ci = clue_index.get(candidates[t].lower())
        if ci is None:
            continue
        sd = float(clue_stats.std[ci, 1])
        if not np.isfinite(sd) or sd <= 0:
            continue
        sim = np.asarray(sims.tensor[ci, idxs, 1], dtype=np.float64)
        rows.append(((sim - float(clue_stats.mean[ci, 1])) / sd, int(t)))
    if not rows:
        return out
    for i in range(n):
        vals = [r[i] for r, t in rows if t != i]
        if vals:
            out[i] = float(np.mean(vals))
    return out


def extract(
    clue: str,
    candidates: list[str],
    number: int | None,
    sims,
    clue_stats,
    word_stats: WordStats,
    clue_index: dict[str, int],
    swow: "SwowTables | None" = None,
    entity: "EntitySims | None" = None,
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

    # Cohesion: mean similarity of w to the other candidates the clue points
    # at most strongly. Uses word-to-word similarity, available because 396 of
    # the 400 board words are themselves in the clue vocabulary (only the
    # multi-word ones are not -- those fall back to NaN, which LightGBM
    # routes on its own).
    coh = _cohesion(candidates, idxs, z[:, 1], sims, clue_stats, clue_index)
    cols.append(coh)
    cols.append(_ranks(coh))
    cols.append(coh - z[:, 1])

    if swow is None:
        for _ in range(5):
            cols.append(np.full(n, np.nan))
    else:
        bcols = np.array([swow.board_pos.get(w.lower(), -1) for w in candidates])
        s1 = swow.row(1, ci, bcols)
        s2 = swow.row(2, ci, bcols)
        cols.append(s1)
        cols.append(s2)
        cols.append(_ranks(s2))
        # Share of the board's total association mass -- a word reachable from
        # the clue matters less when every word is.
        tot = np.nansum(s2)
        cols.append(s2 / tot if tot > 0 else np.full(n, np.nan))
        cols.append(np.where(np.isnan(s2), 0.0, 1.0))

    if entity is None:
        for _ in range(3):
            cols.append(np.full(n, np.nan))
    else:
        ecols = np.array([entity.board_pos.get(w.lower(), -1) for w in candidates])
        e = entity.row(ci, ecols)
        cols.append(e)
        cols.append(_ranks(e))
        cols.append(np.where(np.isnan(e), 0.0, 1.0))

    out = np.column_stack(cols)
    assert out.shape == (n, N_FEATURES), (out.shape, N_FEATURES)
    return out

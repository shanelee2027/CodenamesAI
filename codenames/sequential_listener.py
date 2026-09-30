"""Picks within a turn, modelled as their own choices (docs/log.md, "Picks
within a turn").

**What the listener assumes now.** The guesser's pick j of a turn is its pick
1 with the words already taken removed and the same scores renormalised
(Plackett-Luce with frozen scores). So nothing about pick 2 can depend on
what pick 1 was, and pick 2 is exactly as sharp as pick 1. Both are wrong,
and measurably so:
- Later picks are overconfident: at pick 3 the favourite is predicted 53% and
  picked 34% (docs/log.md, "Is the listener as sure of its later picks").
- The listener's pick-2 accuracy is 0.41, against 0.68-0.75 for any predictor
  that sees the guesser's own repeated answers ("How much is left to
  explain").
- A ranking "stays on the reading its first word committed to" (the prompt
  comparison): after Himalayas for "exploring" it goes on to Africa, not to
  the planets.

**The model.** Pick j >= 2 is a softmax over the words left, with logits

    alpha_j * s(w)  +  beta_j * fit(w, picked)  +  gamma_j * same_sense(w, picked)

where s is the listener's usual score (unchanged, so pick 1 is untouched):
- `alpha_j = exp(a_j + b_j * drop)`, with drop = s(best word left) - s(best
  word picked so far), which is <= 0. It lets pick j flatten once the clue's
  strong words are used up, which a fixed per-pick temperature cannot.
- `fit(w, picked)`: the highest cosine between w and a word already picked
  (mean over the embedding spaces that have both), the "Cap then Glove"
  pull.
- `same_sense(w, picked)`: 1 when w's best WordNet sense of the clue is the
  sense some already-picked word used (scripts/data/build_wordnet_senses.py).
  Measured and kept at 0 in the deployed model: it bought nothing once `fit`
  was in.

Pick 4 and later share one set of parameters. Four numbers per pick, fitted
by maximum likelihood on the stored rankings' later picks. A small model on
purpose: it has to be explained, and it has to run inside the reward, where
the state is the set of own words picked so far (the subset DP of
codenames/spymasters/pick_temperature_listener.py). Every term depends on
that set only, not on the order it was picked in, which is what keeps the DP
exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_STEP_PARAMS = 3          # picks 2, 3, and 4+


@dataclass
class SequentialParams:
    a: np.ndarray          # (3,) log-temperature intercept per pick 2/3/4+
    b: np.ndarray          # (3,) its slope on `drop`
    beta: np.ndarray       # (3,) weight on fit with the words picked
    gamma: np.ndarray      # (3,) weight on sharing a picked word's sense

    @classmethod
    def identity(cls) -> "SequentialParams":
        z = np.zeros(N_STEP_PARAMS)
        return cls(z.copy(), z.copy(), z.copy(), z.copy())

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("a", "b", "beta", "gamma")}

    @classmethod
    def from_dict(cls, d: dict) -> "SequentialParams":
        return cls(*(np.asarray(d[k], dtype=np.float64) for k in ("a", "b", "beta", "gamma")))


def pair_similarity(words: list[str], vw) -> np.ndarray:
    """(n, n) mean cosine over the embedding spaces that have both words
    (codenames/listener_net.WordVectors, L2-normalised), 0 on the diagonal."""
    rows = vw.rows(words)
    if rows is None:
        return np.zeros((len(words), len(words)))
    v = vw.vecs[rows].astype(np.float32)            # (n, S, 300)
    ok = vw.ok[rows].astype(np.float32)              # (n, S)
    cos = np.einsum("isd,jsd->ijs", v, v)            # (n, n, S)
    both = ok[:, None, :] * ok[None, :, :]
    out = (cos * both).sum(-1) / np.maximum(both.sum(-1), 1.0)
    np.fill_diagonal(out, 0.0)
    return out


def step_logits(s: np.ndarray, picked: np.ndarray, left: np.ndarray, sim: np.ndarray,
                sense: np.ndarray, step: int, p: SequentialParams) -> np.ndarray:
    """Logits over one position's words at pick `step` (1-based), -inf off
    `left`. `picked` and `left` are boolean masks; `sense` holds each word's
    best clue-sense index (-1 for none). Pick 1 returns `s` itself."""
    z = np.where(left, s, -np.inf)
    if step == 1 or not picked.any():
        return z
    k = min(step, 4) - 2
    drop = s[left].max() - s[picked].max()
    alpha = np.exp(p.a[k] + p.b[k] * drop)
    fit = sim[:, picked].max(1)
    senses = set(sense[picked][sense[picked] >= 0].tolist())
    same = np.isin(sense, list(senses)) & (sense >= 0) if senses else np.zeros(len(s), bool)
    return np.where(left, alpha * s + p.beta[k] * fit + p.gamma[k] * same, -np.inf)


def sequential_gain_and_penalty(s_own: np.ndarray, s_bad: np.ndarray, costs: np.ndarray, max_k: int,
                                sim: np.ndarray, params: SequentialParams,
                                s_out: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`(gain, penalty)`, each `(n_cand, max_k)` with column m for k = m + 1,
    as pl_reward.gain_and_penalty returns them, under the within-turn model.

    `sim` is the board's (n_own + n_bad, n_own + n_bad) pair similarity, own
    words first, the same order as the score columns. It does not depend on
    the clue. Only own words can have been picked (a wrong pick ends the
    turn), so the state is the set A of own words picked, and the DP is the
    one in pick_temperature_listener.tempered_gain_and_penalty with the
    logits recomputed per state:

        z_A(w) = alpha_A * s(w) + beta * max over o in A of sim(w, o)
        alpha_A = exp(a + b * (max s over words left - max s over A))

    Pick 1 (A empty) uses s unchanged. The gamma (sense) term is not used
    here; the deployed model has it at 0. The outside option, when on, is a
    non-team word with cost 0 and no similarity to anything.
    """
    s_own = np.asarray(s_own, dtype=np.float64)
    s_bad = np.asarray(s_bad, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    C, n_own = s_own.shape
    n_bad0 = s_bad.shape[1]
    if s_out is not None:
        s_bad = np.concatenate([s_bad, np.asarray(s_out, dtype=np.float64).reshape(-1, 1)], axis=1)
        costs = np.concatenate([costs, [0.0]])
    n_bad = s_bad.shape[1]
    sim_own_own = np.asarray(sim[:n_own, :n_own], dtype=np.float64)          # (n_own, n_own)
    sim_bad_own = np.zeros((n_bad, n_own))
    sim_bad_own[:n_bad0] = sim[n_own:n_own + n_bad0, :n_own]
    max_k = min(max_k, n_own)

    n_sub = 1 << n_own
    members = ((np.arange(n_sub)[:, None] >> np.arange(n_own)[None, :]) & 1).astype(bool)
    size = members.sum(1)
    P = np.zeros((C, n_sub))
    P[:, 0] = 1.0
    reach = np.zeros((C, max_k))
    pen = np.zeros((C, max_k))
    bad_max = s_bad.max(1) if n_bad else np.full(C, -np.inf)
    for j in range(1, max_k + 1):
        S = np.flatnonzero(size == j - 1)
        mem = members[S]                                                     # (|S|, n_own)
        if j == 1:
            z_own = np.broadcast_to(s_own[:, None, :], (C, 1, n_own))
            z_bad = np.broadcast_to(s_bad[:, None, :], (C, 1, n_bad))
        else:
            k = min(j, 4) - 2
            # fit to the picked set, per state: max similarity to a member
            fit_own = np.where(mem[:, None, :], sim_own_own[None, :, :], -np.inf).max(2)   # (|S|, n_own)
            fit_bad = np.where(mem[:, None, :], sim_bad_own[None, :, :], -np.inf).max(2)   # (|S|, n_bad)
            picked_max = np.where(mem[None, :, :], s_own[:, None, :], -np.inf).max(2)      # (C, |S|)
            left_own_max = np.where(~mem[None, :, :], s_own[:, None, :], -np.inf).max(2)
            left_max = np.maximum(left_own_max, bad_max[:, None])
            alpha = np.exp(params.a[k] + params.b[k] * (left_max - picked_max))[:, :, None]
            z_own = alpha * s_own[:, None, :] + params.beta[k] * fit_own[None]
            z_bad = alpha * s_bad[:, None, :] + params.beta[k] * fit_bad[None]
        z_own = np.where(mem[None], -np.inf, z_own)
        shift = np.maximum(z_own.max(2), z_bad.max(2) if n_bad else -np.inf)[:, :, None]
        e_own = np.exp(z_own - shift)                                         # (C, |S|, n_own)
        e_bad = np.exp(z_bad - shift)                                         # (C, |S|, n_bad)
        Z = np.maximum(e_own.sum(2) + e_bad.sum(2), 1e-300)
        w = P[:, S] / Z                                                       # (C, |S|)
        pen[:, j - 1] = (w * (e_bad @ costs)).sum(1)
        for o in range(n_own):
            free = ~mem[:, o]
            P[:, S[free] | (1 << o)] += w[:, free] * e_own[:, free, o]
        reach[:, j - 1] = P[:, size == j].sum(1)
    return np.cumsum(reach, 1).astype(np.float32), np.cumsum(pen, 1).astype(np.float32)

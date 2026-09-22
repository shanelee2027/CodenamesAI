"""Expected reward of a clue under a learned Plackett-Luce listener.

The derivation is `docs/clue-selection-learned.tex`; this is its numerical
counterpart, and the parallel to `spymasters/expected_words.py` is deliberate
-- same `(gain, penalty)` interface, same Poisson-binomial recursion, same
"column m is k = m+1" convention -- so the two guesser models can be swapped
without touching the search around them.

**What the listener supplies.** One score `s_x` per unrevealed board word for a
fixed clue, with the guesser picking from a remaining set `S` according to
`softmax(s)` restricted to `S`. Scores do not depend on `S`: removing a word
changes only the normalising sum. That is Luce's axiom, and the distilled model
in `listener_features.py` is fitted under exactly it -- `train_listener.py`
computes the feature matrix once per position and later steps merely drop rows
-- so carrying the assumption here is consistency, not convenience.

**Why there is no tree.** Repeated softmax selection defines a distribution
over orderings, and the expected reward is a sum over it: each first pick opens
a subtree of second picks with renormalised probabilities, and so on, at
`O(n**K)` cost. Setting `lambda_x = exp(s_x)` and giving each word an
independent `Exp(lambda_x)` clock reproduces that distribution exactly --
memorylessness makes the conditional law after each firing the renormalised
softmax again -- which turns the tree into minima of independent exponentials.
`tests/test_pl_reward.py` brute-forces the tree on small boards and checks the
two agree to floating-point, because the whole approach rests on that identity.

**The outside option.** `s_out` is the score of "the guesser picks something
the clue never meant". It is identified by training the listener against words
drawn uniformly from the vocabulary (scripts/data/collect_decoy_data.py): the
board-normalised softmax is shift-invariant, so without such an anchor nothing
pins how good a clue is in absolute terms, only which word it favours -- a
clue at z=1 over four own words and one at z=3 over the same four are
indistinguishable. Measured, a random word sits about 4.1 nats below the best
board word.

It enters as a non-team word with cost ZERO: hitting it ends the turn and
scores nothing, which is the modelling choice that its expected value is that
of passing. Two things follow, and both are the point. A vague clue has a
large `s_out` relative to its board words, so the turn ends sooner and `gain`
falls. And because some endings are now free, `cbar` falls too -- so a vague
clue is driven toward zero reward rather than toward a large penalty. A sharp
clue has `s_out` far below its best word and is barely touched.

Worth naming: cost zero is more optimistic than the measured behaviour. When a
decoy won in the probe, 36.5% of turns ended and the assassin rate was 7.6x
the base -- a real guesser does not pass, it guesses wrong. Zero is therefore
a floor on the harm, and the direction is still correct relative to the status
quo, where the outside option is not priced at all.

**Where this is cheaper than the Gaussian model.** For independent exponentials
the minimum and its argument are independent, so which non-team word ends the
turn does not depend on when. The expected miss cost is therefore one constant
per clue rather than a hazard ratio evaluated at every quadrature point, and
`|B|` leaves the integral entirely. The substitution `u = exp(-Lambda t)` also
maps the integral to `[0, 1]` exactly, so unlike the Gaussian version there is
no interval to choose and no tail to argue is negligible.
"""

from __future__ import annotations

import numpy as np

# Quadrature cells for the u-integral.
#
# The integrand is bounded on [0, 1] but not smooth at u = 0: p_i(u) carries
# u ** (lambda_i / Lambda), whose derivative is unbounded there when that ratio
# is below 1. So the midpoint rule converges at about first order rather than
# second, and more cells buy less than they usually would -- measured against
# brute-force enumeration, the error roughly halves per doubling:
#
#     cells    48      96     192     384    1024    4096
#     gain   1.8e-3  4.7e-4  2.3e-4  1.1e-4  3.8e-5  8.8e-6
#     pen    8.3e-3  2.8e-3  1.3e-3  5.6e-4  1.7e-4  3.3e-5
#
# What the search actually needs is the ORDER of clues, not their values, and
# that converges far sooner. On 500 competing clues over a 9-own/16-other
# board, against an 8192-cell reference: the argmax, the top ten and the chosen
# k all agree exactly from 48 cells upward. 96 is kept to match
# spymasters/expected_words.py and leaves a comfortable margin at 7 ms per 500
# clues; raise it only if absolute values, rather than rankings, start mattering.
GRID_CELLS = 96


def gain_and_penalty(
    s_own: np.ndarray,
    s_bad: np.ndarray,
    costs: np.ndarray,
    max_k: int,
    cells: int = GRID_CELLS,
    s_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`(gain, penalty)`, each `(n_cand, max_k)`, column `m` being `k = m + 1`.

    `s_own` is `(n_cand, n_own)` and `s_bad` is `(n_cand, n_bad)`: the
    listener's scores for our unrevealed words and for everything else, one row
    per candidate clue. `costs` is `(n_bad,)`, the positive cost of the guesser
    picking each non-team word.

    Expected reward for a clue at `k` is `gain[:, k-1] - penalty[:, k-1]`.

    `s_out` is `(n_cand,)` or `(n_cand, 1)`, the outside option's score -- see
    the module docstring. It is appended to `s_bad` with cost 0, so it is
    exactly a non-team word that ends the turn for free. Omitted, the result is
    identical to every run made before the outside option existed.

    Scores are shifted by their per-row maximum before exponentiating. The
    shift cancels in every ratio the derivation uses -- `lambda_i / Lambda` and
    `lambda_w / Lambda` are both scale-invariant -- so it changes nothing but
    keeps `exp` away from overflow on confident rows. NOTE that this is why
    `s_out` must be on the same scale as `s_own`/`s_bad`: the shift-invariance
    the derivation enjoys is exactly what the anchor gives up, deliberately.
    """
    if s_own.ndim != 2 or s_bad.ndim != 2:
        raise ValueError("s_own and s_bad must be (n_cand, n_words)")
    if s_bad.shape[1] != len(costs):
        raise ValueError(f"costs has {len(costs)} entries for {s_bad.shape[1]} non-team words")
    n_cand, n_own = s_own.shape

    if s_out is not None:
        extra = np.asarray(s_out, dtype=np.float64).reshape(n_cand, 1)
        s_bad = np.concatenate([np.asarray(s_bad, dtype=np.float64), extra], axis=1)
        costs = np.concatenate([np.asarray(costs, dtype=np.float64), [0.0]])

    shift = np.maximum(s_own.max(axis=1, keepdims=True), s_bad.max(axis=1, keepdims=True))
    lam_own = np.exp(s_own - shift)                       # (n_cand, n_own)
    lam_bad = np.exp(s_bad - shift)                       # (n_cand, n_bad)

    big_lambda = lam_bad.sum(axis=1, keepdims=True)       # (n_cand, 1)
    big_lambda = np.maximum(big_lambda, 1e-300)

    # E[c_W]: which non-team word ends the turn is independent of when, so this
    # is a constant per clue rather than a function of the integration variable.
    cbar = (lam_bad @ costs)[:, None] / big_lambda        # (n_cand, 1)

    # u = exp(-Lambda t) maps (0, inf) -> (0, 1) with dF_T(t) = -du, so the
    # midpoints below are uniform and each carries equal mass 1/cells.
    u = (np.arange(cells, dtype=np.float64) + 0.5) / cells      # (cells,)

    # p_i(u) = 1 - u ** (lambda_i / Lambda): no special functions anywhere.
    ratio = lam_own / big_lambda                                 # (n_cand, n_own)
    p = 1.0 - u[None, :, None] ** ratio[:, None, :]              # (n_cand, cells, n_own)

    # Poisson-binomial over our words, truncated at max_k since larger counts
    # are never used. q[:, :, r] = P(N = r | u).
    q = np.zeros((n_cand, cells, max_k + 1), dtype=np.float64)
    q[:, :, 0] = 1.0
    for i in range(n_own):
        pi = p[:, :, i][:, :, None]
        keep = q * (1.0 - pi)
        shifted = np.zeros_like(q)
        shifted[:, :, 1:] = q[:, :, :-1] * pi
        q = keep + shifted

    # tail[:, :, m] = P(N >= m+1 | u) = 1 - P(N <= m | u)
    tail = 1.0 - np.cumsum(q[:, :, :max_k], axis=2)

    # Equal-mass midpoint rule: every cell carries 1/cells of T's probability.
    mass = np.full(cells, 1.0 / cells)

    # gain(k) = sum_{j<=k} P(N >= j), integrated over T.
    gain = np.cumsum(np.einsum("c,nck->nk", mass, tail), axis=1)

    # The turn ends on a non-team word exactly when N < k, and the cost of that
    # word is cbar regardless of when -- so the expectation factors.
    penalty = cbar * np.einsum("c,nck->nk", mass, 1.0 - tail)

    return gain.astype(np.float32), penalty.astype(np.float32)


def expected_reward(
    s_own: np.ndarray,
    s_bad: np.ndarray,
    costs: np.ndarray,
    max_k: int,
    cells: int = GRID_CELLS,
    s_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`(best_k, value)` per candidate clue, each `(n_cand,)`.

    `best_k` is 1-based, so it is the number to announce.
    """
    gain, penalty = gain_and_penalty(s_own, s_bad, costs, max_k, cells, s_out)
    net = gain - penalty
    best = net.argmax(axis=1)
    return best + 1, net[np.arange(len(best)), best]

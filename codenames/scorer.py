"""The learned scorer (SCOPE.md §2, §M8): an MLP predicting a distribution
over k -- how many own-words a guesser will reveal for a clue before
stopping -- plus the play-time scoring formula that turns that distribution
into a (clue, number) choice.

**Resolving a gap in §2's play-time formula.** The model's output is *only*
P(k|clue) -- it says nothing about *why* a stop happened (neutral, opponent,
or assassin all just collapse into "not own"). But `reward(k, n)` needs a
penalty value for that stop, and the per-category reward table (neutral 0,
opponent -1, assassin -10) can't be recovered from k alone. Resolved (see
project discussion) by taking SCOPE's own sentence -- "the assassin penalty
is the risk-aversion parameter" -- literally: every stop, regardless of its
true cause, is charged a single configurable `miss_penalty` (default -10,
the assassin value) at *scoring* time, not baked into training. This keeps
P(k|clue) exactly what §2 specifies (5 logits, nothing else) and keeps risk
aversion a genuine runtime knob -- turning it down doesn't require
retraining, because the model was never trained on any particular penalty
value in the first place. The tradeoff: this is a worst-case-flavored
simplification (a neutral miss is charged as if it might have been the
assassin), not a per-category expectation -- documented here because it's
exactly the kind of choice CLAUDE.md says to flag rather than pick silently.

**reward(k, n)**, for n and k both in 0..MAX_K (MAX_K=4, matching
codemasters.base.MAX_CLUE_NUMBER and the training labels' cap -- see
scripts/generate_training_data.py):

    reward(k, n) = k * OWN_REWARD + miss_penalty   if k <= n   (natural
                                                     stop happens within
                                                     the n+1-attempt budget)
                 = (n + 1) * OWN_REWARD             if k > n    (budget runs
                                                     out first, no miss
                                                     encountered)

k=MAX_K is a right-censored "MAX_K or more" bucket (see the training-data
docstring); treating it as exactly MAX_K here is a second, smaller
approximation, and only actually bites when n == MAX_K.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from codenames.board import MAX_CLUE_NUMBER as MAX_K
from codenames.board import Role
from codenames.game import ROLE_REWARD

OWN_REWARD = ROLE_REWARD[Role.OWN]
DEFAULT_MISS_PENALTY = ROLE_REWARD[Role.ASSASSIN]  # -10.0, per SCOPE §2/§6
N_K_CLASSES = MAX_K + 1  # k in 0..MAX_K -> 5 classes


class Scorer(nn.Module):
    """MLP per SCOPE §2: input_dim -> (256, 256, 128) -> N_K_CLASSES logits.
    Returns raw logits (not softmaxed) -- use torch.softmax(...) or
    predict_proba() for an actual probability distribution; training uses
    the logits directly with nn.CrossEntropyLoss."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, N_K_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


def reward_matrix(miss_penalty: float = DEFAULT_MISS_PENALTY, max_k: int = MAX_K) -> np.ndarray:
    """(max_k+1, max_k+1) matrix, reward_matrix[k, n] = reward(k, n). Built
    once per risk-aversion setting; play-time scoring is then just a matrix
    multiply against a batch of P(k|clue) rows."""
    k_vals = np.arange(max_k + 1)
    n_vals = np.arange(max_k + 1)
    k_grid, n_grid = np.meshgrid(k_vals, n_vals, indexing="ij")

    natural_stop = k_grid * OWN_REWARD + miss_penalty
    budget_exhausted = (n_grid + 1) * OWN_REWARD
    return np.where(k_grid <= n_grid, natural_stop, budget_exhausted).astype(np.float32)


def expected_reward_and_best_n(probs: np.ndarray, miss_penalty: float = DEFAULT_MISS_PENALTY) -> tuple[np.ndarray, np.ndarray]:
    """probs: (batch, N_K_CLASSES) distribution over k for each candidate
    clue. Returns (best_n, score), each shape (batch,): the number
    maximizing expected reward, and the expected reward at that number --
    SCOPE §2's `best_n = argmax_n E[reward|clue,n]` and
    `score(clue) = E[reward|clue,best_n]`, vectorized over every candidate
    clue at once (the "one gather plus one small forward pass" §2 asks
    for)."""
    matrix = reward_matrix(miss_penalty)  # (n_k, n_n)
    expected = probs @ matrix  # (batch, n_n) -- E[reward|clue,n] for every n
    best_n = np.argmax(expected, axis=1)
    score = np.take_along_axis(expected, best_n[:, None], axis=1)[:, 0]
    return best_n, score

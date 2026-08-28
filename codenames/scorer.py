"""The learned scorer (SCOPE.md §2, §M8): an MLP predicting a distribution
over (k, cause) -- how many own-words a guesser will reveal for a clue
before stopping, AND what stopped it (neutral / opponent / assassin, or
nothing -- it ran out of budget clean) -- plus the play-time scoring
formula that turns that distribution into a (clue, number) choice.

**Resolving a gap in §2's play-time formula.** An earlier version of this
model predicted only P(k|clue) -- it said nothing about *why* a stop
happened (neutral, opponent, or assassin all collapsed into "not own"),
so every stop had to be charged the same flat worst-case `miss_penalty`
at scoring time, since the per-category reward table (neutral -0.2,
opponent -1, assassin -10) couldn't be recovered from k alone. That's
resolved here by predicting the *cause* too: the label space widens from
5 classes (k in 0..MAX_K) to `N_OUTCOME_CLASSES` = `MAX_K * 3 + 1` = 13
(three causes for each k in 0..MAX_K-1, plus one censored "reached the
cap, no miss" class for k=MAX_K -- see `outcome_class`/
`decode_outcome_class`). The reward formula can then apply each cause's
*true* value instead of one flattened number, and all four reward
components (own/neutral/opponent/assassin) stay adjustable at *scoring*
time, not baked into training -- turning any of them down doesn't require
retraining, because the model was never trained against any particular
reward value in the first place, only against the empirical (k, cause)
outcome itself.

**reward(k, cause, n)**, for n in 0..MAX_K and k in 0..MAX_K (MAX_K=4,
matching codemasters.base.MAX_CLUE_NUMBER and the training labels' cap --
see scripts/generate_training_data.py):

    reward(k, cause, n) = k * own_reward + reward_of(cause)   if k < n
                                             (natural stop happens within
                                             the n-attempt budget)
                         = n * own_reward                      if k >= n
                                             (budget runs out first, no
                                             miss encountered -- cause is
                                             irrelevant/undefined here)

`reward_of(cause)` is `neutral_reward`/`opponent_reward`/`assassin_reward`
depending on which role stopped the rollout. `assassin_reward` (default
-10, matching `DEFAULT_MISS_PENALTY`) is the one meant to double as a
"risk aversion" knob per SCOPE's own "the assassin penalty is the
risk-aversion parameter" -- the other three default to the real game's
`ROLE_REWARD` values (own +1, neutral -0.2, opponent -1), not baseline-3's
separate untuned -0.3-for-neutral constant (`codemasters/linear_scorer.py`),
since this is the reward the model is actually meant to optimize, not an
illustrative heuristic. Neutral being mildly negative (not a true 0) is
deliberate: a neutral guess still burns a turn and gives no progress
toward winning, so it's not actually free -- see docs/log.md. All four are exposed as independent runtime
parameters (see `codenames/codemasters/learned.py` and the web UI) so any
of them can be explored without retraining.

A clue announcing n gives exactly n guesses -- no standard-Codenames "+1
bonus guess" (see docs/log.md's numbering-convention entries for why).

k=MAX_K is a right-censored "MAX_K or more" bucket (see the training-data
docstring). Since n never exceeds MAX_K, a censored k always means the
true k is >= n too, so it always lands correctly in the budget-exhausted
branch regardless of cause -- no approximation error here.
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

# Fixed order -- index into this list is how a cause is packed into a
# class id. Only the three "you stopped on something that isn't your own
# word" roles; OWN never stops a rollout (it's what increments k) and
# there is no "assassin twice" etc. to represent.
STOP_CAUSES: list[Role] = [Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN]
N_STOP_CAUSES = len(STOP_CAUSES)

# For each k in 0..MAX_K-1, one class per possible cause, plus one class
# for k=MAX_K (censored -- reached the cap with no miss, cause undefined).
N_OUTCOME_CLASSES = MAX_K * N_STOP_CAUSES + 1  # 4*3 + 1 = 13


def outcome_class(k: int, cause: Role | None) -> int:
    """Pack a (k, cause) rollout outcome into a single class id in
    0..N_OUTCOME_CLASSES-1. `cause` must be None iff k==MAX_K (the
    censored bucket -- the rollout hit the cap before any miss, so there
    is no cause to record) and one of STOP_CAUSES otherwise (k<MAX_K
    always means *something* stopped it)."""
    if k >= MAX_K:
        if cause is not None:
            raise ValueError(f"k={k} >= MAX_K={MAX_K} (censored) must have cause=None, got {cause!r}")
        return N_OUTCOME_CLASSES - 1
    if cause is None:
        raise ValueError(f"k={k} < MAX_K={MAX_K} must have a real cause, got None")
    return k * N_STOP_CAUSES + STOP_CAUSES.index(cause)


def decode_outcome_class(cls: int) -> tuple[int, Role | None]:
    """Inverse of outcome_class."""
    if cls == N_OUTCOME_CLASSES - 1:
        return MAX_K, None
    k, cause_idx = divmod(cls, N_STOP_CAUSES)
    return k, STOP_CAUSES[cause_idx]


class Scorer(nn.Module):
    """MLP per SCOPE §2: input_dim -> (256, 256, 128) -> N_OUTCOME_CLASSES
    logits. Returns raw logits (not softmaxed) -- use torch.softmax(...)
    or predict_proba() for an actual probability distribution; training
    uses the logits directly with nn.CrossEntropyLoss."""

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
            nn.Linear(128, N_OUTCOME_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


class LinearScorer(nn.Module):
    """SCOPE §6 baseline 4: a linear model over the exact same feature
    vector the MLP uses, no hidden layers. Same interface as Scorer so it's
    a drop-in alternative for scripts/train_scorer.py's model_factory --
    the gap between this and Scorer is the project's headline result (§6:
    "baselines 3 and 4 are the informative pair ... the gap between 3 and 5
    is the project's headline result"). Its weight matrix is also what
    makes the model interpretable: see codenames.features.FeatureLayout.describe."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Linear(input_dim, N_OUTCOME_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


_CAUSE_REWARD_KEYS: dict[Role, str] = {
    Role.NEUTRAL: "neutral_reward",
    Role.OPPONENT: "opponent_reward",
    Role.ASSASSIN: "assassin_reward",
}


def reward_matrix(
    own_reward: float = OWN_REWARD,
    neutral_reward: float = ROLE_REWARD[Role.NEUTRAL],
    opponent_reward: float = ROLE_REWARD[Role.OPPONENT],
    assassin_reward: float = DEFAULT_MISS_PENALTY,
    max_k: int = MAX_K,
) -> np.ndarray:
    """(N_OUTCOME_CLASSES, max_k+1) matrix, reward_matrix[class, n] =
    reward(*decode_outcome_class(class), n). Built once per reward
    setting; play-time scoring is then just a matrix multiply against a
    batch of P(class|clue) rows."""
    cause_reward = {Role.NEUTRAL: neutral_reward, Role.OPPONENT: opponent_reward, Role.ASSASSIN: assassin_reward}
    n_classes = max_k * N_STOP_CAUSES + 1
    n_vals = np.arange(max_k + 1)

    matrix = np.empty((n_classes, max_k + 1), dtype=np.float32)
    for cls in range(n_classes):
        k, cause = decode_outcome_class(cls)
        stop_reward = cause_reward[cause] if cause is not None else 0.0  # unused when cause is None (see below)
        natural_stop = k * own_reward + stop_reward
        budget_exhausted = n_vals * own_reward
        matrix[cls] = np.where(k < n_vals, natural_stop, budget_exhausted)
    return matrix


def expected_reward_and_best_n(
    probs: np.ndarray,
    own_reward: float = OWN_REWARD,
    neutral_reward: float = ROLE_REWARD[Role.NEUTRAL],
    opponent_reward: float = ROLE_REWARD[Role.OPPONENT],
    assassin_reward: float = DEFAULT_MISS_PENALTY,
    min_n: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """probs: (batch, N_OUTCOME_CLASSES) distribution over (k, cause) for
    each candidate clue. Returns (best_n, score), each shape (batch,): the
    number maximizing expected reward, and the expected reward at that
    number -- SCOPE §2's `best_n = argmax_n E[reward|clue,n]` and
    `score(clue) = E[reward|clue,best_n]`, vectorized over every candidate
    clue at once (the "one gather plus one small forward pass" §2 asks
    for).

    `min_n=1` excludes n=0 from consideration: reward_matrix's formula
    treats n=0 as a legitimate (if useless -- 0 attempts, reward always 0)
    play, but every other codemaster in this project floors its announced
    number at 1 (codemasters/_util.py::natural_number), so the learned
    codemaster does too by default, for consistency."""
    matrix = reward_matrix(own_reward, neutral_reward, opponent_reward, assassin_reward)  # (n_classes, n_n)
    expected = probs @ matrix  # (batch, n_n) -- E[reward|clue,n] for every n
    candidate_n = np.arange(matrix.shape[1])
    allowed = candidate_n >= min_n
    expected_allowed = expected[:, allowed]
    allowed_n = candidate_n[allowed]

    best_idx = np.argmax(expected_allowed, axis=1)
    best_n = allowed_n[best_idx]
    score = np.take_along_axis(expected_allowed, best_idx[:, None], axis=1)[:, 0]
    return best_n, score

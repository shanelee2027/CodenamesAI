"""A clue policy: a network that maps a board straight to a distribution over
the clue pool, reading raw clue-word similarities and nothing fitted by
another model.

Two spymasters use it and differ only in their weights:
`codenames/spymasters/imitation_policy.py` (trained to reproduce the
incumbent's picks) and `codenames/spymasters/gptoss_reward_policy.py` (that
policy fine-tuned on what gpt-oss actually does with its clues). This module
holds everything they share, plus what training needs to agree with play on:
the inputs, the network, the legality mask, the positions it is trained on and
the reward read off a guesser's ranking.

## Inputs

For every (pool clue, board word) pair, one fixed vector of raw evidence,
built once by scripts/data/build_policy_features.py into
`cache/policy_features.npy`: z-scored similarity in the tensor's three
embedding spaces and in glove840 and fasttext, SWOW association in both
directions at one and two hops, Wikipedia entity similarity, language-model
PMI, WordNet Wu-Palmer similarity and LCS depth, and the seven spelling and
gloss overlaps. Per board word: the Brysbaert norms and the word's mean and
spread of similarity over the clue vocabulary. Missing values (a clue SWOW
never cued) are 0 after standardisation, with a has-value flag beside them.

No listener score, no expected_words probability, and nothing computed from
either is an input. Unlike the listener, the policy sees each word's role: it
is a spymaster, and the spymaster holds the key.

## Network

DeepSets with order-statistic pooling. A per-word encoder phi maps one (clue,
word) pair to a vector h and a scalar salience u. Per clue, the board is then
summarised without reference to word order:

- the own words, sorted by u, top four kept (h, u and a slot-filled flag);
- each non-own role (neutral, opponent, assassin): the elementwise max of h,
  the logsumexp of u, and whether the role has any word left;
- the mean of h over every unrevealed word, and the unrevealed count per role.

A turn is decided by order statistics -- whether the k-th own word outranks
the strongest dangerous non-own one -- so the pooling hands those over
directly. A self-attention block over the words would also let words interact
(cohesion), at 25x25 per clue and 11k clues per board; the cheaper set
network is the starting point (docs/log.md, "gptoss_reward_policy: design").

## Two heads

- **Policy logit**, one per clue. Masked to the legal clues and
  softmaxed over the pool, it is pi(clue | board).
- **Outcome head**, 13 logits per clue: how a guesser's ranking plays out from
  the top. J own words, then a miss on a neutral, opponent or assassin word
  (j = 0..3, three roles), or at least K_max own words in a row (the cap).
  One ranking yields one category, and the category gives the turn's reward
  for every announced number at once. The expected reward of announcing k,
  under any role costs, follows in closed form (`expected_rewards`), so the
  costs are applied at scoring time and the head itself fits no reward
  values (docs/design-decisions.md, "Reward values are a scoring-time
  knob"). The number the policy announces is the k that maximises it.

## Positions

`deal_position(seed)` builds a board from the TRAINING vocabulary (the 150
held-out words never appear), pre-reveals 0-8 non-assassin cards as the /eval
page deals, and views it from team A on even seeds and team B on odd ones.
Seed ranges are disjoint by purpose, so the train/validation split is by board
seed, never by row.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from codenames.board import MAX_CLUE_NUMBER, Board, OpponentBoardView, Role, load_training_wordlist
from codenames.game import ROLE_REWARD
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

FEATURES_FILE = "policy_features.npy"
AUX_FILE = "policy_features_aux.npz"

# Role ids as the network sees them; the order is part of every checkpoint.
ROLE_ORDER = (Role.OWN, Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN)
ROLE_ID = {r: i for i, r in enumerate(ROLE_ORDER)}
MISS_ROLES = (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN)
N_OUTCOMES = MAX_CLUE_NUMBER * len(MISS_ROLES) + 1      # 13
CAP = N_OUTCOMES - 1
BOARD_SLOTS = 25
MAX_PREREVEAL = 8

# Disjoint seed ranges, one per purpose. Validation positions are never
# trained on by either stage.
IMITATION_SEEDS = (20_000_000, 21_000_000)
RL_SEEDS = (21_000_000, 22_000_000)
VAL_SEEDS = (29_000_000, 29_100_000)


# ---------------------------------------------------------------------------
# Positions


@lru_cache(maxsize=1)
def _training_vocab() -> tuple[str, ...]:
    return tuple(load_training_wordlist())


def deal_position(seed: int) -> Board | OpponentBoardView | None:
    """A mid-game position from the training vocabulary: 0-8 non-assassin cards
    pre-revealed (the rule scripts/tools/play_server.py's `deal_position` uses
    for /eval), seen from team A on even seeds and team B on odd ones. None if
    the draw leaves fewer than two own words, which /eval redeals and the
    callers here skip."""
    board = Board.generate(seed=seed, vocabulary=list(_training_vocab()))
    rng = random.Random(seed ^ 0x5EED)
    pool = [i for i, c in enumerate(board.cards) if c.role is not Role.ASSASSIN]
    pre = rng.sample(pool, rng.randint(0, MAX_PREREVEAL))
    board.revealed = {board.words[i] for i in pre}
    view = board if seed % 2 == 0 else OpponentBoardView(board)
    return view if view.remaining(Role.OWN) >= 2 else None


def positions(seed_range: tuple[int, int], n: int) -> list[tuple[int, Board | OpponentBoardView]]:
    """The first `n` usable positions in a seed range, in seed order."""
    out = []
    seed = seed_range[0]
    while len(out) < n:
        if seed >= seed_range[1]:
            raise ValueError(f"seed range {seed_range} holds fewer than {n} positions")
        view = deal_position(seed)
        if view is not None:
            out.append((seed, view))
        seed += 1
    return out


def k_max(view) -> int:
    return min(view.remaining(Role.OWN), MAX_CLUE_NUMBER)


# ---------------------------------------------------------------------------
# Reward read off one ranking


def turn_reward(ranking: list[str], role_of: dict[str, Role], number: int) -> float:
    """The reward codenames/game.py::play_turn scores when a guesser plays
    `ranking` (every unrevealed word, best first) with `number` guesses:
    exactly `number` guesses, stopping at the first non-own word or once the
    own words run out."""
    reward = 0.0
    own_left = sum(1 for w in ranking if role_of[w] is Role.OWN)
    for w in ranking[:number]:
        role = role_of[w]
        reward += ROLE_REWARD[role]
        if role is not Role.OWN:
            break
        own_left -= 1
        if own_left == 0:
            break
    return reward


def outcome_of(ranking: list[str], role_of: dict[str, Role], kmax: int) -> int:
    """The outcome category of a ranking: j * 3 + (miss role) when the first
    non-own word comes after j < kmax own words, else CAP."""
    j = 0
    for w in ranking:
        if j >= kmax:
            return CAP
        role = role_of[w]
        if role is Role.OWN:
            j += 1
            continue
        return j * len(MISS_ROLES) + MISS_ROLES.index(role)
    return CAP


def reward_of_outcome(category: int, number: int) -> float:
    """What announcing `number` scores when the ranking's outcome is
    `category` -- the same number `turn_reward` gives, from the category alone."""
    if category == CAP:
        return float(number)
    j, e = divmod(category, len(MISS_ROLES))
    if j >= number:
        return float(number)
    return j + ROLE_REWARD[MISS_ROLES[e]]


def role_map(view) -> dict[str, Role]:
    return {w: view.role_of(w) for w in view.words if not view.is_revealed(w)}


# ---------------------------------------------------------------------------
# Features and legality


@dataclass
class PolicyFeatures:
    """The precomputed inputs, loaded once. `pair` is word-major so one
    board's 25 columns are 25 contiguous slabs of the memory map."""

    pair: np.ndarray            # (n_board_words, n_pool, F_pair) float16
    word: np.ndarray            # (n_board_words, F_word) float32
    illegal: np.ndarray         # (n_board_words, n_pool) bool: clue illegal beside that word
    pool: np.ndarray            # (n_pool,) tensor clue indices
    clue_words: list[str]       # the pool's words
    board_index: dict[str, int]
    pair_names: list[str]
    word_names: list[str]

    @classmethod
    def load(cls, cache_dir: Path = DEFAULT_CACHE_DIR, mmap: bool = True) -> "PolicyFeatures":
        cache_dir = Path(cache_dir)
        aux = np.load(cache_dir / AUX_FILE, allow_pickle=False)
        pair = np.load(cache_dir / FEATURES_FILE, mmap_mode="r" if mmap else None)
        board_words = [str(w) for w in aux["board_words"]]
        return cls(pair=pair, word=aux["word"].astype(np.float32), illegal=aux["illegal"],
                   pool=aux["pool"], clue_words=[str(w) for w in aux["clue_words"]],
                   board_index={w.lower(): i for i, w in enumerate(board_words)},
                   pair_names=[str(n) for n in aux["pair_names"]],
                   word_names=[str(n) for n in aux["word_names"]])

    @property
    def n_pool(self) -> int:
        return len(self.pool)

    def encode(self, view) -> "BoardInputs":
        """The 25 slots of one board: word ids, role ids, unrevealed flags, and
        the legal-clue mask over the pool."""
        words = np.array([self.board_index[w.lower()] for w in view.words], dtype=np.int64)
        roles = np.array([ROLE_ID[view.role_of(w)] for w in view.words], dtype=np.int64)
        present = np.array([not view.is_revealed(w) for w in view.words], dtype=bool)
        return BoardInputs(words=words, roles=roles, present=present, legal=self.legal_mask(words))

    def legal_mask(self, words: np.ndarray) -> np.ndarray:
        """Legal clues for a board with these word ids. `is_legal_clue` fails a
        clue if ANY board word (revealed or not) rules it out, word by word, so
        the board's mask is the AND of per-word masks built with it
        (tests/test_clue_policy.py checks the two agree)."""
        return ~self.illegal[words].any(axis=0)


@dataclass
class BoardInputs:
    words: np.ndarray       # (25,) board-word ids
    roles: np.ndarray       # (25,) ROLE_ID
    present: np.ndarray     # (25,) unrevealed
    legal: np.ndarray       # (n_pool,) bool


def stack_inputs(feats: PolicyFeatures, boards: list[BoardInputs]):
    """Numpy batch: pair (B, 25, P, F), word (B, 25, Fw), roles, present, legal."""
    words = np.stack([b.words for b in boards])
    return (np.asarray(feats.pair[words.reshape(-1)]).reshape(len(boards), BOARD_SLOTS, feats.n_pool, -1),
            feats.word[words], np.stack([b.roles for b in boards]),
            np.stack([b.present for b in boards]), np.stack([b.legal for b in boards]))


# ---------------------------------------------------------------------------
# The network (torch imported lazily: nothing above needs it)


def build_net(n_pair: int, n_word: int, width: int = 32, hidden: int = 64, trunk: int = 128):
    import torch
    from torch import nn

    class CluePolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = dict(n_pair=n_pair, n_word=n_word, width=width, hidden=hidden, trunk=trunk)
            self.phi = nn.Sequential(nn.Linear(n_pair + n_word + len(ROLE_ORDER), hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, width))
            self.salience = nn.Linear(width, 1)
            d = (MAX_CLUE_NUMBER + len(MISS_ROLES)) * (width + 2) + width + len(ROLE_ORDER)
            self.rho = nn.Sequential(nn.Linear(d, trunk), nn.GELU(), nn.Linear(trunk, trunk), nn.GELU())
            self.logit = nn.Linear(trunk, 1)
            self.outcome = nn.Linear(trunk, N_OUTCOMES)

        def forward(self, pair, word, roles, present):
            """pair (B, N, P, Fp), word (B, N, Fw), roles (B, N), present (B, N)
            -> (policy logits (B, P), outcome logits (B, P, 13), trunk (B, P, T)).
            Logits are not masked here; see `masked_log_policy`."""
            B, N, P, _ = pair.shape
            neg = torch.finfo(pair.dtype if pair.is_floating_point() else torch.float32).min
            onehot = nn.functional.one_hot(roles, len(ROLE_ORDER)).to(pair.dtype)
            x = torch.cat([pair, word[:, :, None, :].expand(-1, -1, P, -1),
                           onehot[:, :, None, :].expand(-1, -1, P, -1)], dim=-1)
            h = self.phi(x)                                         # (B, N, P, W)
            u = self.salience(h).squeeze(-1)                        # (B, N, P)
            W = h.shape[-1]

            parts = []
            own = (roles == ROLE_ID[Role.OWN]) & present            # (B, N)
            n_own = own.sum(1)                                      # (B,)
            u_own = u.masked_fill(~own[:, :, None], neg)
            top_u, top_i = u_own.topk(MAX_CLUE_NUMBER, dim=1)      # (B, 4, P)
            slot = (torch.arange(MAX_CLUE_NUMBER, device=pair.device)[None, :] < n_own[:, None])
            slot = slot[:, :, None].to(h.dtype)                     # (B, 4, 1)
            h_top = torch.gather(h, 1, top_i[..., None].expand(-1, -1, -1, W)) * slot[..., None]
            top_u = top_u * slot
            parts.append(torch.cat([h_top, top_u[..., None], slot[..., None].expand(-1, -1, P, 1)], dim=-1)
                         .permute(0, 2, 1, 3).reshape(B, P, -1))
            for role in MISS_ROLES:
                m = (roles == ROLE_ID[role]) & present              # (B, N)
                has = m.any(1).to(h.dtype)[:, None]                 # (B, 1)
                h_max = h.masked_fill(~m[:, :, None, None], neg).amax(1) * has[..., None]
                u_lse = torch.logsumexp(u.masked_fill(~m[:, :, None], neg), dim=1) * has
                parts.append(torch.cat([h_max, u_lse[..., None], has[:, :, None].expand(-1, P, 1)], dim=-1))
            pm = present.to(h.dtype)
            parts.append((h * pm[:, :, None, None]).sum(1) / pm.sum(1)[:, None, None])
            counts = torch.stack([((roles == ROLE_ID[r]) & present).sum(1) for r in ROLE_ORDER], dim=1)
            parts.append((counts.to(h.dtype) / 9.0)[:, None, :].expand(-1, P, -1))
            t = self.rho(torch.cat(parts, dim=-1))
            return self.logit(t).squeeze(-1), self.outcome(t), t

    return CluePolicyNet()


def masked_log_policy(logits, legal):
    """log pi over the pool, -inf off the legal clues."""
    import torch

    return torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=-1)


def outcome_log_probs(outcome_logits, kmax):
    """Log-probabilities of the 13 outcomes, with the impossible ones (j >=
    K_max own words before a miss) masked out. `kmax` is (B,)."""
    import torch

    j = torch.arange(N_OUTCOMES, device=outcome_logits.device) // len(MISS_ROLES)
    ok = (j[None, :] < kmax[:, None]) | (torch.arange(N_OUTCOMES, device=outcome_logits.device) == CAP)[None, :]
    return torch.log_softmax(outcome_logits.masked_fill(~ok[:, None, :], float("-inf")), dim=-1)


def expected_rewards(outcome_logp, kmax, costs=None):
    """(B, P, 4) expected reward of announcing k = 1..4, from the outcome
    distribution; -inf where k > K_max. `costs` are the spymaster's prices
    for a neutral, opponent and assassin miss (default: the game's own)."""
    import torch

    if costs is None:
        costs = [-ROLE_REWARD[r] for r in MISS_ROLES]
    p = outcome_logp.exp()
    B, P, _ = p.shape
    miss = p[..., :CAP].reshape(B, P, MAX_CLUE_NUMBER, len(MISS_ROLES))      # (B, P, j, e)
    j = torch.arange(MAX_CLUE_NUMBER, device=p.device, dtype=p.dtype)
    value = (miss * (j[:, None] - torch.as_tensor(costs, device=p.device, dtype=p.dtype))).sum(-1)
    mass = miss.sum(-1)                                                      # (B, P, j)
    k = j + 1
    er = value.cumsum(-1) + (1.0 - mass.cumsum(-1)) * k
    ok = k[None, :] <= kmax[:, None].to(p.dtype)                             # (B, 4)
    return er.masked_fill(~ok[:, None, :], float("-inf"))


def save_policy(net, path: Path, meta: dict) -> None:
    import torch

    torch.save({"config": net.config, "state": net.state_dict(), "meta": meta}, path)


def load_policy(path: Path, device: str = "cpu"):
    import torch

    blob = torch.load(path, map_location=device, weights_only=False)
    net = build_net(**blob["config"]).to(device)
    net.load_state_dict(blob["state"])
    net.eval()
    return net, blob.get("meta", {})


# ---------------------------------------------------------------------------
# The incumbent's view, for the imitation targets


def incumbent_targets(sm, view, sims: SimilarityTensor) -> dict:
    """What `LearnedListenerSpymaster` would play here, and its expected reward
    for every (shortlisted clue, k): {"clue", "number", "score", "shortlist"
    (tensor clue indices), "net" (n, K_max)}.

    Scores the shortlist exactly as `_score_all_clues` does (same candidate
    order, features at K_max, one predict, `gain_and_penalty`) and picks with
    the incumbent's own `_pick_top_clues`. Recomputed here because
    `_score_all_clues` reduces each clue to its best k and the imitation
    targets need all of them; tests/test_clue_policy.py checks the pick
    matches `top_clues`.
    """
    from codenames.pl_reward import gain_and_penalty

    own = view.words_by_role(Role.OWN, unrevealed_only=True)
    others = [w for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN)
              for w in view.words_by_role(r, unrevealed_only=True)]
    costs = np.array([sm.costs[view.role_of(w)] for w in others], dtype=np.float64)
    _, g_scores, g_margin = sm._first_stage._score_all_clues(view, sims)
    finite = np.flatnonzero(np.isfinite(g_scores))
    take = finite[np.argsort(-g_scores[finite])[: sm.shortlist]]
    candidates = own + others
    K = min(len(own), MAX_CLUE_NUMBER)
    rows, keep = [], []
    for ci in take:
        f = sm._listener_features(sims.clue_words[ci], candidates, K, sims)
        if f is not None:
            rows.append(f)
            keep.append(int(ci))
    S = np.asarray(sm.bundle.booster.predict(np.vstack(rows), raw_score=True),
                   dtype=np.float64).reshape(len(keep), len(candidates))
    gain, penalty = gain_and_penalty(S[:, : len(own)], S[:, len(own):], costs, K)
    net = gain - penalty
    n_clues = len(sims.clue_words)
    best_n = np.ones(n_clues, dtype=np.int64)
    scores = np.full(n_clues, -np.inf, dtype=np.float32)
    margin = np.zeros(n_clues, dtype=np.float32)
    idx = np.asarray(keep)
    scores[idx] = net.max(axis=1)
    best_n[idx] = net.argmax(axis=1) + 1
    margin[idx] = g_margin[idx]
    clue, number, score = sm._first_stage._pick_top_clues(sims, view, best_n, scores, margin, 1)[0]
    return {"clue": clue, "number": number, "score": score, "shortlist": idx, "net": net}


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1))

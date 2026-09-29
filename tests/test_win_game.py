"""win_actor_critic's game engine and critic helpers (codenames/win_game.py,
codenames/win_critic.py): same rules as codenames.game, and the targets and
features do what their docstrings say."""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest

from codenames.board import Board, Role
from codenames.clue_policy import ROLE_ID, _training_vocab
from codenames.game import play_turn, play_two_team_game
from codenames.guessers.base import Guesser
from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.base import Spymaster
from codenames.win_critic import lambda_returns, relative_roles
from codenames.win_game import after_boards, make_board, play_attempts, play_game, revealed_ids, view_for


class _Shuffled(Guesser):
    """Ranks the candidates in an order fixed by the clue and the candidates
    alone, ignoring the number: both engines see the same ranking."""

    def score_candidates(self, clue, candidate_words, sims):
        raise NotImplementedError

    def rank_candidates(self, clue, candidate_words, sims, number=None):
        words = sorted(candidate_words)
        random.Random(hash((clue, tuple(words))) & 0xFFFF).shuffle(words)
        return words


class _Counting(Spymaster):
    """A clue that names the turn, and a number that cycles 1..3."""

    def top_clues(self, ctx, sims, k):
        n = len(ctx.board.revealed)
        return [(f"clue{n}", 1 + n % 3, 0.0)]


def test_play_game_matches_play_two_team_game():
    sm, g = _Counting(), _Shuffled()

    def agent(seed, revealed, team):
        board = make_board(seed, revealed)
        clue, number = sm.give_clue(_ctx(view_for(board, team)), None)
        return [{"clue": clue, "k": number, "pool": 0, "kmax": 9}]

    def opponent(seed, revealed, team):
        board = make_board(seed, revealed)
        return sm.give_clue(_ctx(view_for(board, team)), None)

    for seed in range(40):
        for team in ("A", "B"):
            rec = play_game(seed, team, agent, opponent, g)
            ref = play_two_team_game(make_board(seed), (sm, g), (sm, g), sims=None)
            assert rec.error is None
            assert rec.winner == ref.winner
            assert [(t.clue, t.number, t.guesses) for t in rec.turns] == \
                   [(t.turn.clue, t.turn.number, [w for w, _ in t.turn.guesses]) for t in ref.turns]


def _ctx(view):
    from codenames.spymasters.base import TurnContext

    return TurnContext(board=view, turn_index=len(view.revealed))


def test_after_boards_replay_each_k():
    rng = random.Random(0)
    for seed in range(30):
        board = make_board(seed)
        pre = rng.sample(range(25), rng.randint(0, 6))
        board.revealed = {board.words[i] for i in pre}
        if board.remaining(Role.OWN) == 0 or Role.ASSASSIN in {board.role_of(w) for w in board.revealed}:
            continue
        team = "A" if seed % 2 == 0 else "B"
        ranking = [w for w in board.words if not board.is_revealed(w)]
        rng.shuffle(ranking)
        view = view_for(board, team)
        kmax = view.remaining(Role.OWN)
        got = after_boards(seed, revealed_ids(board), team, ranking, kmax)
        for k in range(1, kmax + 1):
            b = copy.deepcopy(board)
            play_attempts(view_for(b, team), ranking[:k])
            assert got[k - 1][0] == revealed_ids(b)


def test_play_attempts_matches_play_turn():
    class Fixed(Spymaster):
        def __init__(self, k):
            self.k = k

        def top_clues(self, ctx, sims, k):
            return [("x", self.k, 0.0)]

    class Ranked(Guesser):
        def __init__(self, order):
            self.order = order

        def score_candidates(self, clue, candidate_words, sims):
            raise NotImplementedError

        def rank_candidates(self, clue, candidate_words, sims, number=None):
            return [w for w in self.order if w in candidate_words]

    rng = random.Random(1)
    for seed in range(30):
        board = Board.generate(seed=seed, vocabulary=list(_training_vocab()))
        order = list(board.words)
        rng.shuffle(order)
        for k in range(1, 6):
            a, b = copy.deepcopy(board), copy.deepcopy(board)
            turn = play_turn(a, Fixed(k), Ranked(order), sims=None)
            guesses, ended = play_attempts(b, [w for w in order if not b.is_revealed(w)][:k])
            assert [w for w, _ in guesses] == [w for w, _ in turn.guesses]
            assert ended == turn.ended_reason
            assert a.revealed == b.revealed


def test_lambda_returns():
    v_next = np.array([0.3, 0.6, 0.2])
    # Monte Carlo: the result, flipping side every half-turn.
    assert lambda_returns(v_next, 1.0, 1.0).tolist() == [0.0, 1.0, 0.0, 1.0]
    # One-step TD: 1 - V(next), and the result at the end.
    assert lambda_returns(v_next, 0.0, 0.0) == pytest.approx([0.7, 0.4, 0.8, 0.0])


def test_relative_roles_swap_own_and_opponent_for_b():
    roles = np.array([ROLE_ID[r] for r in (Role.OWN, Role.OPPONENT, Role.NEUTRAL, Role.ASSASSIN)])
    assert relative_roles(roles, "A").tolist() == roles.tolist()
    assert relative_roles(roles, "B").tolist() == [ROLE_ID[Role.OPPONENT], ROLE_ID[Role.OWN],
                                                   ROLE_ID[Role.NEUTRAL], ROLE_ID[Role.ASSASSIN]]


NEEDS = ["policy_features.npy", "policy_features_aux.npz"]
needs_gpu_features = pytest.mark.skipif(
    any(not (DEFAULT_CACHE_DIR / f).exists() for f in NEEDS)
    or not pytest.importorskip("torch").cuda.is_available(),
    reason="needs cache/policy_features* and a GPU")


@needs_gpu_features
def test_edge_features_brute_force_and_critic_slot_invariance():
    import torch

    from codenames.clue_policy import DeviceFeatures, PolicyFeatures
    from codenames.win_critic import IU, NEAR, PER_SOURCE, SOURCES, CriticFeatures, build_critic

    feats = PolicyFeatures.load()
    gf = DeviceFeatures(feats, "cuda")
    cf = CriticFeatures(feats, gf)
    board = make_board(7, frozenset({3, 11}))
    b = feats.encode(board)
    e = cf.edges(b.words[None], b.roles[None], b.present[None]).float().cpu().numpy()[0]   # (300, F)

    # One pair, one source, by brute force.
    s = SOURCES.index("z_glove")
    col = feats.pair_names.index("z_glove")
    X = np.asarray(feats.pair[b.words][:, :, col], dtype=np.float32)          # (25, P)
    X[:, ~b.legal] = -1e4
    i, j = 0, 1
    m = np.minimum(X[i], X[j])
    c = int(m.argmax())
    level = m[c]
    row = np.flatnonzero((IU[0] == i) & (IU[1] == j))[0]
    f = e[row, s * PER_SOURCE:(s + 1) * PER_SOURCE]
    assert f[0] == pytest.approx(min(level, 8.0), abs=0.02)
    for r in range(4):
        others = [w for w in range(25) if w not in (i, j) and b.present[w] and b.roles[w] == r]
        if others:
            assert f[1 + r] == pytest.approx(np.clip(level - X[others, c].max(), -8, 8), abs=0.02)
            assert f[5 + r] == pytest.approx(sum(X[w, c] >= level - NEAR for w in others) / 4.0, abs=1e-6)
    # A revealed word's pairs are zero.
    row = np.flatnonzero((IU[0] == 3) & (IU[1] == 4))[0]
    assert not e[row].any()

    # The critic does not care which slot holds which word.
    torch.manual_seed(0)
    critic = build_critic(len(feats.word_names)).cuda().eval()
    torch.nn.init.normal_(critic.read[-1].weight)      # the board correction starts at zero; wake it
    perm = np.random.default_rng(0).permutation(25)
    outs = []
    for order in (np.arange(25), perm):
        w, r, p = b.words[order][None], b.roles[order][None], b.present[order][None]
        edges = cf.edges(w, r, p)
        with torch.no_grad():
            outs.append(float(critic(gf.word[torch.as_tensor(w, device="cuda")], torch.as_tensor(r, device="cuda"),
                                     torch.as_tensor(p, device="cuda"), edges, torch.tensor([True], device="cuda"))))
    assert outs[0] == pytest.approx(outs[1], abs=1e-3)

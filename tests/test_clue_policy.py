"""The clue policy's shared machinery (codenames/clue_policy.py).

Three things training and play must agree on:
- the legal-clue mask the policy's softmax runs over is `is_legal_clue`;
- the rewards read off one guesser ranking for k = 1..4 are what
  `game.play_turn` would score, and so is the outcome category they are
  compressed into, and the head's expected reward of a certain outcome;
- the imitation targets pick what the incumbent plays.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest

from codenames.board import Board, OpponentBoardView, Role, clue_number_cap, is_legal_clue, load_holdout_wordlist
from codenames.clue_policy import (
    POLICY_MAX_NUMBER,
    cap_of,
    deal_position,
    k_max,
    outcome_of,
    n_outcomes,
    reward_of_outcome,
    role_map,
    turn_reward,
)
from codenames.game import play_turn
from codenames.guessers.base import Guesser
from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.base import Spymaster


class _Fixed(Spymaster):
    def __init__(self, number):
        self.number = number

    def top_clues(self, ctx, sims, k):
        return [("clue", self.number, 0.0)]


class _Ranked(Guesser):
    def __init__(self, ranking):
        self.ranking = ranking

    def score_candidates(self, clue, candidate_words, sims):
        raise NotImplementedError

    def rank_candidates(self, clue, candidate_words, sims, number=None):
        assert sorted(candidate_words) == sorted(self.ranking)
        return list(self.ranking)


def _views(n):
    out, seed = [], 0
    while len(out) < n:
        v = deal_position(seed)
        seed += 1
        if v is not None:
            out.append(v)
    return out


def test_deal_position_uses_training_words_and_both_sides():
    held_out = set(load_holdout_wordlist())
    views = _views(40)
    assert not any(w in held_out for v in views for w in v.words)
    assert any(isinstance(v, OpponentBoardView) for v in views)
    assert any(isinstance(v, Board) for v in views)
    assert all(v.remaining(Role.OWN) >= 2 for v in views)
    assert any(len(v.revealed) > 0 for v in views)


def test_rewards_from_one_ranking_match_play_turn():
    rng = random.Random(0)
    for view in _views(60):
        ranking = [w for w in view.words if not view.is_revealed(w)]
        rng.shuffle(ranking)
        # Half the time put some own words first, so long runs are covered.
        if rng.random() < 0.5:
            own = [w for w in ranking if view.role_of(w) is Role.OWN]
            ranking = own[: rng.randint(1, len(own))] + [w for w in ranking if w not in own[: len(own)]]
            ranking = list(dict.fromkeys(ranking + own))
        roles = role_map(view)
        M = POLICY_MAX_NUMBER
        kmax = k_max(view, M)
        assert kmax == view.remaining(Role.OWN)            # uncapped: every own word left
        cat = outcome_of(ranking, roles, kmax, M)
        assert 0 <= cat < n_outcomes(M)
        for k in range(1, kmax + 1):
            board = copy.deepcopy(view)
            turn = play_turn(board, _Fixed(k), _Ranked(ranking), sims=None)
            assert turn_reward(ranking, roles, k) == pytest.approx(turn.reward)
            assert reward_of_outcome(cat, k, M) == pytest.approx(turn.reward)


def test_clue_number_cap():
    assert clue_number_cap(7) == 4                          # the game's historical cap, still the default
    assert clue_number_cap(3) == 3
    assert clue_number_cap(7, None) == 7                    # uncapped: every own word left


@pytest.mark.parametrize("M", [4, POLICY_MAX_NUMBER])
def test_outcome_categories(M):
    roles = {"a": Role.OWN, "b": Role.OWN, "n": Role.NEUTRAL, "o": Role.OPPONENT, "x": Role.ASSASSIN}
    assert outcome_of(["x", "a", "b"], roles, 2, M) == 0 * 3 + 2
    assert outcome_of(["a", "n", "b"], roles, 2, M) == 1 * 3 + 0
    assert outcome_of(["a", "b", "o"], roles, 2, M) == cap_of(M)
    assert reward_of_outcome(1 * 3 + 1, 1, M) == 1.0       # the miss comes after the one guess
    assert reward_of_outcome(1 * 3 + 1, 2, M) == 0.0       # +1, then -1 on the opponent word


@pytest.mark.parametrize("M", [4, POLICY_MAX_NUMBER])
def test_expected_rewards_of_a_certain_outcome(M):
    torch = pytest.importorskip("torch")
    from codenames.clue_policy import expected_rewards, outcome_log_probs

    for kmax in range(1, M + 1):
        for cat in range(n_outcomes(M)):
            if cat != cap_of(M) and cat // 3 >= kmax:
                continue
            logits = torch.full((1, 1, n_outcomes(M)), -60.0)
            logits[0, 0, cat] = 60.0
            er = expected_rewards(outcome_log_probs(logits, torch.tensor([kmax])), torch.tensor([kmax]))
            for k in range(1, M + 1):
                want = reward_of_outcome(cat, k, M) if k <= kmax else float("-inf")
                assert float(er[0, 0, k - 1]) == pytest.approx(want, abs=1e-5)


NEEDS = ["policy_features.npy", "policy_features_aux.npz", "similarity_tensor.npy"]
needs_features = pytest.mark.skipif(any(not (DEFAULT_CACHE_DIR / f).exists() for f in NEEDS),
                                    reason="needs cache/policy_features* (scripts/data/build_policy_features.py)")


@needs_features
def test_legality_mask_matches_is_legal_clue():
    from codenames.clue_policy import PolicyFeatures

    feats = PolicyFeatures.load()
    boards = _views(3) + [Board.generate(seed=s, vocabulary=load_holdout_wordlist()) for s in (0, 1)]
    n_illegal = 0
    for view in boards:
        legal = feats.encode(view).legal
        want = np.array([is_legal_clue(c, view.words) for c in feats.clue_words])
        assert np.array_equal(legal, want)
        n_illegal += int((~want).sum())
    assert n_illegal > 0            # the mask is actually doing something


@needs_features
def test_network_is_invariant_to_slot_order():
    torch = pytest.importorskip("torch")
    from codenames.clue_policy import PolicyFeatures, build_net, stack_inputs

    feats = PolicyFeatures.load()
    torch.manual_seed(0)
    net = build_net(len(feats.pair_names), len(feats.word_names), max_number=POLICY_MAX_NUMBER).eval()
    b = feats.encode(_views(1)[0])
    perm = np.random.default_rng(0).permutation(25)
    shuffled = type(b)(words=b.words[perm], roles=b.roles[perm], present=b.present[perm], legal=b.legal)
    outs = []
    for x in (b, shuffled):
        pair, word, roles, present, _ = stack_inputs(feats, [x])
        with torch.no_grad():
            logit, outcome, _ = net(torch.as_tensor(pair[:, :, :500], dtype=torch.float32),
                                    torch.as_tensor(word), torch.as_tensor(roles), torch.as_tensor(present))
        outs.append((logit, outcome))
    assert torch.allclose(outs[0][0], outs[1][0], atol=1e-4)
    assert torch.allclose(outs[0][1], outs[1][1], atol=1e-4)


INCUMBENT = ["listener_gbt.txt", "clue_stats.npz", "swow.npz"]


@pytest.mark.skipif(any(not (DEFAULT_CACHE_DIR / f).exists() for f in INCUMBENT),
                    reason="needs the incumbent's listener in cache/")
@pytest.mark.parametrize("max_number", [4, None])
def test_incumbent_targets_pick_what_the_incumbent_plays(max_number):
    from codenames.clue_policy import incumbent_targets
    from codenames.similarity import SimilarityTensor
    from codenames.spymasters.base import TurnContext
    from codenames.spymasters.learned_listener import LearnedListenerSpymaster

    sims = SimilarityTensor.load()
    sm = LearnedListenerSpymaster(max_number=max_number)
    for view in _views(3):
        t = incumbent_targets(sm, view, sims)
        clue, number, score = sm.top_clues(TurnContext(view, 0), sims, 1)[0]
        assert (t["clue"], t["number"]) == (clue, number)
        assert t["score"] == pytest.approx(score, abs=1e-5)
        row = list(t["shortlist"]).index(sims.clue_index[clue.lower()])
        assert t["net"][row].max() == pytest.approx(score, abs=1e-5)

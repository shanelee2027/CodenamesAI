"""`LearnedListenerSpymaster.listen` is the search's own view of one clue.

The /compare page (scripts/tools/play_server.py) shows intended targets and
guess probabilities from `listen`; if that drifted from what `_score_all_clues`
computes, the page would explain clues with numbers they were not chosen on.
So: rebuild the reward from `listen`'s scores and require the search's score
for its best clue back. Needs the real model files, and skips without them.
"""

from __future__ import annotations

import numpy as np
import pytest

from codenames.board import Board, Role
from codenames.similarity import DEFAULT_CACHE_DIR

NEEDS = ["similarity_tensor.npy", "listener_gbt.txt", "listener_gbt_decoy.txt",
         "listener_gbt_assoc_w0.3.txt", "listener_gbt_assoc_w0.3.assoc.json"]
pytestmark = pytest.mark.skipif(any(not (DEFAULT_CACHE_DIR / f).exists() for f in NEEDS),
                                reason="needs the trained listeners in cache/")


@pytest.fixture(scope="module")
def sims():
    from codenames.similarity import SimilarityTensor
    return SimilarityTensor.load(DEFAULT_CACHE_DIR)


def _spymasters():
    from codenames.spymasters.association_listener import AssociationListenerSpymaster
    from codenames.spymasters.learned_listener import LearnedListenerSpymaster
    return [
        ("incumbent", lambda: LearnedListenerSpymaster()),
        ("decoy_out25", lambda: LearnedListenerSpymaster(
            outside_n=25, model_path=DEFAULT_CACHE_DIR / "listener_gbt_decoy.txt")),
        ("assoc_pass", lambda: AssociationListenerSpymaster(pass_rate=0.05)),
    ]


@pytest.mark.parametrize("name,make", _spymasters(), ids=lambda x: x if isinstance(x, str) else "")
def test_listen_reproduces_the_search_score(name, make, sims):
    from codenames.pl_reward import gain_and_penalty
    from codenames.spymasters.base import MAX_CLUE_NUMBER, TurnContext

    sm = make()
    for seed in (11, 13):
        board = Board.generate(seed=seed)
        clue, number, score = sm.top_clues(TurnContext(board, 0), sims, 1)[0]
        got = sm.listen(board, clue, sims)
        n_own = sum(1 for r in got["roles"] if r is Role.OWN)
        costs = np.array([sm.costs[r] for r in got["roles"][n_own:]])
        s = got["scores"][None, :]
        s_out = None if got["outside"] is None else np.array([got["outside"]])
        gain, penalty = gain_and_penalty(s[:, :n_own], s[:, n_own:], costs,
                                         min(n_own, MAX_CLUE_NUMBER), s_out=s_out)
        assert (got["outside"] is None) == (name == "incumbent")
        assert (gain - penalty)[0, number - 1] == pytest.approx(score, abs=1e-4)

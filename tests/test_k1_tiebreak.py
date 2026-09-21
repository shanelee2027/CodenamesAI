"""The k=1 similarity tiebreak, and the acronym pool restriction.

The tiebreak exists because expected reward saturates at k=1: once one own
word dominates, every safe clue scores within noise of the best, so the argmax
is decided by the third decimal instead of by which clue a teammate would
actually get.

The safety property is the part worth testing. An earlier version took the
highest-similarity legal clue outright and was removed for a reason that is
not hypothetical: with CHICK and EAGLE both on the board and EAGLE the last
word needed, the most obvious clue for EAGLE alone is BIRD, which hands CHICK
to whoever owns it. Raw similarity cannot see CHICK; expected reward prices
it, which is why candidates must clear the tolerance first.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from codenames.acronyms import load_acronym_mask
from codenames.board import Board, Card, Role
from codenames.similarity import SimilarityTensor
from codenames.spymasters.learned_listener import LearnedListenerSpymaster

BOARD_WORDS = ["Eagle", "Chick", "Hawk", "Bear", "Death"] + [f"Pad{i}" for i in range(20)]
CLUES = ["bald", "patriot", "owl", "eagles"]
SPACE = "numberbatch"


@pytest.fixture
def sims(tmp_path):
    """Cosines chosen so BIRD-ish `owl` is closest to Eagle overall, but is
    also close to Chick -- the trap the tolerance has to catch."""
    n_clue, n_board = len(CLUES), len(BOARD_WORDS)
    t = np.full((n_clue, n_board, 3), 0.01, dtype=np.float32)
    idx = {w: i for i, w in enumerate(BOARD_WORDS)}
    t[CLUES.index("owl"), idx["Eagle"], :] = 0.90      # most similar to Eagle...
    t[CLUES.index("owl"), idx["Chick"], :] = 0.80      # ...and to Chick as well
    t[CLUES.index("patriot"), idx["Eagle"], :] = 0.70
    t[CLUES.index("bald"), idx["Eagle"], :] = 0.60
    t[CLUES.index("eagles"), idx["Eagle"], :] = 0.95   # illegal: shares a stem
    np.save(tmp_path / "similarity_tensor.npy", t.astype(np.float16))
    (tmp_path / "clue_vocab.json").write_text(json.dumps(CLUES))
    (tmp_path / "board_vocab.json").write_text(json.dumps(BOARD_WORDS))
    (tmp_path / "similarity_meta.json").write_text(
        json.dumps({"spaces": [SPACE, "glove", "wiki"], "shape": list(t.shape)}))
    return SimilarityTensor.load(cache_dir=tmp_path)


@pytest.fixture
def board():
    roles = [Role.OWN, Role.OPPONENT, Role.OPPONENT, Role.NEUTRAL, Role.ASSASSIN]
    roles += [Role.NEUTRAL] * (len(BOARD_WORDS) - len(roles))
    return Board(cards=tuple(Card(word=w, role=r) for w, r in zip(BOARD_WORDS, roles)), seed=0)


def run_swap(sims, board, tolerance, scores):
    """Call the tiebreak in isolation -- it reads only `k1_tie_tolerance`."""
    words = ["Eagle", "Chick", "Hawk", "Bear", "Death"]
    S = np.zeros((len(CLUES), len(words)))
    S[:, 0] = 5.0                                    # every clue means Eagle
    sm = SimpleNamespace(k1_tie_tolerance=tolerance)
    return LearnedListenerSpymaster._swap_k1(
        sm, sims, board, words=words, own_n=1,
        scores=np.asarray(scores, dtype=np.float64),
        best_n=np.ones(len(CLUES), dtype=np.int64),
        S=S, keep=list(range(len(CLUES))))


class TestTheTieSetIsTheSafeguard:
    def test_a_more_obvious_clue_inside_the_tolerance_is_taken(self, sims, board):
        """patriot is 0.1 worse but much closer to Eagle, so it should win."""
        scores = [1.00, 0.95, 0.20, 1.50]            # bald best legal; owl far below
        out = run_swap(sims, board, 0.5, scores)
        assert out is not None
        new_scores, _ = out
        assert CLUES[int(np.argmax(new_scores))] == "patriot"

    def test_a_clue_that_also_points_at_the_opponent_word_is_out_of_reach(self, sims, board):
        """owl is the closest clue to Eagle of all, and the board-aware reward
        has already marked it down for pointing at Chick too. A tolerance that
        cannot reach it is what stops the CHICK/EAGLE failure."""
        scores = [1.00, 0.95, 0.20, 1.50]            # owl 0.8 below the best
        new_scores, _ = run_swap(sims, board, 0.5, scores)
        assert CLUES[int(np.argmax(new_scores))] != "owl"

    def test_a_wide_enough_tolerance_does_reach_it(self, sims, board):
        """The tolerance is the whole control: widen it and the unsafe clue
        comes back. This is why the default is not larger."""
        scores = [1.00, 0.95, 0.20, 1.50]
        new_scores, _ = run_swap(sims, board, 2.0, scores)
        assert CLUES[int(np.argmax(new_scores))] == "owl"

    def test_the_default_tolerance_is_a_small_fraction_of_a_turn(self):
        """The unit is own-words and a k=1 turn is worth about one, so the
        default has to be read as a share of the turn it is spent on. 0.5 was
        the first default and allowed giving up 38% of a turn, which is not a
        tie between clues but a decision to play a worse one."""
        import inspect
        default = inspect.signature(
            LearnedListenerSpymaster.__init__).parameters["k1_tie_tolerance"].default
        assert default <= 0.15, "a tiebreak must not be able to spend a sixth of the turn"

    def test_an_illegal_clue_never_wins_however_similar(self, sims, board):
        """`eagles` shares a stem with Eagle, which is exactly why it scores
        best. The clue actually played is the best LEGAL one."""
        scores = [1.00, 0.95, 0.90, 1.50]
        out = run_swap(sims, board, 2.0, scores)
        picked = CLUES[int(np.argmax(out[0]))] if out else "bald"
        assert picked != "eagles"

    def test_no_swap_when_the_incumbent_is_already_the_most_similar(self, sims, board):
        """Returning None rather than a no-op swap keeps the caller's arrays
        untouched, so nothing downstream sees an inflated score."""
        scores = [0.20, 0.30, 1.00, 0.10]            # owl is best AND most similar
        assert run_swap(sims, board, 0.05, scores) is None


class TestAcronymMask:
    def test_returns_none_when_the_artifact_is_absent(self, tmp_path):
        """A fresh checkout must still play, acronyms and all."""
        assert load_acronym_mask(tmp_path, ["bird", "cia"]) is None

    def test_flags_by_word_not_by_position(self, tmp_path):
        """Keyed by word so any vocabulary works, including a synthetic one a
        test injects. A positional array with a length check rejected those,
        and would still have mismatched two same-sized vocabularies in
        different orders without noticing."""
        np.savez_compressed(tmp_path / "acronym_mask.npz",
                            clue_words=np.array(["bird", "cia", "phd"], dtype=object),
                            mask=np.array([False, True, True]))
        got = load_acronym_mask(tmp_path, ["phd", "bird", "unseen", "cia"])
        assert list(got) == [True, False, False, True]

    def test_an_unknown_vocabulary_is_simply_unflagged(self, tmp_path):
        np.savez_compressed(tmp_path / "acronym_mask.npz",
                            clue_words=np.array(["cia"], dtype=object),
                            mask=np.array([True]))
        assert not load_acronym_mask(tmp_path, ["bird", "cake"]).any()


class TestAcronymDetection:
    """The rules in scripts/data/build_acronym_mask.py, spot-checked."""

    @staticmethod
    def detect(word):
        import importlib.util
        from pathlib import Path
        from nltk.corpus import wordnet as wn
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "_am", root / "scripts" / "data" / "build_acronym_mask.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.is_acronym(word, wn)

    @pytest.mark.parametrize("word", ["cia", "phd", "nba", "hsbc", "gm", "td", "usa",
                                      "nasa", "fbi", "dvd", "ceo", "wwe", "cpu"])
    def test_acronyms_are_flagged(self, word):
        assert self.detect(word)

    @pytest.mark.parametrize("word", ["bird", "eagle", "cat", "pet", "sky", "rhythm",
                                      "zip", "shape", "australia", "germany", "limbs",
                                      "gods", "mice", "smartphone"])
    def test_ordinary_words_are_not(self, word):
        assert not self.detect(word)

    @pytest.mark.parametrize("word", ["laser", "radar", "scuba"])
    def test_lexicalised_acronyms_survive(self, word):
        """Acronyms historically, ordinary words now -- WordNet spells them in
        lower case, which is exactly the signal the rule reads."""
        assert not self.detect(word)

from __future__ import annotations

import pytest

from codenames.board import (
    BOARD_SIZE,
    ROLE_COUNTS,
    Board,
    Role,
    is_legal_clue,
    load_holdout_wordlist,
    load_training_wordlist,
    load_wordlist,
)

VOCAB = load_wordlist()


class TestHoldoutWordlist:
    def test_holdout_words_are_a_subset_of_the_full_vocabulary(self):
        holdout = load_holdout_wordlist()
        assert set(holdout) <= set(VOCAB)

    def test_holdout_and_training_partition_the_full_vocabulary(self):
        holdout = load_holdout_wordlist()
        training = load_training_wordlist()
        assert set(holdout) & set(training) == set()
        assert set(holdout) | set(training) == set(VOCAB)
        assert len(holdout) + len(training) == len(VOCAB)

    def test_training_wordlist_excludes_every_holdout_word(self):
        holdout = set(load_holdout_wordlist())
        training = load_training_wordlist()
        assert not any(w in holdout for w in training)

    def test_holdout_set_is_large_enough_to_build_a_board_from_alone(self):
        assert len(load_holdout_wordlist()) >= BOARD_SIZE

    def test_training_set_is_large_enough_to_build_a_board_from_alone(self):
        assert len(load_training_wordlist()) >= BOARD_SIZE


class TestBoardGeneration:
    def test_deterministic_for_same_seed(self):
        a = Board.generate(seed=42)
        b = Board.generate(seed=42)
        assert a.words == b.words
        assert [c.role for c in a.cards] == [c.role for c in b.cards]

    def test_different_seeds_differ(self):
        a = Board.generate(seed=1)
        b = Board.generate(seed=2)
        assert a.words != b.words

    def test_role_counts(self):
        board = Board.generate(seed=0)
        for role, count in ROLE_COUNTS.items():
            assert len(board.words_by_role(role)) == count

    def test_board_size(self):
        board = Board.generate(seed=0)
        assert len(board.words) == BOARD_SIZE
        assert len(set(board.words)) == BOARD_SIZE  # no duplicate words

    def test_vocabulary_too_small_raises(self):
        with pytest.raises(ValueError):
            Board.generate(seed=0, vocabulary=["a", "b", "c"])

    def test_wrong_card_count_raises(self):
        with pytest.raises(ValueError):
            Board(cards=(), seed=0)


class TestRevealAndRoles:
    def test_reveal_returns_role_and_marks_revealed(self):
        board = Board.generate(seed=0)
        word = board.words[0]
        expected_role = board.role_of(word)
        assert not board.is_revealed(word)
        assert board.reveal(word) == expected_role
        assert board.is_revealed(word)

    def test_reveal_unknown_word_raises(self):
        board = Board.generate(seed=0)
        with pytest.raises(KeyError):
            board.reveal("not-a-real-board-word-xyz")

    def test_role_of_unknown_word_raises(self):
        board = Board.generate(seed=0)
        with pytest.raises(KeyError):
            board.role_of("not-a-real-board-word-xyz")

    def test_remaining_decreases_after_reveal(self):
        board = Board.generate(seed=0)
        own_word = board.words_by_role(Role.OWN)[0]
        before = board.remaining(Role.OWN)
        board.reveal(own_word)
        assert board.remaining(Role.OWN) == before - 1

    def test_words_by_role_unrevealed_only(self):
        board = Board.generate(seed=0)
        own_words = board.words_by_role(Role.OWN)
        board.reveal(own_words[0])
        unrevealed = board.words_by_role(Role.OWN, unrevealed_only=True)
        assert own_words[0] not in unrevealed
        assert len(unrevealed) == len(own_words) - 1

    def test_lookups_are_case_insensitive(self):
        board = Board.generate(seed=0)
        word = board.words[0]
        assert board.role_of(word.upper()) == board.role_of(word)
        assert board.role_of(word.lower()) == board.role_of(word)

    def test_reveal_with_different_case_is_consistent(self):
        # reveal() must record the card's canonical casing, not the
        # caller's -- otherwise is_revealed()/words_by_role() using the
        # canonical casing would disagree with what was just revealed.
        board = Board.generate(seed=0)
        word = board.words_by_role(Role.OWN)[0]
        board.reveal(word.upper())
        assert board.is_revealed(word)
        assert board.is_revealed(word.upper())
        assert board.is_revealed(word.lower())
        assert word not in board.words_by_role(Role.OWN, unrevealed_only=True)


class TestClueLegality:
    def test_unrelated_clue_is_legal(self):
        assert is_legal_clue("elephant", ["apple", "car", "moon"])

    def test_exact_board_word_is_illegal(self):
        assert not is_legal_clue("apple", ["apple", "car", "moon"])

    def test_exact_match_is_case_insensitive(self):
        assert not is_legal_clue("APPLE", ["apple", "car", "moon"])
        assert not is_legal_clue("apple", ["APPLE", "car", "moon"])

    def test_plural_of_board_word_is_illegal(self):
        # "apple" (clue) is a substring of "apples" (board word)
        assert not is_legal_clue("apple", ["apples", "car", "moon"])

    def test_singular_clue_against_plural_board_word_direction(self):
        # board word is the plural form; clue is the base form -- reverse of
        # the usual example, still must be caught
        assert not is_legal_clue("car", ["cars", "apple", "moon"])

    def test_plural_clue_against_singular_board_word(self):
        # clue is the plural form of a singular board word
        assert not is_legal_clue("cars", ["car", "apple", "moon"])

    def test_y_to_i_stem_variant_is_illegal(self):
        # "happier" contains the y->i stem of "happy" ("happi"), not a
        # plain substring relationship -- this is exactly what
        # _stem_variants exists to catch.
        assert not is_legal_clue("happier", ["happy", "car", "moon"])
        assert not is_legal_clue("happy", ["happier", "car", "moon"])

    def test_hyphenated_board_word_blocks_either_half(self):
        assert not is_legal_clue("ray", ["x-ray", "car", "moon"])
        assert not is_legal_clue("x", ["x-ray", "car", "moon"])

    def test_multi_word_board_entry_blocks_either_word(self):
        # "New york" is a real entry in the shipped word list
        assert not is_legal_clue("york", ["New york", "car", "moon"])
        assert not is_legal_clue("new", ["New york", "car", "moon"])

    def test_legal_clue_against_multi_word_board_entry(self):
        assert is_legal_clue("city", ["New york", "car", "moon"])

    def test_checks_all_board_words_not_just_first(self):
        assert not is_legal_clue("moon", ["apple", "car", "moon"])


class TestWordlistAsset:
    def test_shipped_wordlist_has_400_unique_words(self):
        assert len(VOCAB) == 400
        assert len(set(VOCAB)) == 400

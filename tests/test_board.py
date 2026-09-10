from __future__ import annotations

import pytest

from codenames.board import (
    BOARD_SIZE,
    ROLE_COUNTS,
    Board,
    OpponentBoardView,
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

    def test_holdout_is_exactly_150_words(self):
        # docs/iteration-architecture.md step 5: raised from 60 to 150.
        assert len(load_holdout_wordlist()) == 150

    def test_original_60_word_holdout_is_still_present(self):
        # The pre-step-5 holdout must be a strict subset of the new one
        # (random.Random(42).sample(VOCAB, 60) -- see docs/log.md), so a
        # model trained under the old split is evaluated against a
        # superset of what it was told to avoid.
        import random

        original_60 = set(random.Random(42).sample(sorted(VOCAB), 60))
        assert original_60 <= set(load_holdout_wordlist())

    def test_holdout_selection_is_reproducible_from_the_recorded_snippet(self):
        # Regenerates codenames/assets/board_words_holdout.txt byte for
        # byte from the exact snippet recorded in docs/log.md.
        import random

        vocab = sorted(VOCAB)
        original_60 = set(random.Random(42).sample(vocab, 60))
        remaining = [w for w in vocab if w not in original_60]
        assert len(remaining) == 340
        new_90 = random.Random(43).sample(remaining, 90)
        regenerated = sorted(original_60 | set(new_90))

        from codenames.board import ASSET_HOLDOUT_WORDLIST_PATH

        committed_bytes = ASSET_HOLDOUT_WORDLIST_PATH.read_bytes()
        regenerated_bytes = ("\n".join(regenerated) + "\n").encode()
        assert regenerated_bytes == committed_bytes


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


class TestOpponentBoardView:
    def test_own_and_opponent_swap(self):
        board = Board.generate(seed=1)
        view = OpponentBoardView(board)
        for word in board.words:
            expected = {Role.OWN: Role.OPPONENT, Role.OPPONENT: Role.OWN}.get(
                board.role_of(word), board.role_of(word)
            )
            assert view.role_of(word) == expected

    def test_neutral_and_assassin_unchanged(self):
        board = Board.generate(seed=1)
        view = OpponentBoardView(board)
        for word in board.words_by_role(Role.NEUTRAL) + board.words_by_role(Role.ASSASSIN):
            assert view.role_of(word) == board.role_of(word)

    def test_remaining_counts_are_swapped(self):
        board = Board.generate(seed=1)
        view = OpponentBoardView(board)
        assert view.remaining(Role.OWN) == board.remaining(Role.OPPONENT) == ROLE_COUNTS[Role.OPPONENT]
        assert view.remaining(Role.OPPONENT) == board.remaining(Role.OWN) == ROLE_COUNTS[Role.OWN]

    def test_words_by_role_is_swapped(self):
        board = Board.generate(seed=1)
        view = OpponentBoardView(board)
        assert set(view.words_by_role(Role.OWN)) == set(board.words_by_role(Role.OPPONENT))

    def test_reveal_is_shared_with_the_underlying_board(self):
        # The whole point of a *view*, not a copy: revealing through one
        # side is immediately visible through the other, same physical
        # revealed-state.
        board = Board.generate(seed=1)
        view = OpponentBoardView(board)
        word = board.words[0]
        assert not board.is_revealed(word)
        assert not view.is_revealed(word)

        view.reveal(word)
        assert board.is_revealed(word)
        assert view.is_revealed(word)

    def test_reveal_returns_the_swapped_role(self):
        board = Board.generate(seed=1)
        view = OpponentBoardView(board)
        own_word = board.words_by_role(Role.OWN)[0]
        assert view.reveal(own_word) == Role.OPPONENT
        assert board.role_of(own_word) == Role.OWN  # the underlying Card is never mutated

    def test_words_and_seed_pass_through_unchanged(self):
        board = Board.generate(seed=7)
        view = OpponentBoardView(board)
        assert view.words == board.words
        assert view.seed == board.seed

    def test_revealed_set_is_shared_not_copied(self):
        # Some spymaster code reads board.revealed directly (e.g.
        # codenames/spymasters/_util.py::state_rng, LearnedSpymaster's
        # turn-index calc) rather than going through is_revealed() --
        # needs to see the same set, live, from either perspective.
        board = Board.generate(seed=7)
        view = OpponentBoardView(board)
        word = board.words[0]
        board.reveal(word)
        assert word in view.revealed
        assert view.revealed is board.revealed


class TestLegalityDerivationalForms:
    """The shared-prefix rule. Substring cannot see these: neither
    "mexico" nor "mexican" contains the other, yet on 60 measured boards
    10% of the baseline's chosen clues were of exactly this kind, each
    scoring z > +8 precisely because it was the same word."""

    @pytest.mark.parametrize(
        "clue,word",
        [
            ("mexican", "Mexico"),
            ("canadian", "Canada"),
            ("changing", "Change"),
            ("charging", "Charge"),
            ("australian", "Australia"),
        ],
    )
    def test_derivational_forms_are_illegal(self, clue, word):
        assert not is_legal_clue(clue, [word])

    @pytest.mark.parametrize(
        "clue,word",
        [
            ("centre", "Center"),
            ("colour", "Color"),
            ("theatre", "Theater"),
        ],
    )
    def test_british_spellings_are_illegal(self, clue, word):
        assert not is_legal_clue(clue, [word])

    @pytest.mark.parametrize(
        "clue,word",
        [
            ("coding", "Code"),   # silent -e dropped before a vowel suffix
            ("batter", "Bat"),    # consonant doubled before one
            ("happier", "Happy"),  # y -> i
            ("cities", "City"),
        ],
    )
    def test_stem_spelling_changes_are_illegal(self, clue, word):
        assert not is_legal_clue(clue, [word])

    @pytest.mark.parametrize(
        "clue,word",
        [
            # Coincidental letter overlap. A fuzzy-similarity rule forbade
            # every one of these, which is why the rule is prefix-based.
            ("able", "Marble"),
            ("after", "Water"),
            ("agree", "Green"),
            ("am", "Arm"),
            ("bar", "Bear"),
            ("bat", "Beat"),
            # Real clues the baseline picks; none may become illegal.
            ("artillery", "Mount"),
            ("gospel", "Cricket"),
            ("aviation", "Pilot"),
            ("cartoon", "Paper"),
        ],
    )
    def test_unrelated_words_stay_legal(self, clue, word):
        assert is_legal_clue(clue, [word])

    def test_short_stems_do_not_leak(self):
        """Stripping "bring" must not yield "br" and match everything
        starting with those letters."""
        assert is_legal_clue("bring", ["Brick"])
        assert is_legal_clue("bring", ["Bread"])

    def test_prefix_rule_needs_five_characters(self):
        """Four-character agreement is not enough -- at four the rule puts
        5.2% of the clue pool out of reach while catching nothing extra."""
        assert is_legal_clue("agenda", ["Agent"])       # "agen"
        assert is_legal_clue("amazing", ["Amazon"])     # "amaz"
        assert not is_legal_clue("charging", ["Charge"])  # "charg"

    def test_known_gap_irregular_forms(self):
        """Irregular morphology is out of scope: no dependency-free rule
        sees "led" as a form of "lead". Documented so the gap is a choice
        rather than a surprise."""
        assert is_legal_clue("led", ["Lead"])

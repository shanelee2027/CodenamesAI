"""Board state, role assignment, and clue legality (SCOPE.md §M1).

Design notes for the two non-obvious calls made here:

Role partition is fixed at own=9, opponent=8, neutral=7, assassin=1 (25
total) -- the standard Codenames starting-team split, matching the feature
vector layout in SCOPE.md §2. Every `Card`'s role is fixed at board
generation from one team's perspective -- codemasters, guessers, and the
scorer are all written against that single perspective, never a
parameter. Real two-team play (codenames/game.py::play_two_team_game) is
still possible without changing any of them: `OpponentBoardView` below
just swaps OWN/OPPONENT while sharing the same underlying revealed-state,
so handing the second team's codemaster/guesser that view instead of the
real `Board` is enough.

Legality's "morphological variants" rule is deliberately narrow. Regular
English suffixation (plural -s/-es, -ing, -ed) is *already* a substring
relationship -- "apple" is a substring of "apples", "run" is a substring of
"running" -- so the substring check below catches those for free. The only
common case substring misses is a stem-spelling change: y -> i before a
suffix ("happy"/"happier", "city"/"cities"). _stem_variants() adds exactly
that one variant per word. This will NOT catch irregular forms (mouse/mice,
go/went) -- a full lemmatizer would, at the cost of a new dependency and
much less predictable/testable behavior for a rule where false negatives
(illegal clues let through) silently corrupt every downstream score. Given
SCOPE's own note that "bugs here silently inflate every downstream score,"
predictable and testable beats broad coverage.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

ASSET_WORDLIST_PATH = Path(__file__).parent / "assets" / "board_words.txt"
ASSET_HOLDOUT_WORDLIST_PATH = Path(__file__).parent / "assets" / "board_words_holdout.txt"


class Role(Enum):
    OWN = "own"
    OPPONENT = "opponent"
    NEUTRAL = "neutral"
    ASSASSIN = "assassin"


ROLE_COUNTS: dict[Role, int] = {
    Role.OWN: 9,
    Role.OPPONENT: 8,
    Role.NEUTRAL: 7,
    Role.ASSASSIN: 1,
}
BOARD_SIZE = sum(ROLE_COUNTS.values())

# The learned scorer (M8) outputs a distribution over k in 0..4 -- "the
# number of own-words the guesser will reveal before stopping" (SCOPE §2).
# Baseline codemasters (M6) cap their chosen number at the same bound so
# every codemaster's outputs stay comparable in the arena. Lives here (not
# in codemasters/base.py, where it conceptually belongs) so both
# codemasters/ and scorer.py can import it without a circular dependency --
# codemasters/learned.py already depends on scorer.py, so scorer.py can't
# depend back on anything under codemasters/.
MAX_CLUE_NUMBER = 4


@dataclass(frozen=True)
class Card:
    word: str
    role: Role


@dataclass
class Board:
    cards: tuple[Card, ...]
    seed: int
    revealed: set[str] = field(default_factory=set)

    @classmethod
    def generate(cls, seed: int, vocabulary: list[str] | None = None) -> "Board":
        vocab = vocabulary if vocabulary is not None else load_wordlist()
        if len(vocab) < BOARD_SIZE:
            raise ValueError(f"vocabulary has {len(vocab)} words, need at least {BOARD_SIZE}")

        rng = random.Random(seed)
        words = rng.sample(vocab, k=BOARD_SIZE)

        roles: list[Role] = []
        for role, count in ROLE_COUNTS.items():
            roles.extend([role] * count)
        rng.shuffle(roles)

        cards = tuple(Card(word=w, role=r) for w, r in zip(words, roles))
        return cls(cards=cards, seed=seed)

    def __post_init__(self) -> None:
        if len(self.cards) != BOARD_SIZE:
            raise ValueError(f"board must have exactly {BOARD_SIZE} cards, got {len(self.cards)}")

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(c.word for c in self.cards)

    def _card(self, word: str) -> Card:
        target = word.lower()
        for c in self.cards:
            if c.word.lower() == target:
                return c
        raise KeyError(f"{word!r} is not on this board")

    def role_of(self, word: str) -> Role:
        return self._card(word).role

    def is_revealed(self, word: str) -> bool:
        card = self._card(word)
        return card.word in self.revealed

    def reveal(self, word: str) -> Role:
        card = self._card(word)
        # store the card's canonical casing, not the caller's -- otherwise
        # reveal("king") and is_revealed("King") would disagree.
        self.revealed.add(card.word)
        return card.role

    def words_by_role(self, role: Role, *, unrevealed_only: bool = False) -> list[str]:
        return [
            c.word for c in self.cards
            if c.role == role and (not unrevealed_only or c.word not in self.revealed)
        ]

    def remaining(self, role: Role) -> int:
        return len(self.words_by_role(role, unrevealed_only=True))


def _swap_own_opponent(role: Role) -> Role:
    return {Role.OWN: Role.OPPONENT, Role.OPPONENT: Role.OWN}.get(role, role)


class OpponentBoardView:
    """The same physical Board, seen from the other team's perspective:
    OWN and OPPONENT swap (their 8 or 9 words are what the board's own
    Cards call OPPONENT, and vice versa) -- NEUTRAL and ASSASSIN are
    shared, same as in real Codenames. `words`/`is_revealed`/`reveal`
    delegate straight through to the *same* underlying Board (one
    physical revealed-state, not a copy), so a word either team reveals
    is immediately gone for both -- only the role labels differ.

    This is what makes two-team play (codenames/game.py::play_two_team_game)
    possible without touching Board, Codemaster, Guesser, or the scorer at
    all: every one of them only ever queries a board through role_of/
    words_by_role/remaining/is_revealed/reveal/words, so handing the
    second team's codemaster and guesser this view instead of the real
    Board is enough for them to correctly see "their own" 8 or 9 words as
    Role.OWN, with no code anywhere needing to know two teams exist."""

    def __init__(self, board: Board):
        self._board = board

    @property
    def words(self) -> tuple[str, ...]:
        return self._board.words

    @property
    def seed(self) -> int:
        return self._board.seed

    @property
    def revealed(self) -> set[str]:
        # Which words are revealed doesn't depend on perspective, only
        # what role they turn out to be -- some codemaster code reads
        # this set directly (codenames/codemasters/_util.py::state_rng,
        # LearnedCodemaster's turn-index calc) rather than going through
        # is_revealed()/reveal(), so it needs to exist here too.
        return self._board.revealed

    def role_of(self, word: str) -> Role:
        return _swap_own_opponent(self._board.role_of(word))

    def is_revealed(self, word: str) -> bool:
        return self._board.is_revealed(word)

    def reveal(self, word: str) -> Role:
        return _swap_own_opponent(self._board.reveal(word))

    def words_by_role(self, role: Role, *, unrevealed_only: bool = False) -> list[str]:
        return self._board.words_by_role(_swap_own_opponent(role), unrevealed_only=unrevealed_only)

    def remaining(self, role: Role) -> int:
        return self._board.remaining(_swap_own_opponent(role))


def load_wordlist(path: Path = ASSET_WORDLIST_PATH) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_holdout_wordlist(path: Path = ASSET_HOLDOUT_WORDLIST_PATH) -> list[str]:
    """The board words training data must never be sampled from (first-pass
    generalization check in place of held-out guessers -- see docs/log.md).
    60 of the 400 board words, chosen via
    `random.Random(42).sample(load_wordlist(), 60)` -- fixed and committed
    (codenames/assets/board_words_holdout.txt) rather than resampled at
    runtime, so the split is transparent and inspectable."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_training_wordlist(all_path: Path = ASSET_WORDLIST_PATH, holdout_path: Path = ASSET_HOLDOUT_WORDLIST_PATH) -> list[str]:
    """load_wordlist() minus load_holdout_wordlist() -- what training data
    generation (scripts/generate_training_data.py) samples boards from."""
    holdout = set(load_holdout_wordlist(holdout_path))
    return [w for w in load_wordlist(all_path) if w not in holdout]


def _normalize(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _stem_variants(word: str) -> set[str]:
    norm = _normalize(word)
    variants = {norm}
    if norm.endswith("y") and len(norm) > 1:
        variants.add(norm[:-1] + "i")
    return variants


def is_legal_clue(clue: str, board_words: Iterable[str]) -> bool:
    """A clue is illegal if it or any board word contains the other as a
    substring, checked over both words' y->i stem variants (see module
    docstring). This covers exact matches, regular plurals/verb forms in
    either direction, hyphenated/multi-word board entries (normalization
    strips non-alphanumerics), and case differences.
    """
    clue_variants = _stem_variants(clue)
    for board_word in board_words:
        board_variants = _stem_variants(board_word)
        for c in clue_variants:
            if not c:
                continue
            for b in board_variants:
                if not b:
                    continue
                if c in b or b in c:
                    return False
    return True

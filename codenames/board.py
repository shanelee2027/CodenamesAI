"""Board state, role assignment, and clue legality.

Design notes for the two non-obvious calls made here:

Role partition is fixed at own=9, opponent=8, neutral=7, assassin=1 (25
total) -- the standard Codenames starting-team split, matching the feature
vector layout in codenames/features.py. Every `Card`'s role is fixed at board
generation from one team's perspective -- spymasters, guessers, and the
scorer are all written against that single perspective, never a
parameter. Real two-team play (codenames/game.py::play_two_team_game) is
still possible without changing any of them: `OpponentBoardView` below
just swaps OWN/OPPONENT while sharing the same underlying revealed-state,
so handing the second team's spymaster/guesser that view instead of the
real `Board` is enough.

Legality has two rules, both deliberately narrow.

**Substring over stem variants.** Regular English suffixation (plural
-s/-es, -ing, -ed) is *already* a substring relationship -- "apple" is a
substring of "apples", "run" is a substring of "running" -- so a plain
substring check catches those for free. _stem_variants() adds the cases
substring alone misses: y -> i before a suffix ("happy"/"happier"),
British/American spelling ("centre"/"center", "colour"/"color"), and
suffix stripping where the stem changes shape ("coding"/"code",
"batter"/"bat").

**Shared prefix.** Substring cannot see derivational morphology, because
neither word contains the other: "Mexico"/"mexican", "Canada"/"canadian",
"Change"/"changing". Measured on 60 boards, 10% of the baseline's chosen
clues were of exactly this kind, every one illegal at a real table and
every one scoring z > +8 precisely *because* it was the same word. A
five-character shared prefix catches them. Five was chosen by measurement
(see docs/log.md): it puts 1.0% of the admissible clue pool out of reach,
against 5.2% at four characters, while four caught nothing extra. Both
rules together rule out a mean 2.4% of the pool per board (min 1.6%, max
3.4%), and cut near-identical clues from 6 of 60 boards to 1.

A fuzzy string-similarity threshold was tried and rejected. At the level
needed to catch "led"/"lead" it also forbade "able"/Marble,
"after"/Water, "agree"/Green and "am"/Arm -- coincidental letter overlap,
not shared roots -- taking 7.0% of the pool with it.

Neither rule catches irregular forms (mouse/mice, go/went, lead/led). A
full lemmatizer would, at the cost of a new dependency and much less
predictable behavior for a rule where false negatives silently inflate
every downstream score. Predictable and testable beats broad coverage.
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

# The learned scorer outputs a distribution over k in 0..4 -- the
# number of own-words the guesser will reveal before stopping.
# Baseline spymasters cap their chosen number at the same bound so
# every spymaster's outputs stay comparable in the arena. Lives here (not
# in spymasters/base.py, where it conceptually belongs) so both
# spymasters/ and scorer.py can import it without a circular dependency --
# spymasters/learned.py already depends on scorer.py, so scorer.py can't
# depend back on anything under spymasters/.
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
    possible without touching Board, Spymaster, Guesser, or the scorer at
    all: every one of them only ever queries a board through role_of/
    words_by_role/remaining/is_revealed/reveal/words, so handing the
    second team's spymaster and guesser this view instead of the real
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
        # what role they turn out to be -- some spymaster code reads
        # this set directly (codenames/spymasters/_util.py::state_rng,
        # LearnedSpymaster's turn-index calc) rather than going through
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
    150 of the 400 board words (raised from 60 -- see
    docs/iteration-architecture.md step 5 -- so two "independent" eval
    boards share fewer words: ~4.2 of 25 at H=150 versus ~10.4 at H=60).
    The original 60 (`random.Random(42).sample(load_wordlist(), 60)`) are
    kept as an exact subset, never reshuffled, so a model trained under
    the old 60-word split is still evaluated against a superset of what
    it was told to avoid (though see that doc's contamination warning:
    such a model still trained on 90 words that are now held out). The
    additional 90 were drawn uniformly (no stratification) from the
    remaining 340 words via `random.Random(43).sample(remaining, 90)`,
    where `remaining` is `load_wordlist()` filtered to exclude the
    original 60, in file order -- see docs/log.md for the exact
    reproducing snippet. Fixed and committed
    (codenames/assets/board_words_holdout.txt), sorted alphabetically for
    a stable diff, rather than resampled at runtime, so the split is
    transparent and inspectable."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_training_wordlist(all_path: Path = ASSET_WORDLIST_PATH, holdout_path: Path = ASSET_HOLDOUT_WORDLIST_PATH) -> list[str]:
    """load_wordlist() minus load_holdout_wordlist() -- what training data
    generation (scripts/pipeline/generate_training_data.py) samples boards from."""
    holdout = set(load_holdout_wordlist(holdout_path))
    return [w for w in load_wordlist(all_path) if w not in holdout]


def _normalize(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


# Suffixes stripped when deriving stem variants. Order matters only in
# that longer suffixes are tried first, so "changes" yields "chang"/"change"
# rather than stopping at "change" -- both are kept anyway.
_SUFFIXES = ("ing", "ers", "est", "ies", "ed", "er", "es", "ly", "s")

# A stem shorter than this is dropped rather than compared. Stripping
# leaves fragments ("bring" -> "br") that would match unrelated words as
# substrings, and a rule that silently forbids good clues is worse than
# one that misses a rare bad one. Three keeps the stems that carry a real
# word -- "batter" -> "bat", "coding" -> "cod" -> "code" -- while dropping
# two-letter debris.
_MIN_STEM = 3

# Shared-prefix length that makes two words the same root. Five puts 1.0%
# of the admissible clue pool out of reach; four takes 5.2% while catching
# nothing extra. See the module docstring.
_MIN_SHARED_PREFIX = 5


def _spelling_normalized(norm: str) -> str:
    """Fold British spellings onto their American counterparts, so
    "centre"/"Center" and "colour"/"Color" compare equal. Applied to both
    sides, so the direction of the fold doesn't matter."""
    if norm.endswith("re") and len(norm) > 3:
        norm = norm[:-2] + "er"
    norm = norm.replace("our", "or")
    if norm.endswith("ise") or norm.endswith("isation"):
        norm = norm.replace("isation", "ization").replace("ise", "ize")
    return norm


def _stem_variants(word: str) -> set[str]:
    """Surface forms, compared by substring. These are whole words, so
    containment between them means something: "run" inside "running"."""
    norm = _normalize(word)
    variants = {norm, _spelling_normalized(norm)}
    if norm.endswith("y") and len(norm) > 1:
        variants.add(norm[:-1] + "i")
    return {v for v in variants if v}


def _stems(word: str) -> set[str]:
    """Surface forms plus suffix-stripped stems, compared only by equality
    and shared prefix -- never by substring.

    A stripped stem is a fragment, and fragments land inside unrelated
    words by coincidence: "amazing" strips to "amaz", which sits inside
    "amazon" while sharing no root with it. Equality still catches the
    cases stripping exists for, since both sides get stripped: "coding"
    and "Code" both reach "code"."""
    variants = _stem_variants(word)
    stems = set(variants)
    for base in variants:
        for suffix in _SUFFIXES:
            if not base.endswith(suffix) or len(base) - len(suffix) < _MIN_STEM:
                continue
            stem = base[: -len(suffix)]
            # "coding" -> "cod" -> "code": the silent -e dropped before a
            # vowel-initial suffix. "batter" -> "batt" -> "bat": the
            # consonant doubled before one.
            found = {stem, stem + "e"}
            if len(stem) > 1 and stem[-1] == stem[-2]:
                found.add(stem[:-1])
            # Spelling has to be folded *after* stripping as well as
            # before: "centres" only reaches "centre" once the plural is
            # gone, and only then can it fold to "center".
            stems |= found | {_spelling_normalized(s) for s in found}
    return {s for s in stems if s}


def _shares_root(clue_variants: set[str], board_variants: set[str]) -> bool:
    """True when any variant pair agrees on its first `_MIN_SHARED_PREFIX`
    characters -- the derivational cases substring cannot see, since
    neither "mexico" nor "mexican" contains the other."""
    for c in clue_variants:
        if len(c) < _MIN_SHARED_PREFIX:
            continue
        head = c[:_MIN_SHARED_PREFIX]
        for b in board_variants:
            if len(b) >= _MIN_SHARED_PREFIX and b.startswith(head):
                return True
    return False


def is_legal_clue(clue: str, board_words: Iterable[str]) -> bool:
    """A clue is illegal if it or any board word contains the other as a
    substring, or if the two share a five-character prefix, both checked
    over the words' stem variants (see the module docstring).

    Substring covers exact matches, regular plurals/verb forms in either
    direction, hyphenated/multi-word board entries (normalization strips
    non-alphanumerics), and case differences. The prefix rule covers
    derivational forms, where neither word contains the other.
    """
    clue_variants = _stem_variants(clue)
    clue_stems = _stems(clue)
    for board_word in board_words:
        for b in _stem_variants(board_word):
            for c in clue_variants:
                if c in b or b in c:
                    return False
        board_stems = _stems(board_word)
        if clue_stems & board_stems:
            return False
        if _shares_root(clue_stems, board_stems):
            return False
    return True

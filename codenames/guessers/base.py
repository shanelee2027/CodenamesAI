"""Guesser pool base interface (SCOPE.md §M5/§3).

A guesser only ever sees the clue and the *unrevealed* words currently on
the table -- never roles. It's the mechanism that turns "is this a good
clue" into something simulable: a guesser scores/ranks candidate words,
and whichever ones it would pick determines the outcome.

Two methods, not one, because a single "return the sorted word list"
interface can't support the pool SCOPE.md §3 actually asks for:
NoisyGuesser needs the underlying numeric scores to perturb, and
ConfidenceThresholdGuesser needs to voluntarily return *fewer* than all
candidates. `score_candidates()` is what makes each guesser type
different; `rank_candidates()` has a sensible default (sort by score) and
is only overridden by guessers that truncate their own ranking.

Diversity in this pool must come from differences in what each guesser
knows or how it decides, not from noise sprinkled on an otherwise-
identical base (SCOPE.md §3's own explicit warning) -- exactly one
guesser type here uses noise (NoisyGuesser), and it wraps a genuinely
different base rather than being the pool's only source of diversity.

**The backlog/bonus-guess mechanism** (added for `HistoryAwareGuesser`,
see codenames/guessers/history_aware.py). Standard Codenames always
grants a guessing team `n+1` attempts for an announced number `n` -- this
project deliberately dropped that blanket bonus (docs/log.md: "the +1 is
a bonus a team *chooses* to spend when it still feels confident, and this
project's guessers have no notion of 'still feels confident' to make that
judgment call with"). `history`/`bonus_guesses` reinstates it, but only
as an option a guesser can *earn a reason to use* -- a tracked backlog of
words a past clue plausibly still owes is exactly the "still feels
confident" signal that was missing. Every existing guesser here keeps
declining it (bonus_guesses' default is 0), so nothing about their
behavior changes; only a guesser that overrides `bonus_guesses` can ever
make a turn longer than `n`.

`history` is a list of `(old_clue, owed_count)` pairs: how many own-words
a past clue is still believed to have left unaccounted-for. There's no
stored cursor pointing at a specific word -- re-ranking `old_clue`
against whatever candidates are still unrevealed always reproduces its
best remaining guess, since anything already resolved (guessed correctly,
guessed wrongly, or claimed by a different clue) has already dropped out
of the candidate pool by then. `owed_count` only ever decreases when an
own-word is actually attributed to that clue (see `update_history`); nothing
needs to track *which* word specifically, so this stays a plain,
stateless data structure threaded through `codenames/game.py`'s functions
rather than mutable per-instance state.

That "re-ranking always reproduces the same answer" guarantee depends on
every guesser's `score_candidates` being a pure function of (clue,
candidates) -- true of everything in this pool except, until it was
fixed, `NoisyGuesser` (codenames/guessers/noisy.py), whose noise used to
come from a continuously-advancing RNG stream rather than a fixed
function of (clue, word). Re-scoring the same clue twice -- once during
real play, once retrospectively in `update_history`'s collision check --
could silently disagree, letting an already-satisfied backlog entry look
still-owed. See that module's docstring and docs/log.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from codenames.board import Role
from codenames.similarity import SimilarityTensor

if TYPE_CHECKING:
    # Deferred to avoid a cycle: codenames/game.py imports Guesser from
    # this module. `from __future__ import annotations` already makes
    # every annotation here a lazy string, so this is only ever needed by
    # type checkers.
    from codenames.game import TurnResult


class Guesser(ABC):
    @abstractmethod
    def score_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> dict[str, float]:
        """Higher score = more likely to guess. Not required to be
        bounded or a probability. A candidate this guesser's knowledge
        source has no vector for scores -inf, not 0 -- 0 would
        misleadingly compete with a real low-but-nonzero similarity."""

    def rank_candidates(
        self,
        clue: str,
        candidate_words: list[str],
        sims: SimilarityTensor,
        number: int | None = None,
        history: list[tuple[str, int]] | None = None,
    ) -> list[str]:
        """Candidates in the order this guesser would try them, most
        likely first. May return fewer than all candidates if this
        guesser would voluntarily stop early (see
        ConfidenceThresholdGuesser) -- the caller combines this with the
        number attempt cap and turn-ending-on-a-miss rule (both handled
        by the game loop in M6, not here) to determine what's actually
        played. `number`/`history` are only meaningful to a guesser that
        overrides them (see `HistoryAwareGuesser`) -- ignored here."""
        scores = self.score_candidates(clue, candidate_words, sims)
        return sorted(candidate_words, key=lambda w: -scores[w])

    def bonus_guesses(
        self,
        clue: str,
        candidate_words: list[str],
        sims: SimilarityTensor,
        number: int,
        history: list[tuple[str, int]] | None = None,
    ) -> int:
        """How many guesses beyond `number` this guesser wants to spend
        this turn (see this module's docstring). Always 0 here -- a
        guesser has to have an actual reason to claim it."""
        return 0

    def update_history(
        self,
        history: list[tuple[str, int]] | None,
        clue: str,
        number: int,
        turn: "TurnResult",
        candidates_before_turn: list[str],
        sims: SimilarityTensor,
    ) -> list[tuple[str, int]]:
        """Generic backlog bookkeeping after one turn, usable by any
        guesser (not just a history-aware one) so codenames/game.py can
        maintain `history` uniformly regardless of which guesser is
        plugged in -- for a guesser that never reads `history`, this just
        computes a value nobody looks at.

        An owed count only ever decreases when an own-word is actually
        attributed to that clue:
        - this turn's own guesses might include the word this old clue
          would itself have ranked highest, in which case we assume that
          one word was meant to satisfy both clues at once (counting it
          fully toward *both* rather than splitting it -- treating it as
          still separately owed would overcount how many words are
          actually left, which risks a later guess spent chasing a word
          that never existed); or
        - (for a guesser that actually spends its bonus, see
          HistoryAwareGuesser.rank_candidates) a direct correct guess
          against that backlog.

        A wrong guess never adjusts any owed count -- the word is simply
        gone from every future `candidates_before_turn`, which already
        removes it from consideration with no bookkeeping needed. A new
        entry is added for this turn's own clue only if it ended on a
        genuine miss (`neutral`/`opponent` -- `assassin` ends the game,
        and a clean finish has nothing left owed)."""
        new_history: list[tuple[str, int]] = []
        correct_words = {w for w, role in turn.guesses if role == Role.OWN}

        for old_clue, owed in history or []:
            old_top = self.rank_candidates(old_clue, candidates_before_turn, sims)
            if old_top and old_top[0] in correct_words:
                owed -= 1
            if owed > 0:
                new_history.append((old_clue, owed))

        if turn.ended_reason in ("neutral", "opponent"):
            owed = number - len(correct_words)
            if owed > 0:
                new_history.append((clue, owed))

        return new_history

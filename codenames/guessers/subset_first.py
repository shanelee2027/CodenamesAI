"""A guesser that decides which words a clue points at before ranking them.

**Why.** The arena's gpt-oss guesser ranks every word on the board and the
turn takes the top N, so it never declines a word: the 3rd guess for a clue
that only points at 2 words is whatever ranks 3rd. A human first sees the
clue as pointing at a few words or not, orders those, and stops when none
are left (docs/log.md, "Comfort"). The one-call STOP guesser
(OpenAICompatGuesser(allow_stop=True)) put that decision inside the ranking
and was a weak version of it: it stopped on 12-14% of turns and refused 11
of 100 boards (docs/log.md, "Does a guesser that may stop rescue the pass?").

**Two calls, both to the same model.**
1. Which words does the clue point at? The relatedness prompt
   (codenames/relatedness.py, graded-v2): the clue *without* its number,
   each word sorted into "guess" (clearly points at it), "stretch" (would
   still try it as a 2nd or 3rd word) or neither. Without the number, so
   the set is what the clue says, not what the spymaster asked for.
2. In what order? The arena's usual ranking prompt, with the number, over
   only the words in the set. Skipped when the set has fewer than two words.

The turn guesses the ranked set, at most N words, and then STOP: a set
smaller than N ends the turn early, an empty set is a pass. Which set
counts is `subset`: "stretch" (guess + stretch, the default -- stretch is
defined as exactly the words you would still take when the number asks for
more) or "guess" (only the clear ones).

Both answers are cached in cache/llm_store.db under the plain guesser's
identity -- the relatedness table and the ranking table -- so labels already
bought for the stop listener are reused, and a ranking over a subset can
never be mistaken for a full-board one (its candidate list differs). This
guesser's own identity adds `+subset=<which>`, so a suite that uses it gets
its own suite_id.
"""

from __future__ import annotations

from pathlib import Path

from codenames.guessers.base import STOP, Guesser
from codenames.guessers.openai_compat import OpenAICompatGuesser
from codenames.relatedness import PROMPT_VERSION, RelatednessAsker, RelatednessStore
from codenames.similarity import SimilarityTensor

SUBSETS = ("stretch", "guess")


class SubsetFirstGuesser(Guesser):
    def __init__(self, inner: OpenAICompatGuesser, db_path: str | Path, subset: str = "stretch",
                 version: str = PROMPT_VERSION):
        if subset not in SUBSETS:
            raise ValueError(f"subset must be one of {SUBSETS}, not {subset!r}")
        if inner.allow_stop:
            raise ValueError("the inner guesser ranks a set that is already decided; it must not offer STOP")
        self.inner, self.subset, self.version = inner, subset, version
        self.asker = RelatednessAsker(inner, RelatednessStore(Path(db_path)), version)

    @property
    def cache_model_id(self) -> str:
        return f"{self.inner.cache_model_id}+subset={self.subset}"

    def related(self, clue: str, candidate_words: list[str]) -> list[str]:
        """Step 1: the words the clue points at, in board order."""
        guess, stretch = self.asker.ask(clue, list(candidate_words))
        chosen = set(guess) | (set(stretch) if self.subset == "stretch" else set())
        return [w for w in candidate_words if w in chosen]

    def rank_candidates(
        self,
        clue: str,
        candidate_words: list[str],
        sims: SimilarityTensor,
        number: int | None = None,
    ) -> list[str]:
        words = self.related(clue, candidate_words)
        if len(words) >= 2:
            words = self.inner._ranked(clue, words, number)
        return words + [STOP]

    def score_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> dict[str, float]:
        """The set in ranked order above every other word, which score -inf:
        this guesser would not pick them at all."""
        ranking = self.rank_candidates(clue, candidate_words, sims)[:-1]
        n = len(ranking)
        scores = {w: float("-inf") for w in candidate_words}
        scores.update({w: float(n - i) for i, w in enumerate(ranking)})
        return scores

    def __repr__(self) -> str:
        return f"SubsetFirstGuesser({self.inner!r}, subset={self.subset!r})"

"""Wraps a base guesser and lets it spend the backlog/bonus-guess
mechanism described in codenames/guessers/base.py's module docstring --
the guesser this project's cross-turn-clue-memory open question
(docs/versions/v1.md) pointed at as the cheap first thing to try.

**The comparability problem, checked empirically before building this**
(see docs/log.md): comparing two different clues' raw similarity scores
to decide which is the more confident guess doesn't work, because a
clue's *typical* similarity to the board vocabulary varies enormously by
clue word and correlates strongly with how common that word is (r=0.84
for GloVe specifically -- a well-known "hubness" effect in cosine-
similarity embedding spaces: frequent/central words read as vaguely
similar to almost everything, rare/technical ones read as dissimilar to
almost everything, regardless of actual topical relevance). Left
unnormalized, a backlog word from a generic-sounding old clue would
systematically outrank a genuinely strong candidate under a rarer new
clue, purely because of which embedding neighborhood it happens to sit
in. Fixed by z-scoring every clue's scores against that same clue's own
similarity distribution over the full board vocabulary before any
cross-clue comparison -- putting every clue on a "how unusual is this
*for this clue*" scale instead of an absolute one.
"""

from __future__ import annotations

import math
import statistics

from codenames.guessers.base import Guesser
from codenames.similarity import SimilarityTensor


class HistoryAwareGuesser(Guesser):
    def __init__(self, base: Guesser):
        self.base = base
        # Memoizes a pure function of (clue, the fixed board vocabulary),
        # not any particular game's state, so caching this across many
        # games sharing one instance is safe: the same clue always
        # produces the same baseline no matter which game or how many
        # other clues were scored first -- this relies on self.base's own
        # score_candidates being a pure function of (clue, candidates)
        # too, which wasn't actually true of NoisyGuesser until that was
        # fixed (see codenames/guessers/noisy.py).
        self._baseline_cache: dict[str, tuple[float, float]] = {}

    def score_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> dict[str, float]:
        return self.base.score_candidates(clue, candidate_words, sims)

    def _baseline(self, clue: str, sims: SimilarityTensor) -> tuple[float, float]:
        """(mean, std) of this guesser's own score for `clue` across the
        *full* board vocabulary (not just this turn's ~25 candidates,
        for a more stable estimate) -- the z-score reference distribution
        for this specific clue."""
        if clue not in self._baseline_cache:
            scores = self.base.score_candidates(clue, sims.board_words, sims)
            finite = [s for s in scores.values() if math.isfinite(s)]
            mean = statistics.fmean(finite) if finite else 0.0
            std = statistics.pstdev(finite) if len(finite) > 1 else 0.0
            self._baseline_cache[clue] = (mean, std)
        return self._baseline_cache[clue]

    def _zscored(self, clue: str, scores: dict[str, float], sims: SimilarityTensor) -> dict[str, float]:
        mean, std = self._baseline(clue, sims)
        if std == 0.0:
            return dict.fromkeys(scores, 0.0)
        return {w: (s - mean) / std if math.isfinite(s) else float("-inf") for w, s in scores.items()}

    def _best_backlog_candidate(
        self, candidate_words: list[str], sims: SimilarityTensor, history: list[tuple[str, int]] | None
    ) -> tuple[str, float] | None:
        """Across every pending backlog clue, the single word/z-score
        pair most worth spending the one available bonus guess on --
        just the top-ranked remaining candidate for each old clue (no
        stored cursor, see base.py's docstring), compared on the
        z-scored scale."""
        best: tuple[str, float] | None = None
        for old_clue, _owed in history or []:
            old_scores = self.base.score_candidates(old_clue, candidate_words, sims)
            if not old_scores:
                continue
            top_word = max(old_scores, key=old_scores.get)
            if not math.isfinite(old_scores[top_word]):
                continue
            z = self._zscored(old_clue, {top_word: old_scores[top_word]}, sims)[top_word]
            if best is None or z > best[1]:
                best = (top_word, z)
        return best

    def _merge(
        self, clue: str, candidate_words: list[str], sims: SimilarityTensor, history: list[tuple[str, int]] | None
    ) -> tuple[list[str], tuple[str, int] | None]:
        """Merged ranking (current clue's own ranking, with the single
        best backlog candidate spliced in at the position its z-score
        earns), plus (backlog_word, its 0-indexed position in that
        ranking) if a backlog candidate was found at all -- callers use
        the position to decide whether it's actually within reach of the
        one available bonus guess."""
        base_scores = self.base.score_candidates(clue, candidate_words, sims)
        base_ranked = sorted(candidate_words, key=lambda w: -base_scores[w])

        backlog = self._best_backlog_candidate(candidate_words, sims, history)
        if backlog is None:
            return base_ranked, None
        backlog_word, backlog_z = backlog

        z_current = self._zscored(clue, base_scores, sims)
        merged = [w for w in base_ranked if w != backlog_word]
        insert_at = next((i for i, w in enumerate(merged) if z_current[w] < backlog_z), len(merged))
        merged.insert(insert_at, backlog_word)
        return merged, (backlog_word, insert_at)

    def rank_candidates(
        self,
        clue: str,
        candidate_words: list[str],
        sims: SimilarityTensor,
        number: int | None = None,
        history: list[tuple[str, int]] | None = None,
    ) -> list[str]:
        # Called with no history (e.g. from Guesser.update_history's
        # generic collision check) this is just the base guesser's own
        # ranking -- `_merge` degrades to that automatically when there's
        # no backlog to consider.
        merged, _ = self._merge(clue, candidate_words, sims, history)
        return merged

    def bonus_guesses(
        self,
        clue: str,
        candidate_words: list[str],
        sims: SimilarityTensor,
        number: int,
        history: list[tuple[str, int]] | None = None,
    ) -> int:
        _merged, backlog_info = self._merge(clue, candidate_words, sims, history)
        if backlog_info is None:
            return 0
        _backlog_word, insert_at = backlog_info
        # Only worth claiming the bonus if the backlog word actually made
        # it into the first `number + 1` merged positions -- otherwise
        # every one of the current clue's own candidates outranked it,
        # and claiming a bonus would just spend it on an ordinary
        # current-clue candidate instead of the confident backlog pick
        # the bonus exists for.
        return 1 if insert_at <= number else 0

    def __repr__(self) -> str:
        return f"HistoryAwareGuesser(base={self.base!r})"

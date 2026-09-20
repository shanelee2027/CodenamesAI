"""Clue search against the distilled listener, scored by `pl_reward`.

Two models are in play and they do different jobs. The distilled listener
(`cache/listener_gbt.txt`, features in `codenames/listener_features.py`) says
how a guesser ranks board words for a clue. `codenames/pl_reward.py` turns that
ranking model into the expected reward of a turn. This module is the search
that puts them together.

**Two-stage, because feature extraction is the bottleneck.** `extract()` costs
~750 us per clue, so scoring all 11,145 clues in the pool would be ~8.4 s per
turn. The Gaussian model in `expected_words.py` scores the whole pool in one
vectorised pass, and is good enough to say which clues are worth a second look
even where it is wrong about their order. So: shortlist with the Gaussian,
rescore the shortlist with the listener, and pay the extraction cost only on
the shortlist. At the default 200 that is ~0.15 s per turn.

The obvious risk is that the shortlist is itself chosen by the model we are
trying to replace: a clue the Gaussian ranks 500th can never be recovered, no
matter how good the listener is. `SHORTLIST` is therefore a parameter, and
`scripts/tools/compare_shortlist_depth.py` measures how often the final pick
changes between depths -- if it rarely changes, the depth is enough.

**The listener never sees roles.** Features are extracted for every unrevealed
word at once, exactly as in training, and the scores are split into ours and
theirs only afterwards, inside the reward calculation. A role-aware listener
would look excellent in the arena and be worthless as a model of a guesser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codenames.board import Board, OpponentBoardView, Role
from codenames.clue_search import is_legal_clue
from codenames.clue_stats import ClueStats
from codenames.game import role_costs
from codenames.listener_features import (
    EntitySims,
    ExtraSims,
    SwowTables,
    WordNorms,
    WordStats,
    extract,
)
from codenames.pl_reward import gain_and_penalty
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
from codenames.spymasters.base import MAX_CLUE_NUMBER, Spymaster, TurnContext
from codenames.spymasters.expected_words import ExpectedWordsSpymaster

SHORTLIST = 200


@dataclass(frozen=True)
class ListenerBundle:
    """Every artifact `extract` needs, loaded once.

    Held together rather than passed as eight arguments because the arena
    constructs a fresh spymaster inside each worker process, and a partial load
    (one table missing, its features silently NaN) is the kind of thing that
    degrades a model quietly instead of failing.
    """

    booster: object
    word_stats: WordStats
    swow: SwowTables | None
    entity: EntitySims | None
    pmi: EntitySims | None
    extra: ExtraSims | None
    norms: WordNorms | None
    wordnet: ExtraSims | None
    lexical: ExtraSims | None

    @classmethod
    def load(cls, cache_dir: Path, model_path: Path) -> "ListenerBundle":
        import lightgbm as lgb

        def maybe(loader, name):
            p = cache_dir / name
            return loader(p) if p.exists() else None

        missing = [n for n in ("swow.npz", "lm_pmi.npz", "extra_sims.npz", "word_norms.npz",
                               "wordnet_sims.npz", "lexical_sims.npz", "entity_sims.npz")
                   if not (cache_dir / n).exists()]
        if missing:
            raise FileNotFoundError(
                f"listener feature tables missing from {cache_dir}: {', '.join(missing)}. "
                "A partial load would leave those features NaN and quietly produce a "
                "different model than the one that was fitted; see scripts/data/."
            )
        return cls(
            booster=lgb.Booster(model_file=str(model_path)),
            word_stats=WordStats.load(cache_dir / "word_stats.npz"),
            swow=maybe(SwowTables.load, "swow.npz"),
            entity=maybe(EntitySims.load, "entity_sims.npz"),
            pmi=maybe(EntitySims.load, "lm_pmi.npz"),
            extra=maybe(ExtraSims.load, "extra_sims.npz"),
            norms=maybe(WordNorms.load, "word_norms.npz"),
            wordnet=maybe(ExtraSims.load, "wordnet_sims.npz"),
            lexical=maybe(ExtraSims.load, "lexical_sims.npz"),
        )


class LearnedListenerSpymaster(Spymaster):
    def __init__(
        self,
        shortlist: int = SHORTLIST,
        sigma: float = 1.5,
        max_rarity: float = 10.0,
        k1_max_similarity: bool = False,
        neutral_cost: float | None = None,
        opponent_cost: float | None = None,
        assassin_cost: float | None = None,
        *,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        model_path: Path | None = None,
        bundle: ListenerBundle | None = None,
        clue_stats: ClueStats | None = None,
    ):
        """`sigma` tunes only the shortlisting pass, not the final ranking --
        it is the Gaussian model's parameter and survives here because that
        model is still doing the first stage. `bundle`/`clue_stats` are
        dependency-injection hooks for tests, matching
        `ExpectedWordsSpymaster`'s convention."""
        self.shortlist = shortlist
        self.max_rarity = max_rarity
        self.k1_max_similarity = k1_max_similarity
        self.costs = role_costs(neutral_cost, opponent_cost, assassin_cost)
        self.clue_stats = clue_stats if clue_stats is not None else ClueStats.load(cache_dir=cache_dir)
        # The costs go to the shortlisting stage too. They have to: the second
        # stage can only rerank what the first hands it, so leaving stage one on
        # the default costs would let a risk-averse setting be judged on a
        # shortlist built by a risk-seeking one.
        self._first_stage = ExpectedWordsSpymaster(
            sigma=sigma, max_rarity=max_rarity, cache_dir=cache_dir, clue_stats=self.clue_stats,
            neutral_cost=neutral_cost, opponent_cost=opponent_cost, assassin_cost=assassin_cost,
        )
        self.bundle = bundle if bundle is not None else ListenerBundle.load(
            cache_dir, model_path or (cache_dir / "listener_gbt.txt")
        )
        self._clue_index = {w.lower(): i for i, w in enumerate(self.clue_stats.clue_words)}

    def to_device(self, device) -> None:
        """No-op: LightGBM and numpy, CPU only."""
        return None

    def _listener_scores(
        self, clue: str, candidates: list[str], number: int, sims: SimilarityTensor
    ) -> np.ndarray | None:
        b = self.bundle
        feats = extract(clue, candidates, number, sims, self.clue_stats, b.word_stats,
                        self._clue_index, b.swow, b.entity, b.pmi, b.extra, b.norms,
                        b.wordnet, b.lexical)
        if feats is None:
            return None
        return b.booster.predict(feats, raw_score=True)

    def _score_all_clues(
        self, board: Board | OpponentBoardView, sims: SimilarityTensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(best_n, scores, margin)` indexed by `sims.clue_words`.

        Only the shortlist carries a real score; everything else is `-inf`, so
        the two stages' incomparable scales never compete. The Gaussian's
        expected reward and the listener's are both "expected words" but under
        different guesser models, and mixing them would rank by which model was
        used rather than by which clue is better.
        """
        n_clues = len(sims.clue_words)
        best_n = np.ones(n_clues, dtype=np.int64)
        scores = np.full(n_clues, -np.inf, dtype=np.float32)
        margin = np.zeros(n_clues, dtype=np.float32)

        own = board.words_by_role(Role.OWN, unrevealed_only=True)
        if not own:
            scores[:] = -self.clue_stats.rarity_percentile
            return best_n, scores, margin

        neutral = board.words_by_role(Role.NEUTRAL, unrevealed_only=True)
        opponent = board.words_by_role(Role.OPPONENT, unrevealed_only=True)
        assassin = board.words_by_role(Role.ASSASSIN, unrevealed_only=True)
        non_own = neutral + opponent + assassin
        roles = [Role.NEUTRAL] * len(neutral) + [Role.OPPONENT] * len(opponent) + [Role.ASSASSIN] * len(assassin)
        costs = np.array([self.costs[r] for r in roles], dtype=np.float64)

        # Stage one: the Gaussian model over the whole pool, for the shortlist.
        g_best_n, g_scores, g_margin = self._first_stage._score_all_clues(board, sims)
        finite = np.flatnonzero(np.isfinite(g_scores))
        if finite.size == 0:
            return g_best_n, g_scores, g_margin
        take = finite[np.argsort(-g_scores[finite])[: self.shortlist]]

        # Stage two: the listener, on the candidates in one fixed order. Roles
        # are used only to split the scores afterwards.
        candidates = own + non_own
        n_own, K_max = len(own), min(len(own), MAX_CLUE_NUMBER)
        rows, keep = [], []
        for ci in take:
            clue = sims.clue_words[ci]
            # `number` is a feature, and the clue's own best k is not known
            # until the reward is computed. K_max is the least arbitrary
            # choice available and matches how training sampled it.
            s = self._listener_scores(clue, candidates, K_max, sims)
            if s is None:
                continue
            rows.append(s)
            keep.append(ci)
        if not rows:
            return g_best_n, g_scores, g_margin

        S = np.asarray(rows, dtype=np.float64)              # (n_keep, n_candidates)
        gain, penalty = gain_and_penalty(S[:, :n_own], S[:, n_own:], costs, K_max)
        net = gain - penalty                                 # (n_keep, K_max)
        best_m = np.argmax(net, axis=1)
        idx = np.asarray(keep)
        scores[idx] = np.take_along_axis(net, best_m[:, None], axis=1)[:, 0]
        best_n[idx] = best_m + 1

        if self.k1_max_similarity:
            sub = self._swap_k1(sims, board, words=candidates, own_n=n_own,
                                scores=scores, best_n=best_n, S=S, keep=keep)
            if sub is not None:
                scores, best_n = sub
        # Margin exists only for the arena's tie-break; reuse the first stage's,
        # which is on a stable scale and is not part of the ranking here.
        margin[idx] = g_margin[idx]
        return best_n, scores, margin

    def _swap_k1(self, sims, board, words, own_n, scores, best_n, S, keep):
        """When the best clue is a k=1 clue, keep the word it means but swap the
        clue for the highest raw-similarity legal one.

        The expected reward saturates at k=1 -- once one own word dominates,
        almost any safe clue scores within noise of the best (measured: the top
        two k=1 clues on one board tied to four decimal places), so the argmax
        is settled by the third decimal rather than by which clue a teammate
        would actually get. This picks by raw cosine instead, on the argument
        that for a one-word clue the only thing that matters is how obvious the
        link is.

        **It is off by default and it is not free.** Raw similarity ignores the
        rest of the board, so it can select a clue that also points at the
        assassin -- BASEBALL for Bat carries 23x the assassin mass of CRICKET
        on one measured board, because "on deck" is a baseball term. The
        listener's tail is calibrated (words it gives under 1% are picked 341
        times against 290 predicted), so that risk is real rather than an
        artifact. Enabled deliberately, measured in the arena, not assumed.
        """
        # The best-scoring clue is often ILLEGAL -- it shares a stem with the
        # word it points at, which is exactly what makes it score well. The
        # clue actually played is the best LEGAL one, so the target has to be
        # read from that; taking it from the raw argmax made this rule answer a
        # question about a clue nobody would give.
        finite = np.flatnonzero(np.isfinite(scores))
        if finite.size == 0:
            return None
        best = None
        for ci in finite[np.argsort(-scores[finite])]:
            if is_legal_clue(sims.clue_words[int(ci)], board.words):
                best = int(ci)
                break
        if best is None or best_n[best] != 1:
            return None
        row = S[keep.index(best)] if best in keep else None
        if row is None:
            return None
        target = words[int(np.argmax(row[:own_n]))]      # the own word it means
        ti = sims.board_index.get(target.lower())
        if ti is None:
            return None

        space_i = sims._space_index(self.space) if hasattr(self, "space") else 1
        cand = np.flatnonzero(self.clue_stats.rarity_percentile <= self.max_rarity)
        cos = np.asarray(sims.tensor[cand, ti, space_i], dtype=np.float64)
        for r in np.argsort(-cos):
            ci = int(cand[r])
            if not np.isfinite(cos[r]):
                continue
            if is_legal_clue(sims.clue_words[ci], board.words):
                if ci != best:
                    scores = scores.copy(); best_n = best_n.copy()
                    scores[ci] = float(scores[best]) + 1.0   # outrank the incumbent
                    best_n[ci] = 1
                return scores, best_n
        return None

    def score_batch(self, sims: SimilarityTensor, contexts: list[TurnContext]) -> list[tuple[np.ndarray, np.ndarray]]:
        return [self._score_all_clues(ctx.board, sims)[:2] for ctx in contexts]

    def top_clues(self, ctx: TurnContext, sims: SimilarityTensor, k: int) -> list[tuple[str, int, float]]:
        best_n, scores, margin = self._score_all_clues(ctx.board, sims)
        return self._first_stage._pick_top_clues(sims, ctx.board, best_n, scores, margin, k)

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
matter how good the listener is. `SHORTLIST` is therefore a parameter. How
often the final pick changes with the depth has not been measured.

**The listener never sees roles.** Features are extracted for every unrevealed
word at once, exactly as in training, and the scores are split into ours and
theirs only afterwards, inside the reward calculation. A role-aware listener
would look excellent in the arena and be worthless as a model of a guesser.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codenames.board import Board, OpponentBoardView, Role
from codenames.clue_search import is_legal_clue
from codenames.clue_stats import ClueStats
from codenames.game import role_costs
from codenames.listener_features import (
    FEATURE_NAMES,
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

# How many "the guesser picks something the clue never meant" alternatives the
# reward prices. 0 reproduces every result recorded before the outside option
# existed, byte for byte, which is why it is the default.
#
# The listener's decoy training (scripts/data/collect_decoy_data.py) identifies
# the LEVEL of such a word -- about 4.1 nats below the best board word -- but
# not how many of them a real game contains, because a real game contains none:
# the guesser must pick from the board. So this is a free parameter of the same
# kind as `sigma` and the role costs, to be swept against play rather than
# derived. It matters a great deal: at 1 the outside option takes under 1% of
# the rate and barely moves the argmax, while at 10 it took 22% in collection.
OUTSIDE_N = 0

# Numberbatch, the space the listener's strongest features are built on and the
# one the k=1 tiebreak reads raw cosines from.
NB_SPACE = 1
NCAND = FEATURE_NAMES.index("n_candidates")


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
        outside_n: int = OUTSIDE_N,
        sigma: float = 1.5,
        max_rarity: float = 10.0,
        k1_tiebreak: bool = False,
        k1_tie_tolerance: float = 0.1,
        neutral_cost: float | None = None,
        opponent_cost: float | None = None,
        assassin_cost: float | None = None,
        exclude_acronyms: bool = True,
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
        self.outside_n = outside_n
        self.max_rarity = max_rarity
        self.k1_tiebreak = k1_tiebreak
        self.k1_tie_tolerance = k1_tie_tolerance
        self.costs = role_costs(neutral_cost, opponent_cost, assassin_cost)
        self.clue_stats = clue_stats if clue_stats is not None else ClueStats.load(cache_dir=cache_dir)
        # The costs go to the shortlisting stage too. They have to: the second
        # stage can only rerank what the first hands it, so leaving stage one on
        # the default costs would let a risk-averse setting be judged on a
        # shortlist built by a risk-seeking one.
        self._first_stage = ExpectedWordsSpymaster(
            sigma=sigma, max_rarity=max_rarity, cache_dir=cache_dir, clue_stats=self.clue_stats,
            neutral_cost=neutral_cost, opponent_cost=opponent_cost, assassin_cost=assassin_cost,
            exclude_acronyms=exclude_acronyms,
        )
        self.acronym_mask = self._first_stage.acronym_mask
        self.bundle = bundle if bundle is not None else ListenerBundle.load(
            cache_dir, model_path or (cache_dir / "listener_gbt.txt")
        )
        self._clue_index = {w.lower(): i for i, w in enumerate(self.clue_stats.clue_words)}

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        return [Path(params.get("model_path") or cache_dir / "listener_gbt.txt")]

    def _outside_words(self, board, sims: SimilarityTensor) -> list[str]:
        """`outside_n` vocabulary words that are not on this board.

        Drawn UNIFORMLY, matching how the listener's decoy training sampled
        them -- a top-similarity sample would be cheaper per clue but would
        measure a different quantity, since the level was estimated against
        uniform draws. Screened against the board by `is_legal_clue` so a
        "decoy" is never a legitimate answer in disguise, exactly as
        collect_decoy_data.py does.

        The draw is seeded from the board's own words via a stable hash, not
        `hash()`, whose string seed is randomised per process: the arena builds
        a fresh spymaster in every worker, so a process-dependent draw would
        make the same board score differently from one worker to the next.
        """
        if not self.outside_n:
            return []
        words = list(board.words)
        on_board = {w.lower() for w in words}
        digest = hashlib.blake2b("|".join(sorted(on_board)).encode(), digest_size=8).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        off = [w for w in sims.board_index if w not in on_board]
        out: list[str] = []
        for w in rng.sample(off, len(off)):
            if is_legal_clue(w, words):
                out.append(w.capitalize())
                if len(out) == self.outside_n:
                    break
        return out

    def _fixed_outside_score(self) -> float | None:
        """An outside option whose score is the same for every clue, on the
        listener's own scale -- None here. Meaningful only for a listener
        whose scores have an absolute level, which the board softmax alone
        does not give; see spymasters/association_listener.py."""
        return None

    def _listener_features(
        self, clue: str, candidates: list[str], number: int, sims: SimilarityTensor,
        n_board: int | None = None,
    ) -> np.ndarray | None:
        """The feature matrix for one clue, without scoring it.

        Split out from `_listener_scores` so the whole shortlist can be scored
        in ONE booster call. Profiled, a turn spent 3.94s of 5.3s inside
        LightGBM's predict across 200 one-clue calls, against 0.04s in
        `extract` -- it was per-call overhead on a 25-row matrix, not feature
        building, and it grew with the booster.
        """
        b = self.bundle
        feats = extract(clue, candidates, number, sims, self.clue_stats, b.word_stats,
                        self._clue_index, b.swow, b.entity, b.pmi, b.extra, b.norms,
                        b.wordnet, b.lexical)
        if feats is None:
            return None
        if n_board is not None:
            # Outside words are not board words. extract() counts whatever list
            # it is given, and training overwrote this the same way -- leaving
            # it would feed `outside_n` in as a feature the model never saw.
            feats = np.array(feats, dtype=np.float64)
            feats[:, NCAND] = n_board
        return feats

    def _listener_scores(
        self, clue: str, candidates: list[str], number: int, sims: SimilarityTensor,
        n_board: int | None = None,
    ) -> np.ndarray | None:
        """One clue's scores. Kept for callers outside the search loop."""
        feats = self._listener_features(clue, candidates, number, sims, n_board)
        return None if feats is None else self.bundle.booster.predict(feats, raw_score=True)

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
        n_board = len(candidates)
        outside = self._outside_words(board, sims)
        scored = candidates + outside
        n_own, K_max = len(own), min(len(own), MAX_CLUE_NUMBER)
        rows, keep = [], []
        for ci in take:
            clue = sims.clue_words[ci]
            # `number` is a feature, and the clue's own best k is not known
            # until the reward is computed. K_max is the least arbitrary
            # choice available and matches how training sampled it.
            f = self._listener_features(clue, scored, K_max, sims,
                                        n_board=n_board if outside else None)
            if f is None:
                continue
            rows.append(f)
            keep.append(ci)
        if not rows:
            return g_best_n, g_scores, g_margin

        # One predict over the whole shortlist. Every clue's matrix has the
        # same row count (the candidate list is fixed), so stacking and
        # reshaping back is exact -- see _listener_features on why this matters.
        n_scored = rows[0].shape[0]
        flat = self.bundle.booster.predict(np.vstack(rows), raw_score=True)
        S = np.asarray(flat, dtype=np.float64).reshape(len(keep), n_scored)
        if outside:
            # Total rate of the outside option, on the same scale as the board
            # scores: the whole point of the anchor is that this comparison is
            # meaningful, which a board-normalised softmax cannot make.
            out_block = S[:, n_board:]
            s_out = out_block.max(axis=1) + np.log(
                np.exp(out_block - out_block.max(axis=1, keepdims=True)).sum(axis=1))
            S = S[:, :n_board]
        else:
            fixed = self._fixed_outside_score()
            s_out = None if fixed is None else np.full(len(keep), fixed)
        gain, penalty = gain_and_penalty(S[:, :n_own], S[:, n_own:], costs, K_max, s_out=s_out)
        net = gain - penalty                                 # (n_keep, K_max)
        best_m = np.argmax(net, axis=1)
        idx = np.asarray(keep)
        scores[idx] = np.take_along_axis(net, best_m[:, None], axis=1)[:, 0]
        best_n[idx] = best_m + 1

        if self.k1_tiebreak:
            sub = self._swap_k1(sims, board, words=candidates, own_n=n_own,
                                scores=scores, best_n=best_n, S=S, keep=keep)
            if sub is not None:
                scores, best_n = sub
        # Margin exists only for the arena's tie-break; reuse the first stage's,
        # which is on a stable scale and is not part of the ranking here.
        margin[idx] = g_margin[idx]
        return best_n, scores, margin

    def _swap_k1(self, sims, board, words, own_n, scores, best_n, S, keep):
        """When the best clue is a k=1 clue, break the tie among near-optimal
        clues by raw similarity to the word it means.

        Expected reward saturates at k=1: once one own word dominates, every
        safe clue scores within noise of the best, so the argmax is settled by
        the third decimal rather than by which clue a teammate would actually
        get. BAT is the case -- the model preferred CRICKET where a human says
        BASEBALL, on a difference no guesser could perceive.

        **Why a tie set and not the maximum-similarity clue.** Picking the
        highest-cosine legal clue outright ignores the rest of the board, and
        the failure is not hypothetical: with CHICK and EAGLE both on the
        board and EAGLE the last word needed, the most obvious clue for EAGLE
        in isolation is BIRD -- which hands CHICK to whoever owns it. Raw
        similarity cannot see CHICK. Expected reward can, and prices it.

        So the tie set is the safeguard rather than an implementation detail:
        every candidate is already within `k1_tie_tolerance` of optimal under
        the full board-aware reward, which is computed over all remaining
        words and their roles. A clue that also points at CHICK is penalised
        by exactly that much and drops out of the set before similarity is
        ever consulted. Similarity only ever chooses among clues that are
        already safe, so the tolerance is precisely how much expected reward
        we are willing to spend on being more obvious.

        **Why 0.1.** The unit is own-words, and the best k=1 clue scores 0.96
        on average (0.845 to 0.997 over 18 forced-k=1 positions) -- a k=1 turn
        is worth about one own word and no more. So the tolerance is a
        fraction of a single turn, and it has to be read that way:

            tol    fires    worst spend    as % of the turn
            0.1     9/18       0.093             9.5%
            0.25   10/18       0.219            23.2%
            0.5    11/18       0.378            37.9%

        0.5 was the first default here and it was wrong. Clues 38% apart in
        expected reward are not tied, and swapping between them is not a
        tiebreak -- it is choosing a materially worse clue because it reads
        better. The justification offered for it was that the effect saturates
        at 0.5, 1.0 and 2.0 changing nothing more; that is an argument against
        going higher, not an argument for 0.5 over 0.1.

        0.1 still fires on half the positions, so it does the job the rule
        exists for, and caps the damage at a tenth of a turn.
        """
        # The best-scoring clue is often ILLEGAL -- it shares a stem with the
        # word it points at, which is what makes it score well. The clue
        # actually played is the best LEGAL one, so the target is read from
        # that; taking it from the raw argmax made this rule answer a question
        # about a clue nobody would give.
        finite = np.flatnonzero(np.isfinite(scores))
        if finite.size == 0:
            return None
        order = finite[np.argsort(-scores[finite])]
        best = None
        for ci in order:
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

        cutoff = float(scores[best]) - self.k1_tie_tolerance
        tie = []
        for ci in order:                                  # descending, so stop at the cutoff
            ci = int(ci)
            if scores[ci] < cutoff:
                break
            if best_n[ci] == 1 and is_legal_clue(sims.clue_words[ci], board.words):
                tie.append(ci)
        if len(tie) <= 1:
            return None

        cos = np.asarray(sims.tensor[tie, ti, NB_SPACE], dtype=np.float64)
        if not np.any(np.isfinite(cos)):
            return None
        pick = tie[int(np.nanargmax(cos))]
        if pick == best:
            return None
        scores = scores.copy()
        best_n = best_n.copy()
        scores[pick] = float(scores[best]) + 1.0          # outrank the incumbent
        best_n[pick] = 1
        return scores, best_n

    def top_clues(self, ctx: TurnContext, sims: SimilarityTensor, k: int) -> list[tuple[str, int, float]]:
        best_n, scores, margin = self._score_all_clues(ctx.board, sims)
        return self._first_stage._pick_top_clues(sims, ctx.board, best_n, scores, margin, k)

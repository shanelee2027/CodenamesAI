"""What the board is worth beyond the score: a correction to V(a, b) that
reads which words are left (docs/log.md, "Simulated games").

**Why.** V(a, b) (codenames/win_value.py) sees only how many words each side
has left. Two after-boards of the same move with the same score can differ:
the clue left our hardest word alone, or an own word next to the assassin.
The go/no-go (scripts/tools/eval_board_value.py) found the board predicts
wins beyond the score in real gpt-oss games, mainly through the assassin.

**The features,** per side, from the clue policy's clue × word tables
(s = mean z over five embedding spaces, legal clues only):
- `best_k`, k = 1..4: the cleanest k-clue's margin over the best word that
  is not this side's;
- `worst_word`, `mean_word`: each own word's best one-word margin over every
  other live word, the minimum and the mean;
- `assassin_pair`: how easily an own word and the assassin share a clue.

**The model.** A LightGBM booster fitted on simulated games
(scripts/tools/eval_sim_value.py) on top of the simulated count table's
logit. Its raw output is the board's shift of that logit; in play it is
added to the logit of whatever count table the spymaster uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SIDE_FEATURES = ["best_1", "best_2", "best_3", "best_4", "worst_word", "mean_word", "assassin_pair"]
Z_COLS = slice(0, 5)          # the five embedding z-scores in PolicyFeatures.pair
NEG = -1e9


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def board_matrix(pf, word_ids: np.ndarray, device: str):
    """(25, C) clue × word strength for one board, with illegal clues at NEG."""
    import torch

    z = np.asarray(pf.pair[word_ids][:, :, Z_COLS], dtype=np.float32)          # (25, C, 5)
    S = torch.tensor(np.nanmean(np.where(np.isfinite(z), z, np.nan), axis=2), device=device)
    S = torch.nan_to_num(S, nan=0.0)
    legal = torch.tensor(pf.legal_mask(word_ids), device=device)
    return torch.where(legal[None, :], S, torch.tensor(NEG, device=device))


def side_features_batch(Sl, own, bad, ass):
    """The SIDE_FEATURES of one side for N states of one board. `Sl` is (25, C)
    from `board_matrix`; own/bad/ass are (N, 25) boolean masks over the
    unrevealed words. Masked words are filled with the same NEG as illegal
    clues, so every max, top-k and empty case comes out as in the one-state
    definition in scripts/tools/eval_board_value.py (checked on 500 simulated
    states: they agree to 7e-7)."""
    import torch

    N = own.shape[0]
    out = torch.full((N, len(SIDE_FEATURES)), float("nan"), device=Sl.device)
    n_own = own.sum(1)
    S3 = Sl[None].expand(N, -1, -1)                                      # (N, 25, C)
    neg = torch.full_like(S3, NEG)
    badmax = torch.where(bad[:, :, None], S3, neg).max(1).values         # (N, C)
    top = torch.topk(torch.where(own[:, :, None], S3, neg), 4, dim=1).values   # (N, 4, C)
    best = (top - badmax[:, None, :]).max(2).values                      # (N, 4)
    out[:, :4] = torch.where(torch.arange(1, 5, device=Sl.device)[None, :] <= n_own[:, None], best, out[:, :4])
    # One-word margins: the best other live word is the live words' top, or
    # their second where the word itself is the top (a tie gives the same).
    t2 = torch.topk(torch.where((own | bad)[:, :, None], S3, neg), 2, dim=1)
    word = torch.arange(25, device=Sl.device)[None, :, None]
    other = torch.where(t2.indices[:, :1, :] == word, t2.values[:, 1:2, :], t2.values[:, :1, :])  # (N, 25, C)
    per_word = (S3 - other).max(2).values                                # (N, 25)
    has = n_own > 0
    out[:, 4] = torch.where(has, torch.where(own, per_word, torch.full_like(per_word, float("inf"))).min(1).values,
                            out[:, 4])
    out[:, 5] = torch.where(has, (per_word * own).sum(1) / n_own.clamp(min=1), out[:, 5])
    a = ass.float().argmax(1)                                            # the assassin's slot, if unrevealed
    pair = torch.minimum(S3, Sl[a][:, None, :]).max(2).values            # (N, 25)
    ap = torch.where(own, pair, torch.full_like(pair, -float("inf"))).max(1).values
    out[:, 6] = torch.where(has & ass.any(1), ap, out[:, 6])
    return out.cpu().numpy()


class BoardValue:
    """The board's shift of the win-probability logit for the side to move.

    `term` takes N states of one board as masks over its 25 words (unrevealed
    only) and returns (N,) logit shifts, or None when a board word is outside
    the clue policy's tables (the caller then uses the score alone)."""

    CHUNK = 128                # states per GPU batch: (128, 25, C) float32 is ~140 MB

    def __init__(self, model_path: Path, counts_path: Path, cache_dir: Path):
        import lightgbm as lgb
        import torch

        from codenames.clue_policy import PolicyFeatures

        self.booster = lgb.Booster(model_file=str(model_path))
        self.count_logit = logit(np.load(counts_path)["V"])
        self.pf = PolicyFeatures.load(cache_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._matrix: tuple[tuple, object] | None = None

    def word_ids(self, words: list[str]) -> np.ndarray | None:
        try:
            return np.array([self.pf.board_index[w.lower()] for w in words])
        except KeyError:
            return None

    def term(self, word_ids: np.ndarray, mover_own: np.ndarray, other_own: np.ndarray,
             neutral: np.ndarray, assassin: np.ndarray) -> np.ndarray:
        import torch

        key = tuple(word_ids.tolist())
        if self._matrix is None or self._matrix[0] != key:
            self._matrix = (key, board_matrix(self.pf, word_ids, self.device))
        Sl = self._matrix[1]
        nf = len(SIDE_FEATURES)
        N = mover_own.shape[0]
        F = np.empty((N, 2 * nf), dtype=np.float32)
        for i in range(0, N, self.CHUNK):
            sl = slice(i, i + self.CHUNK)
            m, o, n, a = (torch.tensor(x[sl], device=self.device) for x in (mover_own, other_own, neutral, assassin))
            F[sl, :nf] = side_features_batch(Sl, m, o | n | a, a)
            F[sl, nf:] = side_features_batch(Sl, o, m | n | a, a)
        a_, b_ = mover_own.sum(1), other_own.sum(1)
        X = np.column_stack([a_, b_, self.count_logit[a_, b_], F])
        return np.asarray(self.booster.predict(X, raw_score=True), dtype=np.float64)

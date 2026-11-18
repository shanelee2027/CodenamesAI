"""A guesser that behaves exactly like a fitted listener (docs/log.md,
"Simulated games").

It reads a clue the way a listener spymaster's booster reads it (the same
features, codenames/spymasters/learned_listener.py) and samples a ranking
pick by pick:
- pick 1 from a softmax over the booster's scores;
- each later pick from the within-turn model over the words left
  (codenames/sequential_listener.py: flatter, and pulled toward the words
  already picked), or from the same scores renormalised with
  `sequential_path=None` (the frozen Plackett-Luce model).

The game loop reads the ranking exactly as it reads gpt-oss's: the first
`number` words, stopping at the first that is not the team's own. So games
played with this guesser are games in which the listener is the truth, at no
API cost. That is what the simulated-games value model is trained on; its
gap to gpt-oss is measured, not assumed: configs/eval_suite_sim_assoc.json
replays the gpt-oss suite's boards with it (docs/log.md, "Simulated games:
simulator fidelity").

The k feature is the announced number, as in the listener's training data
(the spymaster's search scores at its number cap instead).

`pick_temperatures` (preset `fresh_split`, the thesis's GBT) draws pick j from
softmax(s / tau_j) over the words left, tau_j the GBT spymaster's fitted
temperature for pick j (picks past the last share it), which is exactly the
model `fresh_split_listener` scores clues with. That preset reads features
through `FreshSplitListenerSpymaster` itself, so the guesser and the
spymaster see the same columns and normalisation.

Sampling is deterministic in (seed, clue, candidates, number): rerunning a
game reproduces it, and the two seatings of a board share draws whenever they
reach the same position with the same clue (common random numbers).
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from codenames.guessers.base import Guesser
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

PRESETS = {
    # name: (booster, within-turn parameters fitted on it)
    "assoc": ("listener_gbt_assoc_features.txt", "sequential_listener_assoc.json"),
    "conceptnet": ("listener_gbt_conceptnet.txt", "sequential_listener.json"),
    "incumbent": ("listener_gbt.txt", None),
    # The thesis's GBT: frozen scores, a temperature per pick (see the docstring).
    "fresh_split": ("listener_gbt_fresh_split.txt", None),
}


class ListenerSampleGuesser(Guesser):
    def __init__(self, model_path: Path, sequential_path: Path | None = None, seed: int = 0,
                 cache_dir: Path = DEFAULT_CACHE_DIR, pick_temperatures: tuple[float, ...] | None = None,
                 reader: str = "assoc"):
        from codenames.sequential_listener import SequentialParams

        # The spymaster class is only borrowed for its feature extraction and
        # booster; it reads whichever columns the booster was fitted on.
        if reader == "fresh_split":
            from codenames.spymasters.fresh_split_listener import FreshSplitListenerSpymaster

            self._reader = FreshSplitListenerSpymaster(cache_dir=cache_dir, model_path=Path(model_path), shortlist=1)
        else:
            from codenames.spymasters.assoc_feature_listener import AssocFeatureListenerSpymaster

            self._reader = AssocFeatureListenerSpymaster(cache_dir=cache_dir, model_path=Path(model_path))
        self.model_path = Path(model_path)
        self.seq = (SequentialParams.from_dict(json.loads(Path(sequential_path).read_text()))
                    if sequential_path else None)
        if self.seq is not None and pick_temperatures is not None:
            raise ValueError("pick temperatures replace the within-turn model; give one or the other")
        self.pick_temperatures = tuple(float(t) for t in pick_temperatures) if pick_temperatures else None
        self.seed = seed
        self._vw = None

    @classmethod
    def preset(cls, name: str, seed: int = 0, cache_dir: Path = DEFAULT_CACHE_DIR) -> "ListenerSampleGuesser":
        booster, seq = PRESETS[name]
        if name == "fresh_split":
            from codenames.spymasters.fresh_split_listener import PICK_TEMPERATURES

            return cls(Path(cache_dir) / booster, None, seed, cache_dir,
                       pick_temperatures=PICK_TEMPERATURES, reader="fresh_split")
        return cls(Path(cache_dir) / booster, Path(cache_dir) / seq if seq else None, seed, cache_dir)

    def scores(self, clue: str, candidate_words: list[str], sims: SimilarityTensor, number: int) -> np.ndarray | None:
        f = self._reader._listener_features(clue, list(candidate_words), number, sims)
        if f is None:
            return None
        return np.asarray(self._reader.bundle.booster.predict(f, raw_score=True), dtype=np.float64)

    def score_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor) -> dict[str, float]:
        s = self.scores(clue, candidate_words, sims, 1)
        if s is None:
            return {w: 0.0 for w in candidate_words}
        return dict(zip(candidate_words, s.tolist()))

    def rank_candidates(self, clue: str, candidate_words: list[str], sims: SimilarityTensor,
                        number: int | None = None) -> list[str]:
        from codenames.sequential_listener import pair_similarity, step_logits

        words = list(candidate_words)
        n = len(words)
        k = max(1, int(number or 1))
        rng = np.random.default_rng([self.seed, zlib.crc32(clue.lower().encode()),
                                     zlib.crc32("|".join(sorted(w.lower() for w in words)).encode()), k])
        s = self.scores(clue, words, sims, k)
        if s is None:                        # no features: an uninformed guesser
            return [words[i] for i in rng.permutation(n)]
        if self.seq is not None and self._vw is None:
            from codenames.listener_net import WordVectors

            self._vw = WordVectors()
        sim = pair_similarity(words, self._vw) if self.seq is not None else None
        sense = np.full(n, -1)
        left = np.ones(n, bool)
        order: list[int] = []
        for step in range(1, min(k, n) + 1):
            picked = ~left
            if self.pick_temperatures is not None:
                tau = self.pick_temperatures[min(step, len(self.pick_temperatures)) - 1]
                z = np.where(left, s / tau, -np.inf)
            elif self.seq is None or step == 1:
                z = np.where(left, s, -np.inf)
            else:
                z = step_logits(s, picked, left, sim, sense, step, self.seq)
            p = np.exp(z - z.max())
            p /= p.sum()
            i = int(rng.choice(n, p=p))
            order.append(i)
            left[i] = False
        rest = [i for i in np.argsort(-s) if left[i]]
        return [words[i] for i in order + rest]

    def __repr__(self) -> str:
        temps = f", pick_temperatures={self.pick_temperatures}" if self.pick_temperatures else ""
        return (f"ListenerSampleGuesser({self.model_path.name}, within_turn={self.seq is not None}{temps}, "
                f"seed={self.seed})")

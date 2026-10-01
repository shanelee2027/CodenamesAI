"""A listener spymaster that picks the clue and number with the highest
probability of winning (codenames/win_value.py;
docs/versions/win_prob_listener.md).

**Why.** Every listener spymaster so far maximised expected own words minus
hand-set costs (0.2 / 1 / 10) for the current turn. That objective cannot
see tempo, which is what wins against gpt-oss. The evidence:
- every change that made the spymaster more cautious lost (role costs, the
  outside option, pick temperatures, within_turn_listener);
- a better listener was confounded with the objective it was plugged into.

Here the listener says how the turn will end. V(a, b), the incumbent's
probability of winning from a score position (estimated from 10,266
recorded games, scripts/data/build_win_value.py), says what each ending is
worth. No costs are left anywhere.

**The opponent is the incumbent.** V was estimated from games in which both
sides played like it. Choosing clues greedily against V is one step of
policy iteration: the best clue now, assuming later turns are played like
the incumbent's.

**Parameters, so the objective can be held fixed while the listener varies:**
- `model_path`: the booster. The default is conceptnet_listener's, the
  best-supported listener. Any booster works, since each reads its own
  features.
- `turn_model`: how picks 2+ of a turn are modelled.
  - `"frozen"`: Plackett-Luce with frozen scores, as the incumbent.
  - `"within_turn"`: within_turn_listener's model
    (codenames/sequential_listener.py).
- `value_path`: the V table.
- `calibration_path`: optional. A turn calibration fitted on the turns
  this spymaster actually played (scripts/pipeline/fit_turn_calibration.py):
  every listener score becomes alpha * s + beta[role] before the objective
  reads it. On chosen clues the listener rates own words too high and bad
  words too low (the search picks the clues whose errors flatter them). The
  role offsets undo that selection bias, and alpha its overall confidence.

Search and shortlist are learned_listener's. The shortlist's Gaussian stage
still ranks by its own cost-based reward: it only has to keep good clues
among the 200, not order them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from codenames.game import Role
from codenames.sequential_listener import SequentialParams, pair_similarity, sequential_gain_and_penalty
from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.spymasters.conceptnet_listener import ConceptnetListenerSpymaster
from codenames.win_value import WinValue, win_probability

VALUE = "win_value.npz"
SEQUENTIAL = "sequential_listener.json"
_VW: dict = {}


class WinProbListenerSpymaster(ConceptnetListenerSpymaster):
    def __init__(self, *args, cache_dir: Path = DEFAULT_CACHE_DIR, value_path: Path | None = None,
                 turn_model: str = "frozen", sequential_path: Path | None = None,
                 calibration_path: Path | None = None, **kwargs):
        super().__init__(*args, cache_dir=cache_dir, **kwargs)
        self.calibration = json.loads(Path(calibration_path).read_text()) if calibration_path else None
        self.value = WinValue(Path(value_path) if value_path else Path(cache_dir) / VALUE)
        if turn_model not in ("frozen", "within_turn"):
            raise ValueError(f"turn_model must be 'frozen' or 'within_turn', not {turn_model!r}")
        self.turn_model = turn_model
        self.sequential = None
        if turn_model == "within_turn":
            from codenames.listener_net import WordVectors

            path = Path(sequential_path) if sequential_path else Path(cache_dir) / SEQUENTIAL
            self.sequential = SequentialParams.from_dict(json.loads(path.read_text()))
            if "vw" not in _VW:
                _VW["vw"] = WordVectors()
            self._vw = _VW["vw"]

    @classmethod
    def model_files(cls, params: dict) -> list[Path]:
        cache_dir = Path(params.get("cache_dir", DEFAULT_CACHE_DIR))
        files = [*super().model_files(params), Path(params.get("value_path") or cache_dir / VALUE)]
        if params.get("calibration_path"):
            files.append(Path(params["calibration_path"]))
        if params.get("turn_model", "frozen") == "within_turn":
            files += [Path(params.get("sequential_path") or cache_dir / SEQUENTIAL), cache_dir / "word_vectors.npz"]
        return files

    def _calibrated(self, s_own, s_bad, roles):
        """alpha * s + beta[role], or the scores unchanged without a calibration."""
        c = self.calibration
        if c is None:
            return s_own, s_bad
        beta = {Role.NEUTRAL: 0.0, Role.OPPONENT: c["beta_opponent"], Role.ASSASSIN: c["beta_assassin"]}
        shift = np.array([beta[r] for r in roles])
        return c["alpha"] * s_own + c["beta_own"], c["alpha"] * s_bad + shift[None, :]

    def _clue_values(self, s_own, s_bad, roles, costs, max_k, s_out, words=None):
        s_own, s_bad = self._calibrated(np.asarray(s_own, dtype=np.float64), np.asarray(s_bad, dtype=np.float64), roles)
        if self.turn_model == "within_turn":
            if words is None:
                raise ValueError("the within-turn model needs the board words")
            sim = pair_similarity(words, self._vw)

            def gp(c):
                return sequential_gain_and_penalty(s_own, s_bad, c, max_k, sim, self.sequential, s_out)
        else:
            def gp(c):
                return self._gain_and_penalty(s_own, s_bad, c, max_k, s_out, words=words)

        n_own = s_own.shape[1]
        n_opp = sum(r == Role.OPPONENT for r in roles)
        values = win_probability(gp, roles, max_k, n_own, n_opp, self.value)
        return values, self.value.one_word(n_own, n_opp)

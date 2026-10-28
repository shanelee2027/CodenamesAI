"""Play Codenames against the real spymaster, on localhost -- and run a blind
one-clue evaluation of spymasters against a human guesser.

The published artifact runs a JavaScript port of the model over ~20 boards whose
raw feature inputs were exported ahead of time -- it has to, because a browser
cannot hold the 11,145 x 25 x 3 similarity tensor. That buys portability at the
cost of a fixed board pool. Here there is no such constraint: this serves the
pages and calls `LearnedListenerSpymaster` directly, so every board is freshly
generated and the clue is the same one the arena would see.

Three pages:

    /       a full game. Pick the spymaster from a menu (SPYMASTERS below);
            the choice applies from the next clue. When the game ends, a clue
            history lists every clue with the words it was meant for, its near
            misses (as on /compare) and what was guessed (/api/explain).
    /eval   the blind study. One position, one clue, you guess, next. Each
            clue comes from one of `--eval-arms`, assigned here and never sent
            to the browser. Every turn is appended to `cache/human_eval.jsonl`;
            scripts/tools/analyze_human_eval.py reads it.
    /compare  two spymasters side by side, on positions where their clues
            differ: the board in spymaster view, each clue with the own words
            it is meant for, and each model's predicted first guesses. You
            vote which is better; votes go to `cache/compare_votes.jsonl`.

**Why the eval is built the way it is**, since each choice protects the data:

- *The arm and the key stay on the server.* The page gets the words, the roles
  of cards already revealed, and the clue -- nothing else. Each pick is a round
  trip that returns that one card's role. A guesser cannot learn which
  spymaster they are facing, or the key, even from devtools.
- *Arms are assigned in shuffled blocks of one each*, so the two stay balanced
  to within one position however long a session runs.
- *Stopping is a first-class outcome.* A human who no longer sees a connection
  passes; gpt-oss never does, which is exactly why the arena cannot evaluate the
  outside option. The Stop button is what records it.
- *There is no skip.* If people skipped bad clues more than good ones, the
  skips would depend on the arm -- the same arm-specific censoring that made
  the guesser-refusal discards a worry in the arena sweeps.
- *Positions span a game.* 0-8 non-assassin cards are pre-revealed at random,
  as the listener's training positions were, rather than always turn one.

Boards in the full game are addressed by seed, and the client sends the seed
plus the revealed words back with every request, so that page keeps no
server-side state. The eval page does keep state -- it has to, to hold the key.

The k=1 tiebreak (`k1_tiebreak`) is ON by default here and OFF in
`configs/spymasters.json`. Among clues already within `k1_tie_tolerance` of
optimal under the full board-aware reward, it plays the one most obviously tied
to the word it means -- BASEBALL for Bat, rather than whichever safe clue wins
by the third decimal. Pass --no-k1 to turn it off.

Usage:
    python scripts/tools/play_server.py                    # http://127.0.0.1:8000
    python scripts/tools/play_server.py --eval-arms assoc_profile,profile_temperature
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

# Two layouts: in the repo this file sits at scripts/tools/, and in a demo
# bundle (scripts/tools/make_demo_bundle.py) it sits beside `codenames/` at the
# top level. Pick whichever actually contains the package rather than assuming
# a depth, so the bundle does not depend on where it was unzipped.
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next((c for c in (_HERE.parents[1], _HERE) if (c / "codenames").is_dir()), _HERE)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Board, OpponentBoardView, Role, clue_number_cap
from codenames.clue_stats import ClueStats
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
from codenames.spymasters.base import TurnContext
from codenames.spymasters.learned_listener import ListenerBundle, LearnedListenerSpymaster
from codenames.spymasters.pick_index_lookahead_listener import PickIndexLookaheadListenerSpymaster
from codenames.spymasters.assoc_profile_listener import AssocProfileListenerSpymaster
from codenames.spymasters.profile_temperature_listener import ProfileTemperatureListenerSpymaster
from codenames.spymasters.stop_listener import StopListenerSpymaster
from codenames.spymasters.stop_net_words_listener import StopNetWordsListenerSpymaster
from codenames.spymasters.overpromise_listener import OverpromiseListenerSpymaster
from codenames.spymasters.win_prob_overpromise_listener import WinProbOverpromiseListenerSpymaster
from codenames.spymasters.margin_value_overpromise_listener import MarginValueOverpromiseListenerSpymaster
from codenames.spymasters.floored_overpromise_listener import FlooredOverpromiseListenerSpymaster
from codenames.spymasters.win_prob_listener import WinProbListenerSpymaster

PAGES = {"/": "index.html", "/index.html": "index.html", "/eval": "eval.html",
         "/compare": "compare.html"}
# The tab bar every page carries, defined once and injected at __NAV__.
TABS = [("/", "Play a game"), ("/eval", "Guess blind"), ("/compare", "Compare clues")]
_NAV_STYLE = """<style>
.tabs{display:flex;gap:4px;margin:0 0 12px;padding:3px;border:1px solid var(--line);
  border-radius:11px;background:var(--surface)}
.tabs a{flex:1;text-align:center;padding:7px 6px;border-radius:8px;font-size:12.5px;
  font-weight:600;color:var(--muted);text-decoration:none;white-space:nowrap}
.tabs a:hover{color:var(--ink)}
.tabs a[aria-current=page]{background:var(--gold);color:#1A1509}
.tabs a:focus-visible{outline:2px solid var(--gold);outline-offset:1px}
</style>"""


def nav(path: str) -> str:
    here = PAGES.get(path)
    links = "".join(
        f'<a href="{href}"{" aria-current=page" if PAGES[href] == here else ""}>{label}</a>'
        for href, label in TABS)
    return f'{_NAV_STYLE}<nav class="tabs" aria-label="Pages">{links}</nav>'


WEB = Path(__file__).parent / "webplay"
ROLE_CODE = {Role.OWN: "a", Role.OPPONENT: "b", Role.NEUTRAL: "n", Role.ASSASSIN: "x"}
DEFAULT_EVAL_LOG = DEFAULT_CACHE_DIR / "human_eval.jsonl"
DEFAULT_COMPARE_LOG = DEFAULT_CACHE_DIR / "compare_votes.jsonl"

# Every spymaster the pages can use. The menu is for the author's own play,
# so it carries only the current models; older ones stay in the registry
# (configs/spymasters.json) and their docs, just not here. An entry whose
# model file is missing is simply not offered -- the laptop demo bundle may
# not carry every booster.
SPYMASTERS: dict[str, dict] = {
    # assoc_features plus the clue's association profile and reverse
    # associations (codenames/spymasters/assoc_profile_listener.py): the
    # highest-R2 single-score booster, with the September incumbent's
    # net-words reward. The incumbent since 2026-10-16 (docs/log.md).
    "assoc_profile": {
        "label": "Incumbent (association-profile listener)", "model": "listener_gbt_assoc_profile.txt", "outside_n": 0,
        "cls": AssocProfileListenerSpymaster, "kwargs": {"outside_n": 0},
        "needs": ["isa_sims.npz", "conceptnet_sims.npz", "assoc_sims.npz", "assoc_profile.npz"],
        "about": "The incumbent since 2026-10-16: the September spymaster with the best plain booster "
                 "(67 features: ConceptNet, gpt-oss's free associations, how vague the clue's associations "
                 "are and whether each board word's own associations name the clue). Chooses by expected "
                 "net words (+1, -0.2, -1, -10); its guesser never stops. Beat the September incumbent "
                 "56.7% on the gpt-oss suite (22 boards swept vs 9, p = 0.03)."},
    # The association-profile listener with per-pick temperatures fitted for
    # its booster (codenames/spymasters/profile_temperature_listener.py).
    "profile_temperature": {
        "label": "Association-profile listener + pick temperatures", "model": "listener_gbt_assoc_profile.txt",
        "outside_n": 0, "cls": ProfileTemperatureListenerSpymaster, "kwargs": {"outside_n": 0},
        "needs": ["isa_sims.npz", "conceptnet_sims.npz", "assoc_sims.npz", "assoc_profile.npz",
                  "sequential_listener_assoc_profile_temperature.json"],
        "about": "The association-profile listener, but the guesser's 2nd, 3rd and 4th picks are "
                 "less certain than its first (temperatures 1.09, 1.26, 1.41, fitted on gpt-oss), so "
                 "clues that rely on weak later words are worth less. Expected net words (+1, -0.2, -1, -10)."},
    # win_prob_listener with the assoc booster and frozen scores
    # (codenames/spymasters/win_prob_listener.py): the configuration that won
    # 59.1% against learned_listener on the gpt-oss suite (105dc03b7f48). The
    # k=1 tiebreak is off, as in that run.
    "win_prob_assoc": {
        "label": "Win probability (association-feature listener)", "model": "listener_gbt_assoc_features.txt",
        "outside_n": 0, "cls": WinProbListenerSpymaster, "kwargs": {"outside_n": 0}, "no_k1": True,
        "score_is": "win",
        "needs": ["isa_sims.npz", "conceptnet_sims.npz", "assoc_sims.npz", "win_value.npz"],
        "about": "Chooses the clue that maximises the chance of winning after the turn (V(a, b) from "
                 "10,266 gpt-oss games), with the association-feature listener and one score per word. "
                 "Its guesser never stops. ~1 s per clue."},
    # Win-probability objective with the assoc booster, a pick-index model of
    # the turn and a one-move lookahead at the incumbent's reply
    # (codenames/spymasters/pick_index_lookahead_listener.py). Its value is a
    # P(win), not net words, so the k=1 tiebreak (tuned in net words) is off,
    # as in the eval suite.
    "pick_index_lookahead": {
        "label": "Pick-index lookahead (win probability)", "model": "listener_gbt_assoc_features.txt",
        "outside_n": 0, "cls": PickIndexLookaheadListenerSpymaster, "kwargs": {"outside_n": 0},
        "no_k1": True,
        "needs": ["isa_sims.npz", "conceptnet_sims.npz", "assoc_sims.npz", "win_value.npz",
                  "listener_gbt_pick_indexassoc_depth9.txt", "reply_offset_pick_index.npz"],
        "about": "Chooses the clue that maximises the chance of winning after the turn and the "
                 "incumbent's reply. Its listener is the association-feature model, with a "
                 "pick-index booster for how the guesser's 2nd, 3rd and 4th picks go. ~1.5 s per clue."},
    # win_prob_listener with a guesser that can end its turn
    # (codenames/spymasters/stop_listener.py), trained on gpt-oss's
    # "which words does this clue point at" labels. Its explanation shows
    # pick 1 under that listener, with STOP as the pass.
    "stop": {
        "label": "Stop listener (win probability)", "model": "listener_gbt_assoc_profile.txt",
        "outside_n": 0, "cls": StopListenerSpymaster, "kwargs": {"outside_n": 0}, "no_k1": True,
        "needs": ["isa_sims.npz", "conceptnet_sims.npz", "assoc_sims.npz", "assoc_profile.npz", "win_value.npz",
                  "listener_gbt_stop.txt"],
        "about": "Chooses the clue that maximises the chance of winning after the turn, with a "
                 "guesser that stops when the clue points at nothing else, learned from which "
                 "words gpt-oss says a clue points at (asked without the number). PASS in the "
                 "explanation is that guesser's chance of stopping at once."},
    # The same, with the STOP listener trained on the strict "guess" set only.
    "stop_guess": {
        "label": "Stop listener, strict (win probability)", "model": "listener_gbt_assoc_profile.txt",
        "outside_n": 0, "cls": StopListenerSpymaster, "no_k1": True,
        "kwargs": {"outside_n": 0, "stop_model_path": DEFAULT_CACHE_DIR / "listener_gbt_stop_guess.txt"},
        "needs": ["isa_sims.npz", "conceptnet_sims.npz", "assoc_sims.npz", "assoc_profile.npz", "win_value.npz",
                  "listener_gbt_stop_guess.txt"],
        "about": "The stop listener, trained only on the words gpt-oss says a clue clearly points "
                 "at (not the 'stretch' ones), so its guesser stops sooner and its numbers are smaller."},
    # The two stop listeners with the incumbent's reward: expected net words
    # (+1, -0.2, -1, -10) instead of P(win) (codenames/spymasters/stop_net_words_listener.py).
    **{key: {"label": label, "model": "listener_gbt_assoc_profile.txt", "outside_n": 0,
             "cls": StopNetWordsListenerSpymaster, "no_k1": True,
             "kwargs": {"outside_n": 0, "stop_model_path": DEFAULT_CACHE_DIR / stop},
             "needs": ["isa_sims.npz", "conceptnet_sims.npz", "assoc_sims.npz", "assoc_profile.npz",
                       "win_value.npz", stop],
             "about": about}
       for key, label, stop, about in (
           ("stop_net", "Stop listener (net words)", "listener_gbt_stop.txt",
            "The stop listener, choosing by expected net words (+1 per own word, -0.2 neutral, "
            "-1 opponent, -10 assassin) instead of the chance of winning."),
           ("stop_net_guess", "Stop listener, strict (net words)", "listener_gbt_stop_guess.txt",
            "The strict stop listener, choosing by expected net words (+1 per own word, -0.2 "
            "neutral, -1 opponent, -10 assassin) instead of the chance of winning."))},
    # The net-words stop listeners with a cost per promised word the guesser
    # does not attempt, the number chosen by the objective
    # (codenames/spymasters/overpromise_listener.py).
    **{key: {"label": label, "model": "listener_gbt_assoc_profile.txt", "outside_n": 0,
             "cls": OverpromiseListenerSpymaster, "no_k1": True,
             "kwargs": {"outside_n": 0, "stop_model_path": DEFAULT_CACHE_DIR / stop, "promise_cost": lam},
             "needs": ["isa_sims.npz", "conceptnet_sims.npz", "assoc_sims.npz", "assoc_profile.npz",
                       "win_value.npz", stop],
             "about": f"{about} Each promised word the guesser stops short of costs {lam} words, and the "
                      "number is the one with the best expected net words after that cost."}
       for key, label, stop, lam, about in (
           ("overpromise", "Overpromise penalty 0.5, loose (guess + stretch)", "listener_gbt_stop.txt", 0.5,
            "The stop listener (net words), trained on guess + stretch: it expects the guesser to "
            "stop when the stretch words run out."),
           ("overpromise_1", "Overpromise penalty 1, loose (guess + stretch)", "listener_gbt_stop.txt", 1.0,
            "The stop listener (net words), trained on guess + stretch: it expects the guesser to "
            "stop when the stretch words run out."),
           ("overpromise_guess", "Overpromise penalty 0.5, strict (guess only)", "listener_gbt_stop_guess.txt", 0.5,
            "The strict stop listener (net words), trained on guess only: it expects the guesser to "
            "stop when the clearly-meant words run out."),
           ("overpromise_guess_1", "Overpromise penalty 1, strict (guess only)", "listener_gbt_stop_guess.txt", 1.0,
            "The strict stop listener (net words), trained on guess only: it expects the guesser to "
            "stop when the clearly-meant words run out."))},
    # The strict stop listener's P(win) objective with the overpromise cost:
    # each promised word the guesser stops short of costs `promise_cost` own
    # words, in win probability at the current score; the number is the
    # objective's (codenames/spymasters/win_prob_overpromise_listener.py).
    **{key: {"label": label, "model": "listener_gbt_assoc_profile.txt", "outside_n": 0,
             "cls": WinProbOverpromiseListenerSpymaster, "no_k1": True,
             "kwargs": {"outside_n": 0, "stop_model_path": DEFAULT_CACHE_DIR / "listener_gbt_stop_guess.txt",
                        "promise_cost": lam},
             "needs": ["isa_sims.npz", "conceptnet_sims.npz", "assoc_sims.npz", "assoc_profile.npz",
                       "win_value.npz", "listener_gbt_stop_guess.txt"],
             "about": "The strict stop listener (trained on guess only), choosing by the chance of winning "
                      f"after the turn. Each promised word the guesser stops short of costs {lam} of an own "
                      "word's worth in win probability at the current score, and the number is the one with "
                      "the best value after that cost. The value shown is P(win) minus that cost."}
       for key, label, lam in (
           ("win_prob_overpromise", "Win probability + overpromise penalty 0.5, strict (guess only)", 0.5),
           ("win_prob_overpromise_1", "Win probability + overpromise penalty 1, strict (guess only)", 1.0))},
    # The same on a V table without ties: every step at least half the
    # logistic prior's, so the cost also acts on the opening clue
    # (codenames/spymasters/margin_value_overpromise_listener.py).
    **{key: {"label": label, "model": "listener_gbt_assoc_profile.txt", "outside_n": 0,
             "cls": MarginValueOverpromiseListenerSpymaster, "no_k1": True,
             "kwargs": {"outside_n": 0, "stop_model_path": DEFAULT_CACHE_DIR / "listener_gbt_stop_guess.txt",
                        "promise_cost": lam},
             "needs": ["isa_sims.npz", "conceptnet_sims.npz", "assoc_sims.npz", "assoc_profile.npz",
                       "win_value_margin.npz", "listener_gbt_stop_guess.txt"],
             "about": "Win probability + overpromise penalty, strict, on a V table without ties (same "
                      "10,266 games; every step at least half the smooth fit's), so the cost also acts "
                      f"on the opening clue. Each promised word the guesser stops short of costs {lam} of "
                      "an own word's worth in win probability. The value shown is P(win) minus that cost."}
       for key, label, lam in (
           ("margin_value_overpromise", "Win probability + overpromise 0.5, strict, V without ties", 0.5),
           ("margin_value_overpromise_1", "Win probability + overpromise 1, strict, V without ties", 1.0))},
    # The same with a floor under the cost, so it does not vanish where one
    # word cannot change the game (codenames/spymasters/floored_overpromise_listener.py).
    **{key: {"label": label, "model": "listener_gbt_assoc_profile.txt", "outside_n": 0,
             "cls": FlooredOverpromiseListenerSpymaster, "no_k1": True,
             "kwargs": {"outside_n": 0, "stop_model_path": DEFAULT_CACHE_DIR / "listener_gbt_stop_guess.txt",
                        "promise_cost": lam},
             "needs": ["isa_sims.npz", "conceptnet_sims.npz", "assoc_sims.npz", "assoc_profile.npz",
                       "win_value_margin.npz", "listener_gbt_stop_guess.txt"],
             "about": "Win probability + overpromise penalty, strict, V without ties, with a floor under "
                      f"the penalty: each promised word the guesser stops short of costs {lam} times the "
                      "larger of one own word's worth in win probability and the floor, so the penalty "
                      "stays on when behind. The value shown is P(win) minus that cost."}
       for key, label, lam in (
           ("floored_overpromise", "Win probability + overpromise 0.5, strict, V without ties, floored", 0.5),
           ("floored_overpromise_1", "Win probability + overpromise 1, strict, V without ties, floored", 1.0))},
}

# Clue computation holds one lock: LightGBM and the tensor are shared, and a
# single user never needs two clues at once. ~0.7-1.1 s per clue.
_LOCK = threading.Lock()


class Engine:
    def __init__(self, k1: bool, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = cache_dir
        self.k1 = k1
        self.sims = SimilarityTensor.load(cache_dir)
        self.clue_stats = ClueStats.load(cache_dir=cache_dir)
        self._bundles: dict[str, ListenerBundle] = {}
        self._spymasters: dict[str, LearnedListenerSpymaster] = {}
        # The search score of each clue given, for entries whose score is the
        # value they chose on ("score_is": "win") but which have no
        # `clue_value` to recompute it: the explanation shows that score.
        self._scores: dict[tuple, float] = {}

    def available(self) -> list[str]:
        return [k for k, v in SPYMASTERS.items()
                if all((self.cache_dir / f).exists() for f in [v["model"], *v.get("needs", [])])]

    def spymaster(self, key: str) -> LearnedListenerSpymaster:
        """Built on first use and kept. Call under _LOCK."""
        if key not in SPYMASTERS or key not in self.available():
            raise ValueError(f"unknown or unavailable spymaster {key!r}; have {self.available()}")
        if key not in self._spymasters:
            spec = SPYMASTERS[key]
            if spec.get("policy"):
                self._spymasters[key] = spec["cls"](model_path=self.cache_dir / spec["model"],
                                                    cache_dir=self.cache_dir)
                return self._spymasters[key]
            if spec["model"] not in self._bundles:
                self._bundles[spec["model"]] = ListenerBundle.load(
                    self.cache_dir, self.cache_dir / spec["model"])
            cls = spec.get("cls", LearnedListenerSpymaster)
            kwargs = spec.get("kwargs") or {"outside_n": spec["outside_n"]}
            self._spymasters[key] = cls(
                **kwargs, k1_tiebreak=self.k1 and not spec.get("no_k1"), model_path=self.cache_dir / spec["model"],
                bundle=self._bundles[spec["model"]], clue_stats=self.clue_stats,
                cache_dir=self.cache_dir)
        return self._spymasters[key]

    @staticmethod
    def describe(board: Board) -> dict:
        return {"seed": board.seed, "words": list(board.words),
                "roles": [ROLE_CODE[c.role] for c in board.cards]}

    def new_board(self) -> dict:
        return self.describe(Board.generate(seed=random.randrange(1, 2**31)))

    def best_clue(self, board, key: str, turn_index: int) -> tuple[str, int, float] | None:
        ctx = TurnContext(board=board, turn_index=turn_index)
        with _LOCK:
            picks = self.spymaster(key).top_clues(ctx, self.sims, 1)
        if not picks:
            return None
        word, number, score = picks[0]
        if SPYMASTERS[key].get("score_is"):
            self._scores[(key, board.seed, frozenset(board.revealed), word, int(number))] = float(score)
        return word, int(number), float(score)

    SHOW = 6
    RISKS = 3

    def explain(self, board, key: str, clue: str, number: int) -> dict:
        """How spymaster `key` reads its own clue: the `number` own words it
        rates highest (its intended targets), and its listener's first-guess
        distribution over the board -- plus the pass, when it has one. From
        `LearnedListenerSpymaster.listen`, i.e. the numbers the clue was
        chosen on."""
        with _LOCK:
            sm = self.spymaster(key)
            # A trained policy has no listener to explain its clue with.
            got = sm.listen(board, clue, self.sims) if hasattr(sm, "listen") else None
        if got is None:
            return {"targets": [], "order": [], "value": None}
        words, roles, s = got["words"], got["roles"], got["scores"]
        # Expected net words at the clue's number, recomputed rather than taken
        # from the search: with the k=1 tiebreak on, a swapped clue's search
        # score is an artificial "best + 1" that only makes it win.
        n_own = sum(1 for r in roles if r is Role.OWN)
        s_out = None if got["outside"] is None else np.array([got["outside"]])
        # The model's own reward (pick_temperature_listener overrides it), so
        # the value shown is the one the clue was chosen on.
        gain, penalty = sm._gain_and_penalty(
            s[None, :n_own], s[None, n_own:], np.array([sm.costs[r] for r in roles[n_own:]]),
            clue_number_cap(n_own, sm.max_number), s_out, words=words)
        value = float((gain - penalty)[0, number - 1])
        kind = "net_words"
        if hasattr(sm, "clue_value"):           # a win-probability model: show the P(win) it chose on
            with _LOCK:
                v = sm.clue_value(board, clue, number, self.sims)
            value, kind = (v, getattr(sm, "value_kind", "win")) if v is not None else (value, kind)
        elif SPYMASTERS[key].get("score_is"):
            v = self._scores.get((key, board.seed, frozenset(board.revealed), clue, int(number)))
            value, kind = (v, SPYMASTERS[key]["score_is"]) if v is not None else (value, kind)
        points = None
        if hasattr(sm, "clue_points_at"):       # stop_listener: the own words the clue points at, unrounded
            with _LOCK:
                points = sm.clue_points_at(board, clue, self.sims)
        entries = [(w, ROLE_CODE[r], float(x)) for w, r, x in zip(words, roles, s)]
        if got["outside"] is not None:
            entries.append(("PASS", "pass", got["outside"]))
        z = np.array([e[2] for e in entries])
        p = np.exp(z - z.max())
        p /= p.sum()
        rate = getattr(sm, "rate", None)            # association models only
        order = sorted(range(len(entries)), key=lambda i: -entries[i][2])
        out = [{"word": entries[i][0], "role": entries[i][1], "p": round(float(p[i]), 4),
                **({"rate": round(float(min(rate(entries[i][2]), 1.0)), 3)} if rate else {})}
               for i in order[:self.SHOW]]
        own = sorted((i for i, e in enumerate(entries) if e[1] == "a"), key=lambda i: -entries[i][2])
        # The words that are not ours the listener rates highest at pick 1: what
        # the clue could be taken for instead (shown with the targets on /compare).
        risks = [{"word": entries[i][0], "role": entries[i][1]} for i in order
                 if entries[i][1] not in ("a", "pass")][:self.RISKS]
        return {"targets": [entries[i][0] for i in own[:number]], "risks": risks, "order": out,
                "value": round(value, 3), "value_kind": kind,
                **({"points_at": round(points, 2)} if points is not None else {})}

    @staticmethod
    def position(seed: int, revealed: list[str], turn: str):
        """The board `seed` with `revealed` turned over, as `turn`'s spymaster
        sees it. Roles are stored from team A's perspective; team B plays the
        same board through the view that swaps OWN and OPPONENT, exactly as
        the two-team game loop does."""
        board = Board.generate(seed=seed)
        on_board = {w.lower(): w for w in board.words}
        board.revealed = {on_board[w.lower()] for w in revealed if w.lower() in on_board}
        return board if turn == "a" else OpponentBoardView(board)

    def explain_clue(self, seed: int, revealed: list[str], turn: str, key: str, clue: str, number: int) -> dict:
        """`explain` for a clue given earlier in a game, at the position it was
        given in: the game page's history after the game ends."""
        return self.explain(self.position(seed, revealed, turn), key, clue, number)

    def clue(self, seed: int, revealed: list[str], turn: str, key: str) -> dict:
        view = self.position(seed, revealed, turn)
        t0 = time.time()
        got = self.best_clue(view, key, turn_index=len(view.revealed))
        if got is None:
            return {"clue": None}
        return {"clue": got[0], "number": got[1], "score": got[2], "spymaster": key,
                "ms": int((time.time() - t0) * 1000)}


MAX_PREREVEAL = 8


def deal_position() -> tuple[int, Board, set[int]]:
    """A random mid-game position: a fresh board with 0-8 non-assassin cards
    pre-revealed, as the listener's training positions were, and at least two
    own words left. Returns (seed, board with `revealed` set, revealed indices)."""
    while True:
        seed = random.randrange(1, 2**31)
        board = Board.generate(seed=seed)
        rng = random.Random(seed ^ 0x5EED)
        roles = [ROLE_CODE[c.role] for c in board.cards]
        pool = [i for i, r in enumerate(roles) if r != "x"]
        pre = set(rng.sample(pool, rng.randint(0, MAX_PREREVEAL)))
        if sum(1 for i, r in enumerate(roles) if r == "a" and i not in pre) >= 2:
            board.revealed = {board.words[i] for i in pre}
            return seed, board, pre


class EvalTurn:
    """One guessing turn under real Codenames rules, with the key held here.

    Pure bookkeeping, no model, so the rules can be tested on their own: a
    wrong pick ends the turn; a right one allows another, up to number + 1;
    finding every remaining own word ends it; Stop ends it at once.
    """

    ENDED_BY = {"b": "opponent", "n": "neutral", "x": "assassin"}

    def __init__(self, roles: list[str], revealed: list[bool], number: int):
        self.roles, self.revealed, self.number = roles, revealed, number
        self.picks: list[int] = []
        self.ended_by: str | None = None

    @property
    def done(self) -> bool:
        return self.ended_by is not None

    def own_left(self) -> int:
        return sum(1 for i, r in enumerate(self.roles)
                   if r == "a" and not self.revealed[i] and i not in self.picks)

    def pick(self, i: int) -> str:
        if self.done:
            raise ValueError("the turn is over")
        if not 0 <= i < len(self.roles) or self.revealed[i] or i in self.picks:
            raise ValueError("that card cannot be picked")
        self.picks.append(i)
        role = self.roles[i]
        if role != "a":
            self.ended_by = self.ENDED_BY[role]
        elif self.own_left() == 0:
            self.ended_by = "cleared"
        elif len(self.picks) >= self.number + 1:
            self.ended_by = "exhausted"
        return role

    def stop(self) -> None:
        if self.done:
            raise ValueError("the turn is over")
        self.ended_by = "stopped"

    def own_found(self) -> int:
        return sum(1 for i in self.picks if self.roles[i] == "a")


class EvalStudy:
    """Serves positions, assigns arms blind, records outcomes.

    **The next position is computed while the current one is played.** A clue
    takes a second or more; after serving a position, a background thread
    deals the next one and computes its clue, so "next" is usually instant.
    What makes this safe for the data:

    - A prefetched position is only a board and a clue. The player, the
      position number and `t_shown` are stamped when it is SERVED, so
      ms_to_first_pick still runs from when the guesser saw it.
    - Positions are served in the order they were computed, and arms are drawn
      at compute time, so block balance holds exactly as before. The only
      position ever lost is the one waiting when the server stops -- never
      served, never seen, so it cannot depend on any outcome.
    - Nothing about a position depends on earlier turns, so computing it
      early changes nothing about what is dealt.
    """

    MAX_PREREVEAL = MAX_PREREVEAL

    def __init__(self, engine: Engine, arms: list[str], log_path: Path, prefetch: bool = True):
        self.engine, self.arms, self.log_path = engine, arms, log_path
        self.prefetch = prefetch
        self.session = secrets.token_hex(4)
        self._lock = threading.Lock()
        self._ready_cv = threading.Condition()
        self._ready: list[dict] = []
        self._computing = False
        self._block: list[str] = []
        self._pending: dict[str, dict] = {}
        self._served = 0
        self._recorded = 0

    def _next_arm(self) -> str:
        with self._lock:
            if not self._block:
                self._block = list(self.arms)
                random.shuffle(self._block)
            return self._block.pop()

    def _compute(self) -> dict:
        """One unserved position: an arm, a board, and that arm's clue."""
        arm = self._next_arm()
        while True:                      # redraw the rare position with nothing to clue
            seed, board, pre = deal_position()
            got = self.engine.best_clue(board, arm, turn_index=len(pre))
            if got is not None:
                return {"arm": arm, "seed": seed, "board": board, "pre": pre, "got": got}

    def warm(self) -> None:
        """Start computing the next position if none is ready or underway."""
        if not self.prefetch:
            return
        with self._ready_cv:
            if self._ready or self._computing:
                return
            self._computing = True
        threading.Thread(target=self._prefetch_one, daemon=True).start()

    def _prefetch_one(self) -> None:
        item = None
        try:
            item = self._compute()
        except Exception as exc:         # next() falls back to computing it itself
            sys.stderr.write(f"  eval prefetch failed: {type(exc).__name__}: {exc}\n")
        with self._ready_cv:
            if item is not None:
                self._ready.append(item)
            self._computing = False
            self._ready_cv.notify_all()

    def next(self, player: str) -> dict:
        with self._ready_cv:
            # A position already underway is nearly done; waiting for it keeps
            # the served order equal to the arm-assignment order.
            while not self._ready and self._computing:
                self._ready_cv.wait()
            item = self._ready.pop(0) if self._ready else None
        if item is None:
            item = self._compute()
        self.warm()
        arm, seed, board, pre = item["arm"], item["seed"], item["board"], item["pre"]
        roles = [ROLE_CODE[c.role] for c in board.cards]
        clue, number, score = item["got"]
        token = secrets.token_hex(8)
        revealed = [i in pre for i in range(len(roles))]
        with self._lock:
            self._served += 1
            self._pending[token] = {
                "turn": EvalTurn(roles, revealed, number), "arm": arm, "seed": seed,
                "clue": clue, "number": number, "score": score, "player": player,
                "words": list(board.words), "t_shown": time.time(), "t_first": None,
                "position": self._served,
            }
        return {"token": token, "words": list(board.words), "clue": clue, "number": number,
                "revealed": [roles[i] if revealed[i] else None for i in range(len(roles))]}

    def _finish(self, token: str, p: dict) -> dict:
        turn: EvalTurn = p["turn"]
        spec = SPYMASTERS[p["arm"]]
        first = turn.roles[turn.picks[0]] == "a" if turn.picks else None
        row = {
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
            "session": self.session, "player": p["player"], "position": p["position"],
            "seed": p["seed"], "arm": p["arm"], "model": spec["model"],
            "outside_n": spec["outside_n"], "k1_tiebreak": self.engine.k1,
            "max_number": (spec.get("kwargs") or {}).get("max_number", 4),
            "clue": p["clue"], "number": p["number"], "score": p["score"],
            "pre_revealed": [w for w, r in zip(p["words"], turn.revealed) if r],
            "own_at_start": sum(1 for i, r in enumerate(turn.roles)
                                if r == "a" and not turn.revealed[i]),
            "picks": [{"word": p["words"][i], "role": turn.roles[i]} for i in turn.picks],
            "ended_by": turn.ended_by, "own_found": turn.own_found(),
            "first_pick_own": first, "stopped": turn.ended_by == "stopped",
            "ms_to_first_pick": (int((p["t_first"] - p["t_shown"]) * 1000)
                                 if p["t_first"] else None),
            "ms_total": int((time.time() - p["t_shown"]) * 1000),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            del self._pending[token]
            self._recorded += 1
            recorded = self._recorded
        return {"done": True, "ended_by": turn.ended_by, "own_found": turn.own_found(),
                "key": turn.roles, "recorded": recorded}

    def _get(self, token: str) -> dict:
        p = self._pending.get(token)
        if p is None:
            raise ValueError("unknown or finished position -- deal a new one")
        return p

    def pick(self, token: str, index: int) -> dict:
        p = self._get(token)
        if p["t_first"] is None:
            p["t_first"] = time.time()
        role = p["turn"].pick(index)
        out = {"role": role}
        if p["turn"].done:
            out.update(self._finish(token, p))
        return out

    def stop(self, token: str) -> dict:
        p = self._get(token)
        p["turn"].stop()
        return self._finish(token, p)

    def recorded(self) -> int:
        """Turns recorded by this server run -- the page's session counter."""
        return self._recorded


class CompareStudy:
    """Side-by-side clues from two spymasters on positions where they differ.

    Positions come from `deal_position`, the same as /eval. Boards on which
    the two give the same clue AND number are dealt past and counted, since
    how often two models agree is itself worth knowing. A different number
    on the same word counts as different.

    Blind mode shuffles the sides and keeps the names and each model's own
    probabilities on the server until the vote: before it the page shows only
    the clue and its intended targets, so the judgement is of the clue and not
    of a model's confidence in it (the association models' pass would give
    them away anyway). Every vote is appended to `log_path`.
    """

    MAX_TRIES = 40
    VOTES = ("left", "right", "tie", "both_bad")
    # Boards dealt ahead for the pair last asked for, while the user reads the
    # current one: a deal is then instant unless the user deals faster than
    # the models clue. A board's clock (t_shown) starts when it is handed
    # out, not when it was computed. Asking for another pair (or blind
    # setting) drops the queue.
    PREFETCH = 3

    def __init__(self, engine: Engine, log_path: Path):
        self.engine, self.log_path = engine, log_path
        self.session = secrets.token_hex(4)
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}
        self._cv = threading.Condition()
        self._want: tuple | None = None
        self._queue: list[dict] = []
        self._worker: threading.Thread | None = None

    def next(self, left: str, right: str, blind: bool) -> dict:
        if left == right:
            raise ValueError("pick two different spymasters")
        key = (left, right, blind)
        with self._cv:
            if self._want != key:
                self._want, self._queue = key, []
            item = self._queue.pop(0) if self._queue else None
            self._cv.notify_all()
            if self._worker is None:
                self._worker = threading.Thread(target=self._prefetch, daemon=True)
                self._worker.start()
        if item is None:
            item = self._deal(*key)
        token = secrets.token_hex(8)
        with self._lock:
            self._pending[token] = {**item["pending"], "t_shown": time.time()}
        with self._cv:
            queued = len(self._queue) if self._want == key else 0
        return {**item["public"], "token": token, "queued": queued}

    def _prefetch(self) -> None:
        """Background: keep PREFETCH boards ready for the pair last asked for."""
        while True:
            with self._cv:
                while self._want is None or len(self._queue) >= self.PREFETCH:
                    self._cv.wait()
                key = self._want
            try:
                item = self._deal(*key)
            except Exception:                   # e.g. no differing clue: wait for the next request
                with self._cv:
                    if self._want == key:
                        self._want = None
                continue
            with self._cv:
                if self._want == key:
                    self._queue.append(item)

    def _deal(self, left: str, right: str, blind: bool) -> dict:
        """One position where the two clue differently, with both explanations."""
        agreed = 0
        for _ in range(self.MAX_TRIES):
            seed, board, pre = deal_position()
            got = {k: self.engine.best_clue(board, k, turn_index=len(pre)) for k in (left, right)}
            if None in got.values():
                continue
            if got[left][:2] != got[right][:2]:
                break
            agreed += 1
        else:
            raise ValueError(f"no differing clue in {self.MAX_TRIES} boards "
                             f"({agreed} gave the same clue)")
        keys = [left, right]
        if blind:
            random.shuffle(keys)
        sides = []
        for k in keys:
            clue, number, _ = got[k]
            sides.append({"key": k, "label": SPYMASTERS[k]["label"], "clue": clue,
                          "number": number,
                          **self.engine.explain(board, k, clue, number)})
        public = ("clue", "number", "targets", "risks")
        return {"pending": {"seed": seed, "words": list(board.words), "blind": blind,
                            "pre": sorted(pre), "sides": sides, "agreed": agreed},
                "public": {"seed": seed, "words": list(board.words),
                           "roles": [ROLE_CODE[c.role] for c in board.cards],
                           "revealed": [i in pre for i in range(len(board.words))],
                           "agreed": agreed, "blind": blind,
                           "sides": [{f: sd[f] for f in public} if blind else sd for sd in sides]}}

    def vote(self, token: str, choice: str, note: str) -> dict:
        if choice not in self.VOTES:
            raise ValueError(f"vote must be one of {self.VOTES}")
        with self._lock:
            p = self._pending.pop(token, None)
        if p is None:
            raise ValueError("unknown or already-voted board -- deal a new one")
        sides = p["sides"]
        winner = {"left": sides[0]["key"], "right": sides[1]["key"]}.get(choice)
        row = {
            "ts": _dt.datetime.now().isoformat(timespec="seconds"), "session": self.session,
            "seed": p["seed"], "pre_revealed": [p["words"][i] for i in p["pre"]],
            "blind": p["blind"], "agreed_before": p["agreed"], "k1_tiebreak": self.engine.k1,
            "left": sides[0]["key"], "right": sides[1]["key"],
            "vote": choice, "winner": winner, "note": note,
            "clues": {sd["key"]: {f: sd.get(f) for f in ("clue", "number", "value", "value_kind", "targets")}
                      for sd in sides},
            "ms": int((time.time() - p["t_shown"]) * 1000),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        return {"sides": sides, "winner": winner}


def make_handler(engine: Engine, study: EvalStudy | None, default_key: str,
                 compare: CompareStudy | None = None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):      # one line per clue, not per asset
            if "/api/clue" in (args[0] if args else ""):
                sys.stderr.write("  %s\n" % (fmt % args))

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in PAGES:
                html = (WEB / PAGES[path]).read_text(encoding="utf-8").replace(
                    "__K1__", "on" if engine.k1 else "off").replace("__NAV__", nav(path))
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/spymasters":
                return self._json({"default": default_key, "spymasters": [
                    {"id": k, **{f: SPYMASTERS[k][f] for f in ("label", "about")}}
                    for k in engine.available()]})
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            try:
                if self.path == "/api/board":
                    seed = req.get("seed")
                    return self._json(engine.describe(Board.generate(seed=int(seed)))
                                      if seed is not None else engine.new_board())
                if self.path == "/api/clue":
                    return self._json(engine.clue(int(req["seed"]),
                                                  list(req.get("revealed") or []),
                                                  req.get("turn", "a"),
                                                  req.get("spymaster") or default_key))
                if self.path == "/api/explain":
                    return self._json(engine.explain_clue(int(req["seed"]), list(req.get("revealed") or []),
                                                          req.get("turn", "a"),
                                                          req.get("spymaster") or default_key,
                                                          str(req["clue"]), int(req["number"])))
                if self.path == "/api/compare/next" and compare is not None:
                    return self._json(compare.next(str(req["left"]), str(req["right"]),
                                                   bool(req.get("blind", True))))
                if self.path == "/api/compare/vote" and compare is not None:
                    return self._json(compare.vote(str(req["token"]), str(req["vote"]),
                                                   str(req.get("note") or "")[:500]))
                if self.path.startswith("/api/eval/"):
                    if study is None:
                        return self._json({"error": "the eval study is not enabled -- "
                                           "its arms need model files this install lacks"}, 400)
                    if self.path == "/api/eval/next":
                        out = study.next(str(req.get("player") or "")[:40])
                        out["recorded"] = study.recorded()
                        return self._json(out)
                    if self.path == "/api/eval/pick":
                        return self._json(study.pick(str(req["token"]), int(req["index"])))
                    if self.path == "/api/eval/stop":
                        return self._json(study.stop(str(req["token"])))
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
            except Exception as exc:                       # surface it in the page
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            self._json({"error": "no such endpoint"}, 404)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-k1", dest="k1", action="store_false",
                    help="disable the k=1 similarity tiebreak")
    # The current best model (docs/versions/floored_overpromise_listener.md);
    # the incumbent is still on the menu as "assoc_profile".
    ap.add_argument("--spymaster", default="floored_overpromise_1", choices=sorted(SPYMASTERS),
                    help="the game page's starting choice")
    ap.add_argument("--eval-arms", default="assoc_profile,profile_temperature",
                    help="comma-separated spymasters the blind study compares. The default "
                         "differs ONLY in the per-pick temperatures -- same booster, same reward.")
    ap.add_argument("--eval-log", type=Path, default=DEFAULT_EVAL_LOG)
    ap.add_argument("--compare-log", type=Path, default=DEFAULT_COMPARE_LOG)
    ap.add_argument("--no-open", dest="open_browser", action="store_false")
    ap.set_defaults(k1=True)
    args = ap.parse_args()

    print("loading the model...", flush=True)
    engine = Engine(k1=args.k1)
    have = engine.available()
    default_key = args.spymaster if args.spymaster in have else have[0]
    arms = [a.strip() for a in args.eval_arms.split(",") if a.strip()]
    missing = [a for a in arms if a not in have]
    study = None if missing or len(arms) < 2 else EvalStudy(engine, arms, args.eval_log)

    url = f"http://{args.host}:{args.port}/"
    print(f"game:  {url}   spymasters available: {', '.join(have)}")
    if study:
        print(f"study: {url}eval   arms {' vs '.join(arms)} (blind)   -> {args.eval_log}")
    else:
        print(f"study: disabled -- arms {arms} need {missing or 'at least two entries'}")
    if study:
        study.warm()                     # the first position is ready when the page opens
    compare = CompareStudy(engine, args.compare_log)
    print(f"compare: {url}compare   -> {args.compare_log}")
    print("Ctrl-C to stop.")
    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(engine, study, default_key, compare))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()

"""Replay the leftover +1 rule on the comfort arena's recorded games (no API
calls: every answer it reads was stored when the games were played).

**The rule** (docs/log.md, "The leftover +1"). The arena's guesser is the
strict-subset guesser (codenames/guessers/subset_first.py): it asks which
words the clue clearly points at (the relatedness "guess" group, without the
number), ranks them with the number, guesses at most the number of them and
stops when they run out. So a turn leaves its clue unfinished in only one
way: a wrong guess cuts it short, and the group's words ranked after the
wrong one are left. The rule, for a turn of team T:
1. Only T's previous clue counts.
2. Its leftovers: the group's words ranked after that turn's wrong guess,
   minus every word revealed since, by either team.
3. Unfound: the previous number, minus the own words that turn found, minus
   the group's own words the opponent revealed since (the guesser sees "your
   word" on one of its group's words and counts it as found). Nothing to do
   at 0.
4. If a leftover remains and unfound > 0, and this turn's guesses were all
   own words with own words still to find, one more guess goes to the top
   remaining leftover.

Each turn is replayed on its own: the game as recorded, plus the one guess
the rule would have added there, without playing on from it.

    python scripts/tools/replay_leftover_plus_one.py
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUITE = "dc9083567bfcf453"                 # holdout_v1_gptoss_subset_guess
GUESSER = "deepinfra:openai/gpt-oss-120b:low:subset=guess"


def offline_guesser():
    """The suite's guesser, unable to call the API: an answer that was not
    stored raises instead of being bought."""
    from codenames.guessers.openai_compat import OpenAICompatGuesser
    from codenames.guessers.registry import build_guesser

    def refuse(self, request):
        raise RuntimeError("replay makes no API calls")

    OpenAICompatGuesser._create = refuse
    return build_guesser(GUESSER)


def replay(db: Path, sims, sm=None) -> tuple[Counter, Counter, list]:
    from codenames.board import Board, OpponentBoardView, Role, load_holdout_wordlist
    from codenames.guessers.base import STOP

    guesser = offline_guesser()
    vocab = load_holdout_wordlist()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute("SELECT seed, turns FROM game_records WHERE suite_id = ?", (SUITE,)).fetchall()
    n, roles_of_plus = Counter(), Counter()
    plus: list[tuple[str, float, float]] = []    # (role, p(word), p(STOP)) per +1
    for seed, turns_json in rows:
        turns = json.loads(turns_json)
        board = Board.generate(seed, vocabulary=vocab)
        n["games"] += 1
        before: list[set] = []                   # revealed words before each turn
        revealed: set = set()
        for t in turns:
            before.append(set(revealed))
            revealed |= {w for w, _ in t["guesses"]}
        last: dict = {}                          # team -> index of its previous clue
        for i, t in enumerate(turns):
            team = t["team"]
            if not t["clue"]:
                continue
            p = last.get(team)
            last[team] = i
            if p is None:
                continue
            n["turns"] += 1
            prev = turns[p]
            if not prev["guesses"] or prev["guesses"][-1][1] == "own":
                continue
            n["previous cut short by a wrong guess"] += 1
            cands = [w for w in board.words if w not in before[p]]
            try:
                group = guesser.rank_candidates(prev["clue"], cands, sims, number=int(prev["number"]))
            except RuntimeError:
                n["previous group not stored"] += 1
                continue
            group = group[:group.index(STOP)]
            j = len(prev["guesses"])
            if group[:j] != [w for w, _ in prev["guesses"]]:
                n["previous group does not match its guesses"] += 1
                continue
            gone = before[i]
            view = board if team == "A" else OpponentBoardView(board)
            own_words = set(view.words_by_role(Role.OWN))
            between = [w for k in range(p + 1, i) for w, _ in turns[k]["guesses"]]
            left = [w for w in group[j:] if w not in gone]
            taken = [w for w in group[j:] if w in between]
            unfound = (int(prev["number"]) - sum(r == "own" for _, r in prev["guesses"])
                       - sum(w in own_words for w in taken))
            if taken:
                n["a leftover taken by the opponent"] += 1
            if not group[j:]:
                continue
            n["the group had words after the wrong guess"] += 1
            if not left or unfound <= 0:
                n["... but none left or nothing unfound"] += 1
                continue
            n["a leftover to go back to"] += 1
            guesses = t["guesses"]
            clean = bool(guesses) and all(r == "own" for _, r in guesses)
            now = {w for w, _ in guesses}
            own_left = own_words - gone - now
            left = [w for w in left if w not in now]
            if not (clean and own_left):
                continue
            if not left:
                n["... the turn guessed the leftover itself"] += 1
                continue
            n["the +1 is taken"] += 1
            role = view.role_of(left[0]).value
            roles_of_plus[role] += 1
            if sm is not None:
                # The STOP listener's pick j+1 for the previous clue, over the
                # words on the board now (after this turn's guesses) and STOP.
                words = [w for w in board.words if w not in gone and w not in now]
                s = sm.stop_scores([prev["clue"]], words, j + 1, sims).get(0)
                if s is not None:
                    e = np.exp(s[j] - s[j].max())
                    e /= e.sum()
                    plus.append((role, float(e[words.index(left[0])]), float(e[-1])))
    return n, roles_of_plus, plus


def main() -> None:
    from codenames.similarity import SimilarityTensor
    from codenames.spymasters.registry import spymaster_spec

    sims = SimilarityTensor.load()
    cls, kw = spymaster_spec("online_vocab_listener")
    n, roles, plus = replay(PROJECT_ROOT / "cache" / "llm_store.db", sims, cls(**kw))
    turns = n["turns"]
    print(f"{n['games']} games, {turns} turns after a team's first")
    for k in ("previous cut short by a wrong guess", "previous group not stored",
              "previous group does not match its guesses",
              "the group had words after the wrong guess", "a leftover taken by the opponent",
              "... but none left or nothing unfound", "a leftover to go back to",
              "... the turn guessed the leftover itself", "the +1 is taken"):
        print(f"  {k:45s} {n[k]:6d}  ({n[k] / turns:6.1%} of turns)")
    total = sum(roles.values())
    if total:
        print("  the +1's word: " + ", ".join(f"{r} {c} ({c / total:.1%})" for r, c in roles.most_common()))
    # The vagueness check: the +1's word by the STOP listener's probability
    # of it at the previous clue's next pick.
    if plus:
        role = np.array([r for r, _, _ in plus])
        pw = np.array([p for _, p, _ in plus])
        ps = np.array([q for _, _, q in plus])
        print(f"  STOP-listener p(word) on {len(plus)} +1s: mean {pw.mean():.3f} against own rate "
              f"{(role == 'own').mean():.3f}; mean p(STOP) {ps.mean():.3f}")
        edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01]
        for lo, hi in zip(edges, edges[1:]):
            m = (pw >= lo) & (pw < hi)
            if m.any():
                print(f"    p(word) {lo:.1f}-{min(hi, 1):.1f}: {m.sum():5d}  mean p {pw[m].mean():.3f}   own "
                      f"{(role[m] == 'own').mean():.3f}  neutral {(role[m] == 'neutral').mean():.3f}  opponent "
                      f"{(role[m] == 'opponent').mean():.3f}  assassin {(role[m] == 'assassin').mean():.3f}")


if __name__ == "__main__":
    main()

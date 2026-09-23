"""Reading head-to-head games back out of the game store, paired by board.

A matchup plays every board twice with the sides swapped, so its games are
matched pairs, not independent draws. The right unit is the board: a board
each side wins once carries no evidence about the spymasters, and a board
swept 2-0 carries the most. The sign test on swept boards is the paired
comparison, and it is the one with power.

Every recorded matchup game carries a label of the form
`<run>|A=<spymaster>,B=<spymaster>` (codenames/two_team_arena.py), which is
the only place the side assignment is stored.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

from codenames.stats import binom_two_sided, wilson


def seat_names(label: str) -> tuple[str, str]:
    """'run|A=x,B=y' -> ('x', 'y')."""
    _, _, sides = label.partition("|")
    a, _, b = sides.partition(",")
    return a.split("=", 1)[1], b.split("=", 1)[1]


def game_outcome(label: str, winner: str | None, turns_json: str) -> tuple[str | None, str | None]:
    """(spymaster that won, spymaster that revealed the assassin or None)."""
    names = seat_names(label)
    won = None if winner is None else names[0] if str(winner).upper().endswith("A") else names[1]
    killer = None
    for t in json.loads(turns_json):
        if t.get("ended_reason") == "assassin":
            killer = names[0] if str(t.get("team", "")).upper() == "A" else names[1]
    return won, killer


@dataclass
class PairedSummary:
    boards: int = 0
    wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    assassin: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    swept: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    split: int = 0

    @property
    def games(self) -> int:
        return 2 * self.boards

    def win_rate(self, name: str) -> float:
        return self.wins.get(name, 0) / self.games if self.games else 0.0

    def wilson(self, name: str) -> tuple[float, float]:
        return wilson(self.wins.get(name, 0), self.games) if self.games else (0.0, 0.0)

    def decisive(self) -> int:
        return sum(self.swept.values())

    def sign_p(self, name: str) -> float:
        n = self.decisive()
        return binom_two_sided(self.swept.get(name, 0), n) if n else float("nan")


def paired_summary(rows: list[tuple[str, int, str | None, str]]) -> PairedSummary:
    """Summarise (label, seed, winner, turns_json) rows, keeping only boards
    played both ways.

    A board can be half-present because the guesser refused to rank on one
    assignment. Counting its surviving half toward win rate while it is absent
    from the sign test would make the two disagree about which games they
    describe -- with the first-move advantage landing entirely on whichever
    side happened to survive.
    """
    boards: dict[int, list[tuple[str | None, str | None]]] = defaultdict(list)
    for label, seed, winner, turns in rows:
        boards[seed].append(game_outcome(label, winner, turns))
    s = PairedSummary()
    for gs in boards.values():
        if len(gs) != 2:
            continue
        s.boards += 1
        for won, killer in gs:
            if won is not None:
                s.wins[won] += 1
            if killer is not None:
                s.assassin[killer] += 1
        if gs[0][0] is not None and gs[0][0] == gs[1][0]:
            s.swept[gs[0][0]] += 1
        else:
            s.split += 1
    return s


@dataclass
class SideTurns:
    clues: int = 0
    number_sum: int = 0
    guesses: int = 0
    own: int = 0

    @property
    def mean_k(self) -> float:
        return self.number_sum / self.clues if self.clues else 0.0

    @property
    def own_per_clue(self) -> float:
        return self.own / self.clues if self.clues else 0.0

    @property
    def own_rate(self) -> float:
        return self.own / self.guesses if self.guesses else 0.0


def side_turn_stats(rows: list[tuple[str, int, str | None, str]]) -> dict[str, SideTurns]:
    """Per-spymaster clue statistics over (label, seed, winner, turns_json)
    rows, each turn credited to whichever spymaster held that side."""
    out: dict[str, SideTurns] = defaultdict(SideTurns)
    for label, _, _, turns in rows:
        names = seat_names(label)
        for t in json.loads(turns):
            st = out[names[0] if str(t.get("team", "")).upper() == "A" else names[1]]
            st.clues += 1
            st.number_sum += t.get("number", 0) or 0
            for _, role in t.get("guesses", []):
                st.guesses += 1
                st.own += role == "own"
    return dict(out)

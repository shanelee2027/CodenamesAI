"""Full games against the learned listener, played the way win_actor_critic
trains on them (docs/log.md, "win_actor_critic: design").

The rules are codenames.game's: `play_two_team_game`'s turn order and win
conditions and `play_turn`'s guessing. tests/test_win_game.py checks the
two agree. Two things differ from the arena, and both are about what the
agent's turns record, not the rules:

- **On the agent's turns the guesser ranks every unrevealed word with no
  number announced** (as `clue_policy.rollout` does). The turn then plays
  the first k of that ranking. One real ranking so fixes the board after
  every k = 1..K_max exactly, not just the k played (`after_boards`).
- **Branching.** The agent may propose several clues for one position.
  Each is ranked for real, and the first one is played. The others are
  recorded as same-board comparisons for the actor. A branch whose clue
  repeats an earlier one is the same cached call, so it costs nothing.

The opponent's turns are the arena's exactly: its clue and number, and the
guesser's ranking for that number, truncated to it.

Positions are identified by (seed, revealed card indices, side to move).
The board is regenerated from the seed on the training vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from codenames.board import Board, OpponentBoardView, Role
from codenames.clue_policy import _training_vocab
from codenames.guessers.base import STOP

TEAMS = ("A", "B")
MAX_HALF_TURNS = 80          # game.DEFAULT_MAX_TURNS per side


def other(team: str) -> str:
    return "B" if team == "A" else "A"


def make_board(seed: int, revealed: frozenset[int] | set[int] = frozenset()) -> Board:
    board = Board.generate(seed=seed, vocabulary=list(_training_vocab()))
    board.revealed = {board.words[i] for i in revealed}
    return board


def view_for(board: Board, team: str):
    """The board as `team` sees it: team A's words are OWN on the board
    itself, team B's through OpponentBoardView."""
    return board if team == "A" else OpponentBoardView(board)


def revealed_ids(board: Board) -> frozenset[int]:
    return frozenset(i for i, w in enumerate(board.words) if w in board.revealed)


def candidates(board: Board) -> list[str]:
    """Unrevealed words in board order, the list play_turn hands the guesser
    (and so part of every cached call's key)."""
    return [w for w in board.words if not board.is_revealed(w)]


def play_attempts(view, attempts: list[str]) -> tuple[list[tuple[str, Role]], str]:
    """Reveal `attempts` in order under play_turn's rules. Returns the guesses
    made (word, role as `view` sees it) and why the turn ended."""
    stopped = STOP in attempts
    if stopped:
        attempts = attempts[: attempts.index(STOP)]
    if not attempts:
        return [], "stopped" if stopped else "no_guesses"
    guesses = []
    for word in attempts:
        role = view.reveal(word)
        guesses.append((word, role))
        if role is Role.ASSASSIN:
            return guesses, "assassin"
        if role is not Role.OWN:
            return guesses, role.value
        if view.remaining(Role.OWN) == 0:
            return guesses, "own_words_complete"
    return guesses, "stopped" if stopped else "exhausted_guesses"


def winner_after(board: Board, mover: str, ended: str) -> str | None:
    """play_two_team_game's end check after `mover`'s turn: the assassin loses
    it for the mover; otherwise a side with no own words left has won (even
    when the other side revealed its last one)."""
    if ended == "assassin":
        return other(mover)
    if board.remaining(Role.OWN) == 0:
        return "A"
    if OpponentBoardView(board).remaining(Role.OWN) == 0:
        return "B"
    return None


def after_boards(seed: int, revealed: frozenset[int], mover: str, ranking: list[str],
                 kmax: int) -> list[tuple[frozenset[int], str | None]]:
    """For k = 1..kmax: the revealed set after `mover` plays the first k words
    of `ranking` from this position, and the winner if that ends the game."""
    out = []
    for k in range(1, kmax + 1):
        board = make_board(seed, revealed)
        _, ended = play_attempts(view_for(board, mover), ranking[:k])
        out.append((revealed_ids(board), winner_after(board, mover, ended)))
    return out


@dataclass
class HalfTurn:
    mover: str
    revealed: list[int]                   # before the turn
    clue: str
    number: int
    guesses: list[str]
    ended: str
    agent: bool
    ranking: list[str] | None = None      # agent turns: the full no-number ranking
    branches: list[dict] = field(default_factory=list)   # agent turns: every proposal, the played one first


@dataclass
class GameRecord:
    seed: int
    agent: str
    turns: list[HalfTurn] = field(default_factory=list)
    winner: str | None = None
    error: str | None = None


AgentMove = Callable[[int, frozenset[int], str], list[dict]]
OpponentMove = Callable[[int, frozenset[int], str], tuple[str, int]]


def play_game(seed: int, agent: str, agent_move: AgentMove, opponent_move: OpponentMove,
              guesser, rank_many: Callable | None = None, sims=None) -> GameRecord:
    """One game on `seed`, the agent playing `agent`. Team A moves first.

    `agent_move(seed, revealed, team)` returns proposals [{"clue", "k", ...}],
    the first to be played. `opponent_move(seed, revealed, team)` returns
    (clue, number). `rank_many(jobs)`, if given, ranks several (clue,
    candidates) at once; otherwise they are ranked one after another.
    `sims` is handed to the guesser as play_turn does; LLM guessers ignore
    it, the synthetic ones the tests use need it.
    Exceptions from either side or the guesser end the game with `error`
    set and no winner. Such games are dropped, like the arena's refusals."""
    rec = GameRecord(seed=seed, agent=agent)
    board = make_board(seed)
    try:
        for half in range(MAX_HALF_TURNS):
            mover = TEAMS[half % 2]
            view = view_for(board, mover)
            before = revealed_ids(board)
            cands = candidates(board)
            if mover == agent:
                props = agent_move(seed, before, mover)
                jobs = [(p["clue"], cands) for p in props]
                rankings = rank_many(jobs) if rank_many else [
                    guesser.rank_candidates(c, w, sims, number=None) for c, w in jobs]
                for p, r in zip(props, rankings):
                    p["ranking"] = r
                clue, number, ranking = props[0]["clue"], props[0]["k"], rankings[0]
                attempts = ranking[:number]
            else:
                clue, number = opponent_move(seed, before, mover)
                ranking = None
                attempts = guesser.rank_candidates(clue, cands, sims, number=number)[:number]
                props = []
            guesses, ended = play_attempts(view, attempts)
            rec.turns.append(HalfTurn(mover=mover, revealed=sorted(before), clue=clue, number=number,
                                      guesses=[w for w, _ in guesses], ended=ended, agent=mover == agent,
                                      ranking=ranking, branches=props))
            winner = winner_after(board, mover, ended)
            if winner is not None:
                rec.winner = winner
                return rec
    except Exception as exc:                                              # noqa: BLE001
        rec.error = f"{type(exc).__name__}: {exc}"
        return rec
    rec.error = "timeout"
    return rec

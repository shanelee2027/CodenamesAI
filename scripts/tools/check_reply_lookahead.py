"""How good is reply_lookahead_listener's cheap model of the incumbent's
reply? (docs/log.md, "reply lookahead")

On real mid-game positions (turn starts of win_prob_listener in the
simulated games, cache/sim_games.db), every probable outcome of every
candidate clue is checked twice:
- **cheap:** the spymaster's own estimate (the incumbent's top clues on the
  current board, re-scored with the revealed words dropped);
- **exact:** the incumbent's full search on the actual after-board, and that
  clue's P(win) from features recomputed on that board.

Reported: how often the two pick the same clue, and the same clue and
number; the error in the incumbent's P(win); the error in each candidate's
correction; how often our final choice differs; how often the lookahead
changes the choice from win_prob_listener's; and the time per move.

    python scripts/tools/check_reply_lookahead.py --positions 60
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def positions(n: int, rng) -> list[tuple[int, str, set, str]]:
    """(board seed, mover team, revealed words, label) for n turn starts where
    win_prob_listener moves, one per game, mid-game only."""
    conn = sqlite3.connect(f"file:{PROJECT_ROOT / 'cache' / 'sim_games.db'}?mode=ro", uri=True)
    rows = conn.execute("SELECT seed, label, turns FROM game_records WHERE label LIKE 'sim_v1|%'").fetchall()
    conn.close()
    out = []
    for i in rng.permutation(len(rows)):
        seed, label, turns = rows[i]
        seat = dict(kv.split("=", 1) for kv in label.split("|", 1)[1].split(","))
        team = next(t for t, name in seat.items() if name.startswith("win_prob_listener"))
        revealed: set = set()
        starts = []
        for t in json.loads(turns):
            if t["team"] == team and revealed:
                starts.append(set(revealed))
            revealed |= {w for w, _ in t["guesses"]}
        if starts:
            out.append((seed, team, starts[rng.integers(len(starts))], label))
        if len(out) == n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", type=int, default=60)
    args = ap.parse_args()

    from codenames.board import Board, OpponentBoardView, clue_number_cap, load_training_wordlist
    from codenames.game import Role
    from codenames.pl_reward import gain_and_penalty
    from codenames.similarity import SimilarityTensor
    from codenames.spymasters.reply_lookahead_listener import (ReplyLookaheadListenerSpymaster, other_side,
                                                                turn_outcomes)
    from codenames.win_value import win_probability

    sims = SimilarityTensor.load()
    sm = ReplyLookaheadListenerSpymaster(shortlist=200, sigma=1.5, max_rarity=10.0, turn_model="frozen",
                                         model_path=PROJECT_ROOT / "cache" / "listener_gbt.txt")
    vocab = load_training_wordlist()
    rng = np.random.default_rng(0)
    same_clue = same_pair = n_out = 0
    win_err, corr_err, choice_same, changed, times = [], [], [], [], []
    for seed, team, revealed, _ in positions(args.positions, rng):
        board = Board.generate(seed, vocabulary=vocab)
        for w in revealed:
            board.reveal(w)
        view = board if team == "A" else OpponentBoardView(board)

        t0 = time.perf_counter()
        best_n, scores, _ = sm._score_all_clues(view, sims)
        times.append(time.perf_counter() - t0)

        # The same steps as _score_all_clues, keeping the pieces.
        own = view.words_by_role(Role.OWN, unrevealed_only=True)
        bad = {r: view.words_by_role(r, unrevealed_only=True) for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN)}
        words = own + bad[Role.NEUTRAL] + bad[Role.OPPONENT] + bad[Role.ASSASSIN]
        roles = [r for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN) for _ in bad[r]]
        n_own, K = len(own), clue_number_cap(len(own), sm.max_number)
        _, base_scores, _ = super(ReplyLookaheadListenerSpymaster, sm)._score_all_clues(view, sims)
        finite = np.flatnonzero(np.isfinite(base_scores))
        top = finite[np.argsort(-base_scores[finite])[: sm.n_top_clues]]
        keep, S = sm._scores(sm, [sims.clue_words[i] for i in top], words, K, sims)
        opp = sm._opponent(view, sims)
        if not keep or opp is None:
            continue
        top = top[keep]
        costs = np.array([sm.costs[r] for r in roles], dtype=np.float64)
        base, _ = sm._clue_values(S[:, :n_own], S[:, n_own:], roles, costs, K, None, words=words)
        b = sum(r == Role.OPPONENT for r in roles)
        vals_cheap, vals_exact = [], []
        exact: dict = {}                      # after-board -> (clue, number, P(win)); outcomes recur across clues
        for flat in np.argsort(-base, axis=None)[: sm.n_candidates].tolist():
            t, m = divmod(flat, base.shape[1])
            outs = sorted(turn_outcomes(S[t], n_own, m + 1).items(), key=lambda x: -x[1])
            c_cheap = c_exact = 0.0
            for (found, end), p in outs[: sm.max_outcomes]:
                if p < sm.min_outcome:
                    break
                left = n_own - len(found)
                role = None if end is None else roles[end - n_own]
                if left == 0 or role == Role.ASSASSIN or (role == Role.OPPONENT and b == 1):
                    continue
                w_v = 1.0 - sm.value.V[b - 1 if role == Role.OPPONENT else b, left]
                removed = frozenset([words[i] for i in found] + ([] if end is None else [words[end]]))
                clue_c, k_c, win_c = sm._reply(opp, removed)
                if removed in exact:
                    clue_x, k_x, win_x = exact[removed]
                    n_out += 1
                    same_clue += clue_c == clue_x
                    same_pair += clue_c == clue_x and k_c == k_x
                    win_err.append(win_c - win_x)
                    c_cheap += p * ((1 - win_c) - w_v)
                    c_exact += p * ((1 - win_x) - w_v)
                    continue
                # Exact: the incumbent's full search on the actual after-board.
                after = Board(cards=board.cards, seed=board.seed, revealed=set(board.revealed))
                for w in removed:
                    after.reveal(w)
                after_view = other_side(after if team == "A" else OpponentBoardView(after))
                bn, sc, _ = sm.opp._score_all_clues(after_view, sims)
                ci = int(np.argmax(sc))
                clue_x, k_x = sims.clue_words[ci], int(bn[ci])
                o_own = after_view.words_by_role(Role.OWN, unrevealed_only=True)
                o_bad = {r: after_view.words_by_role(r, unrevealed_only=True)
                         for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN)}
                o_words = o_own + o_bad[Role.NEUTRAL] + o_bad[Role.OPPONENT] + o_bad[Role.ASSASSIN]
                o_roles = [r for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN) for _ in o_bad[r]]
                Ko = clue_number_cap(len(o_own), sm.opp.max_number)
                _, Sx = sm._scores(sm, [clue_x], o_words, Ko, sims)
                so, sb = Sx[:, :len(o_own)], Sx[:, len(o_own):]
                wp = win_probability(lambda c: gain_and_penalty(so, sb, c, Ko), o_roles, Ko, len(o_own),
                                     sum(r == Role.OPPONENT for r in o_roles), sm.value)
                win_x = float(wp[0, k_x - 1])
                exact[removed] = (clue_x, k_x, win_x)
                n_out += 1
                same_clue += clue_c == clue_x
                same_pair += clue_c == clue_x and k_c == k_x
                win_err.append(win_c - win_x)
                c_cheap += p * ((1 - win_c) - w_v)
                c_exact += p * ((1 - win_x) - w_v)
            corr_err.append(c_cheap - c_exact)
            vals_cheap.append(base[t, m] + c_cheap)
            vals_exact.append(base[t, m] + c_exact)
        if vals_cheap:
            choice_same.append(int(np.argmax(vals_cheap)) == int(np.argmax(vals_exact)))
            changed.append(int(np.argmax(vals_cheap)) != 0)
            print(f"  {len(times)} positions done", flush=True)

    win_err, corr_err = np.array(win_err), np.array(corr_err)
    print(f"{len(times)} positions, {n_out} after-boards checked")
    print(f"  incumbent's reply: same clue {same_clue / n_out:.1%}, same clue and number {same_pair / n_out:.1%}")
    print(f"  its P(win): cheap - exact mean {win_err.mean():+.4f}, mean |error| {np.abs(win_err).mean():.4f}, "
          f"90th percentile {np.percentile(np.abs(win_err), 90):.4f}")
    print(f"  candidate corrections: mean |cheap - exact| {np.abs(corr_err).mean():.4f}")
    print(f"  our choice: cheap and exact agree on {np.mean(choice_same):.1%} of positions; "
          f"the lookahead changes win_prob's choice on {np.mean(changed):.1%}")
    print(f"  time per move {np.mean(times):.2f}s (median {np.median(times):.2f}s)")


if __name__ == "__main__":
    main()

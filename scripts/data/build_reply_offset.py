"""The reply lookahead's offset from V at each score, for
reply_lookahead_listener to subtract (docs/log.md, "reply lookahead").

**Why.** The lookahead values an after-board at 1 - the incumbent's P(win)
with its reply, worked out from the listener and V. V is measured from real
game outcomes at each score; the lookahead is a model's estimate with no
such anchor. On average the two differ, and by an amount that grows with
the own words our turn found (about -0.01 at none to -0.05 at four). Used
raw, that would penalise productive clues for no reason specific to the
board. So the spymaster keeps V's level at each score and takes from the
lookahead only how this after-board differs from its average at that score:

    value(after-board) = (1 - incumbent's P(win)) - offset[b, a]

where a is our words left and b the opponent's after our turn, and offset
is the mean of (lookahead - V) over many after-boards at that score.

**Data.** Turn starts of win_prob_listener in the simulated games (sim_v1,
training boards, never the test boards): every probable outcome (P >= 0.01)
of the best clues at every number. Each cell's mean is shrunk toward 0 with
the weight of 50 outcomes.

With `--spymaster pick_index_lookahead_listener` the outcomes are its turns
under the pick-index booster (its top clues at every number), and the table
goes to cache/reply_offset_pick_index.npz.

    python scripts/data/build_reply_offset.py --positions 300
    python scripts/data/build_reply_offset.py --positions 300 --spymaster pick_index_lookahead_listener
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))

OUT = {"reply_lookahead_listener": PROJECT_ROOT / "cache" / "reply_offset.npz",
       "pick_index_lookahead_listener": PROJECT_ROOT / "cache" / "reply_offset_pick_index.npz"}
SHRINK = 50.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", type=int, default=300)
    ap.add_argument("--spymaster", choices=list(OUT), default="reply_lookahead_listener")
    args = ap.parse_args()

    from check_reply_lookahead import positions

    from codenames.board import Board, OpponentBoardView, clue_number_cap, load_training_wordlist
    from codenames.game import Role
    from codenames.similarity import SimilarityTensor
    from codenames.spymasters.pick_index_lookahead_listener import PickIndexLookaheadListenerSpymaster
    from codenames.spymasters.reply_lookahead_listener import ReplyLookaheadListenerSpymaster, turn_outcomes
    from codenames.spymasters.win_prob_listener import WinProbListenerSpymaster

    sims = SimilarityTensor.load()
    # Raw lookahead values: no offset table yet.
    pick = args.spymaster == "pick_index_lookahead_listener"
    if pick:
        sm = PickIndexLookaheadListenerSpymaster(shortlist=200, sigma=1.5, max_rarity=10.0, reply_offset_path="")
    else:
        sm = ReplyLookaheadListenerSpymaster(shortlist=200, sigma=1.5, max_rarity=10.0,
                                             model_path=PROJECT_ROOT / "cache" / "listener_gbt.txt", reply_offset_path="")
    vocab = load_training_wordlist()
    size = sm.value.V.shape[0]
    total, count = np.zeros((size, size)), np.zeros((size, size))
    for n, (seed, team, revealed, _) in enumerate(positions(args.positions, np.random.default_rng(1))):
        board = Board.generate(seed, vocabulary=vocab)
        for w in revealed:
            board.reveal(w)
        view = board if team == "A" else OpponentBoardView(board)
        own = view.words_by_role(Role.OWN, unrevealed_only=True)
        bad = {r: view.words_by_role(r, unrevealed_only=True) for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN)}
        words = own + bad[Role.NEUTRAL] + bad[Role.OPPONENT] + bad[Role.ASSASSIN]
        roles = [r for r in (Role.NEUTRAL, Role.OPPONENT, Role.ASSASSIN) for _ in bad[r]]
        n_own, K = len(own), clue_number_cap(len(own), sm.max_number)
        _, sc, _ = WinProbListenerSpymaster._score_all_clues(sm, view, sims)
        if pick:
            cand = sm.candidate_turns(view, sims, sc)
            turns = [] if cand is None else [o for _, _, _, o in cand["pairs"]]
        else:
            fin = np.flatnonzero(np.isfinite(sc))
            top = fin[np.argsort(-sc[fin])[: sm.n_top_clues]]
            keep, S = sm._scores(sm, [sims.clue_words[i] for i in top], words, K, sims)
            turns = [turn_outcomes(S[t], n_own, k) for t in range(len(keep)) for k in range(1, K + 1)]
        opp = sm._opponent(view, sims)
        if not turns or opp is None:
            continue
        b = sum(r == Role.OPPONENT for r in roles)
        cache: dict = {}
        for outcomes in turns:
            for (found, end), p in outcomes.items():
                if p < sm.min_outcome:
                    continue
                left = n_own - len(found)
                role = None if end is None else roles[end - n_own]
                if left == 0 or role == Role.ASSASSIN or (role == Role.OPPONENT and b == 1):
                    continue
                b_after = b - 1 if role == Role.OPPONENT else b
                removed = frozenset([words[i] for i in found] + ([] if end is None else [words[end]]))
                if removed not in cache:
                    cache[removed] = sm._reply_win(opp, removed)
                total[b_after, left] += (1 - cache[removed]) - (1 - sm.value.V[b_after, left])
                count[b_after, left] += 1
        if (n + 1) % 25 == 0:
            print(f"  {n + 1} positions", flush=True)
    offset = total / (count + SHRINK)
    np.savez(OUT[args.spymaster], offset=offset, count=count)
    print(f"{int(count.sum())} outcomes -> {OUT[args.spymaster]}")
    print("offset[b, a] (rows b = opponent's words left after our turn, columns a = ours), cells with 100+ outcomes:")
    for bb in range(1, size):
        cells = [f"{offset[bb, a]:+.3f}" if count[bb, a] >= 100 else "   .  " for a in range(1, size)]
        print(f"  b={bb}: " + " ".join(cells))


if __name__ == "__main__":
    main()

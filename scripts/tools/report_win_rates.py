"""Every spymaster tried since 2026-09-25 against the incumbent on one test,
for the notebook: cache/report/win_rates.csv.

**The test.** The gpt-oss suite (configs/eval_suite_gptoss.json: 100 fixed
boards, gpt-oss-120b guessing for both sides, each board played twice with
the seats swapped) against `learned_listener`, the incumbent. Every row is
read from the game store by the challenger's identity (a hash of its
parameters and model files, codenames/eval_suite.py), so a row is exactly
one model.

Per challenger:
- **own boards**: every board it played both ways. Win rate over those
  games, the boards it swept 2-0 against those the incumbent swept, and the
  sign test on them. This is the headline.
- **common boards**: only the boards every challenger played both ways (a
  board drops out when gpt-oss refused one of its rankings in any run), so
  every row is on identical boards. A check that the headline order is not
  an artefact of which boards each row happened to keep.
- **vs reference**: where a row changes one thing relative to another row
  (a booster under a fixed decision rule, a decision rule with a fixed
  booster), the paired difference in win rate from that reference on the
  boards both played, with a 95% bootstrap over boards.

    python scripts/tools/report_win_rates.py
"""

from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from codenames.eval_suite import load_eval_suite
from codenames.headtohead import game_outcome, paired_summary, side_turn_stats

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "cache" / "llm_store.db"
OUT = ROOT / "cache" / "report" / "win_rates.csv"
INCUMBENT = "learned_listener:0e07db5a45d1"

# (group, label, identity, reference identity or None). Settings behind each
# identity are in docs/versions/<spymaster>.md and docs/log.md.
WP_INC, WP_CN, WP_ASSOC, IMIT = ("win_prob_listener:9c304cc348d6", "win_prob_listener:196b0da1ceec",
                                 "win_prob_listener:105dc03b7f48", "imitation_policy:1145ac0c435c")
CHALLENGERS = [
    ("listener, incumbent's objective", "is-a features (isa_listener)", "isa_listener:f52107ffd86b", None),
    ("listener, incumbent's objective", "+ ConceptNet (conceptnet_listener)", "conceptnet_listener:e2a8d0251955", None),
    ("listener, incumbent's objective", "+ free associations (assoc_feature_listener)", "assoc_feature_listener:934b9d580162", None),
    ("listener, incumbent's objective", "association-count target, no pass", "association_listener:b66f09ebe520", None),
    ("listener, incumbent's objective", "association-count target + pass at 5%", "association_listener:fb2c4897ee6f",
     "association_listener:b66f09ebe520"),
    ("listener, win objective", "win_prob, incumbent booster", WP_INC, None),
    ("listener, win objective", "win_prob, is-a booster", "win_prob_listener:1c7c224e8a29", WP_INC),
    ("listener, win objective", "win_prob, ConceptNet booster", WP_CN, WP_INC),
    ("listener, win objective", "win_prob, assoc booster", WP_ASSOC, WP_INC),
    ("listener, win objective", "win_prob, assoc profile booster", "win_prob_listener:41eb6276edd7", WP_INC),
    ("later picks", "per-pick temperatures (pick_temperature_listener)", "pick_temperature_listener:26349423c966", None),
    ("later picks", "within-turn, incumbent's objective (within_turn_listener)", "within_turn_listener:2a49d8135ffd", None),
    ("later picks", "within-turn, fit term only", "within_turn_listener:4e282e694d86", "within_turn_listener:2a49d8135ffd"),
    ("later picks", "win_prob, ConceptNet booster + within-turn", "win_prob_listener:c7327d19e579", WP_CN),
    ("decision rule", "win_prob, assoc booster + turn calibration", "win_prob_listener:3e26714643f4", WP_ASSOC),
    ("decision rule", "win_prob, assoc booster, V2 (policy iteration)", "win_prob_listener:0f6f83e8f214", WP_ASSOC),
    ("decision rule", "board-reading V (board_value_listener)", "board_value_listener:9bd8d05c6da9", WP_INC),
    ("decision rule", "reply lookahead (reply_lookahead_listener)", "reply_lookahead_listener:b307a9fcd833", WP_INC),
    ("clue policy", "imitation of the incumbent (imitation_policy)", IMIT, None),
    ("clue policy", "+ REINFORCE on gpt-oss reward (gptoss_reward_policy)", "gptoss_reward_policy:a042335e52f5", IMIT),
    ("clue policy", "actor-critic on game results (win_actor_critic)", "win_actor_critic:fa4e342ec8d7", None),
]


def board_scores(rows) -> dict[int, int]:
    """Per board played both ways: the challenger's wins out of 2."""
    games = defaultdict(list)
    for label, seed, winner, turns in rows:
        games[seed].append(game_outcome(label, winner, turns)[0])
    return {s: sum(w is not None and w != INCUMBENT for w in g) for s, g in games.items() if len(g) == 2}


def main() -> None:
    suite = load_eval_suite(ROOT / "configs" / "eval_suite_gptoss.json")
    seeds = set(suite.board_seeds)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = {}
    for _, _, cid, _ in CHALLENGERS:
        rows[cid] = [r for r in con.execute(
            "SELECT label, seed, winner, turns FROM game_records WHERE suite_id = ? AND spymaster_id IN (?, ?)",
            (suite.suite_id, f"A={cid},B={INCUMBENT}", f"A={INCUMBENT},B={cid}")) if r[1] in seeds]
        assert rows[cid], f"no games stored for {cid}"
    con.close()

    scores = {cid: board_scores(r) for cid, r in rows.items()}
    common = set.intersection(*(set(s) for s in scores.values()))
    rng = np.random.default_rng(0)
    out = []
    print(f"suite {suite.suite_id}, guesser {suite.guesser}; {len(common)} boards played both ways by every challenger\n")
    for group, label, cid, ref in CHALLENGERS:
        s = paired_summary(rows[cid])
        t = side_turn_stats([r for r in rows[cid] if r[1] in scores[cid]])
        lo, hi = s.wilson(cid)
        sc = scores[cid]
        row = {"group": group, "model": label, "id": cid, "boards": s.boards, "win": s.win_rate(cid),
               "win_lo": lo, "win_hi": hi, "swept_for": s.swept.get(cid, 0), "swept_against": s.swept.get(INCUMBENT, 0),
               "sign_p": s.sign_p(cid), "assassin": s.assassin.get(cid, 0), "assassin_incumbent": s.assassin.get(INCUMBENT, 0),
               "mean_k": t[cid].mean_k, "own_per_clue": t[cid].own_per_clue,
               "common_win": sum(sc[b] for b in common) / (2 * len(common)),
               "reference": "", "vs_ref": np.nan, "vs_lo": np.nan, "vs_hi": np.nan, "vs_boards": 0}
        if ref:
            both = sorted(set(sc) & set(scores[ref]))
            d = np.array([sc[b] - scores[ref][b] for b in both]) / 2
            boot = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            row.update(reference=next(l for _, l, c, _ in CHALLENGERS if c == ref), vs_ref=d.mean(),
                       vs_lo=np.percentile(boot, 2.5), vs_hi=np.percentile(boot, 97.5), vs_boards=len(both))
        out.append(row)
        print(f"  {label:58s} {row['boards']:3d} boards  {row['win']:6.1%} [{lo:.2f}, {hi:.2f}]  "
              f"{row['swept_for']:2d} vs {row['swept_against']:2d} (p = {row['sign_p']:.3f})  "
              f"common {row['common_win']:6.1%}"
              + (f"  vs ref {row['vs_ref']:+.3f} [{row['vs_lo']:+.3f}, {row['vs_hi']:+.3f}]" if ref else ""))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()

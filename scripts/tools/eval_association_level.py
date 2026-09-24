"""Does a listener know how GOOD a clue is, not just which word it favours?
Free: reads the stores only.

The board softmax is blind to one number per clue -- a constant added to
every word's score -- so a listener trained on it alone has no reason to get
that number right. Two held-out measurements of it, neither of which any
listener here trained on unless named:

  association counts   per board word, how many of the teacher's free-
                       association lists for the clue name it (codenames/
                       listener_training.py::association_report): deviance
                       explained, and `level rho` -- whether the predicted
                       total association mass of a board ranks boards the way
                       the observed totals do. That last one is exactly the
                       number the softmax cannot see.
  decoy boards         gpt-oss rankings with random vocabulary words mixed
                       in (scripts/data/collect_decoy_data.py): a listener
                       with a correct per-clue level knows when a random word
                       should beat a vague clue's board words. Validation
                       decoy boards only, so listener_gbt_decoy.txt can be
                       listed too.

Also the board R^2 on the same validation boards, to see what the level costs.
Positions and split are train_listener.py's for `--max-seed 0 --collected
40000`, the incumbent's recipe.

Every listener gets its own best rate link (a, b) on training rows before
being scored, so one never trained on associations is judged on what its
scores know, not on an arbitrary level. `a` for the incumbent is also the
slope to pass to train_listener.py --assoc-slope.

    python scripts/tools/eval_association_level.py \\
        --booster incumbent=cache/listener_gbt_oss_recipe.txt \\
        --booster assoc=cache/listener_gbt_assoc.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from codenames.listener_training import (
    ASSOCIATIONS, CACHE, DB, DEFAULT_MODEL, association_report, association_targets, build_groups,
    decoy_group_mask, load_associations, load_positions, mcfadden_on, split_positions,
)

DECOYS = CACHE / "training_data" / "decoys.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--booster", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--assoc", type=Path, default=ASSOCIATIONS)
    ap.add_argument("--decoys", type=Path, default=DECOYS)
    args = ap.parse_args()

    import lightgbm as lgb

    positions, _ = load_positions(DB, DEFAULT_MODEL, 0, 40000, decoys=args.decoys)
    tr, va = split_positions(positions, 0.25, 0)
    tr = [p for p in tr if not p.get("decoy")]
    va_board = [p for p in va if not p.get("decoy")]
    # The split is by board, and clues repeat across boards while association
    # counts depend only on (clue, word) -- so the trees could score well on a
    # validation clue by recognising it from training. Unseen clues rule that out.
    seen = {p["clue"].lower() for p in tr}
    va_new = [p for p in va_board if p["clue"].lower() not in seen]
    va_dec = [p for p in va if p.get("decoy")]
    lists = load_associations(args.assoc)

    Xtr, _, gtr, _ = build_groups(tr)
    Xva, yva, gva, _ = build_groups(va_board)
    Xd, yd, gd, _ = build_groups(va_dec)
    a_ytr, a_rtr = association_targets(tr, lists)
    a_yva, a_rva = association_targets(va_board, lists)
    Xn, _, gn, _ = build_groups(va_new)
    a_yn, a_rn = association_targets(va_new, lists)
    # First pick only on decoy boards: that is where the signal lives (the
    # collector's docstring), and where the level matters most.
    d_s1 = np.array([j == 0 for p in va_dec for j in range(min(p["k"], p["n"] - 1))])
    print(f"val: {len(va_board)} board positions ({int((a_rva > 0).sum())} rows with association counts), "
          f"{len(va_dec)} decoy boards; {len(va_new)} board positions whose clue is unseen in training\n")
    print(f"{'model':14s} {'board R2':>9s} {'decoy R2':>9s} {'decoy R2 s1':>12s} "
          f"{'assoc D2':>9s} {'level rho':>10s} {'within rho':>11s} {'a':>7s} {'b':>7s}"
          f"   {'unseen: D2':>10s} {'level rho':>10s}")
    for spec in args.booster:
        name, _, path = spec.partition("=")
        b = lgb.Booster(model_file=path)
        s_va = b.predict(Xva, raw_score=True)
        s_d = b.predict(Xd, raw_score=True)
        rep = association_report(b.predict(Xtr, raw_score=True), a_ytr, a_rtr, s_va, a_yva, a_rva,
                                 gva, va_board)
        new = association_report(b.predict(Xtr, raw_score=True), a_ytr, a_rtr, b.predict(Xn, raw_score=True),
                                 a_yn, a_rn, gn, va_new)
        print(f"{name:14s} {mcfadden_on(s_va, gva, yva):9.4f} {mcfadden_on(s_d, gd, yd):9.4f} "
              f"{mcfadden_on(s_d, gd, yd, d_s1):12.4f} {rep['d2']:9.4f} {rep['level_rho']:10.4f} "
              f"{rep['within_rho']:11.4f} {rep['a']:7.3f} {rep['b']:7.3f}"
              f"   {new['d2']:10.4f} {new['level_rho']:10.4f}")


if __name__ == "__main__":
    main()

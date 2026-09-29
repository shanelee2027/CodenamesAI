"""Does board structure predict winning beyond the counts, and does the
critic's board correction pick it up? (docs/log.md, "win_actor_critic:
critic pilot".)

    python scripts/tools/probe_win_critic.py cache/some_critic.pt cache/training_data/win_games_pilot.jsonl

Use a critic fitted WITHOUT --final, so the held-out games (the same seed
hash as fit-critic) were never trained on.

Per held-out position, from the critic's own pair features (win_critic.py):
- **clean_own**: over pairs of the mover's unrevealed words, the best
  "safe margin". A pair's safe margin is how far its best shared clue sits
  above the nearest non-own word (neutral, opponent or assassin), averaged
  over the five embedding sources. Large means the mover has a pair it can
  clue without risk.
- **clean_opp**: the same for the other side's words, against its own
  non-own words.

Reported:
1. Does clean_own / clean_opp predict the real result beyond a counts-only
   model? This is a logistic regression of the result on the counts model's
   logit plus the two, fitted on the training positions and scored on
   held-out ones. If they add nothing, board nuance of this kind barely
   moves win probability against this opponent at this skill level.
2. Does the critic's board correction (its logit minus its counts part)
   rise with clean_own and fall with clean_opp, within positions that have
   the same counts?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from train_win_actor_critic import (  # noqa: E402
    GameStates,
    Positions,
    count_features,
    critic_logits,
    read_games,
    setup,
    split_games,
)

from codenames.win_critic import IU, PER_SOURCE, SOURCES, load_critic  # noqa: E402

EMBEDDINGS = [SOURCES.index(s) for s in ("z_glove", "z_numberbatch", "z_wiki2vec", "z_glove840", "z_fasttext")]
OWN, NEUTRAL, OPP, ASSASSIN = 0, 1, 2, 3


def clean_pairs(edges: np.ndarray, roles: np.ndarray, present: np.ndarray, side: int) -> np.ndarray:
    """Best safe margin over pairs of `side`'s unrevealed words, per position."""
    bad = [r for r in (OWN, NEUTRAL, OPP, ASSASSIN) if r != side]
    i, j = IU
    out = np.full(len(edges), np.nan)
    for n in range(len(edges)):
        ok = (roles[n, i] == side) & (roles[n, j] == side) & present[n, i] & present[n, j]
        if not ok.any():
            continue
        e = edges[n, ok].astype(np.float32).reshape(-1, len(SOURCES), PER_SOURCE)[:, EMBEDDINGS]
        margins = e[:, :, [1 + r for r in bad]].min(-1)          # (pairs, sources)
        out[n] = margins.mean(-1).max()
    return out


def main() -> None:
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("critic", type=Path)
    ap.add_argument("games", nargs="+", type=Path)
    args = ap.parse_args()

    feats, gf, _ = setup(None)
    S = GameStates(Positions(feats, gf))
    S.add(read_games(args.games))
    tr, va = split_games(S)
    z = S.outcomes()
    edges, roles, present = S.P["edges"].numpy(), S.P["roles"].numpy(), S.P["present"].numpy()
    own, opp = clean_pairs(edges, roles, present, OWN), clean_pairs(edges, roles, present, OPP)

    gbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                             verbose=-1).fit(count_features(S, tr), z[tr])
    base = gbm.predict_proba(count_features(S, np.arange(len(z))))[:, 1].clip(1e-6, 1 - 1e-6)
    base_logit = np.log(base / (1 - base))
    X = np.stack([base_logit, np.nan_to_num(own), np.nan_to_num(opp), np.isnan(own), np.isnan(opp)], 1)

    def ll(p, y):
        p = p.clip(1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    lr = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], z[tr])
    print(f"{len(tr)} train / {len(va)} held-out positions")
    print("1. does board structure predict the result beyond counts? (held-out log loss)")
    print(f"   counts model alone            {ll(base[va], z[va]):.4f}")
    print(f"   counts + clean_own + clean_opp {ll(lr.predict_proba(X[va])[:, 1], z[va]):.4f}")
    print(f"   coefficients: counts logit {lr.coef_[0][0]:+.3f}, clean_own {lr.coef_[0][1]:+.3f}, "
          f"clean_opp {lr.coef_[0][2]:+.3f} (per sd of margin)")

    critic, meta = load_critic(args.critic, device="cuda")
    with torch.no_grad():
        full = critic_logits(critic, gf, S.P, va).float().cpu().numpy()
        critic.board = False
        counts_part = critic_logits(critic, gf, S.P, va).float().cpu().numpy()
    corr = full - counts_part
    cf = count_features(S, va)
    key = [tuple(r) for r in cf]
    groups: dict[tuple, list[int]] = {}
    for n, k in enumerate(key):
        groups.setdefault(k, []).append(n)
    xs, ys_own, ys_opp = [], [], []
    for idx in groups.values():
        if len(idx) < 5:
            continue
        idx = np.array(idx)
        o, p = own[va][idx], opp[va][idx]
        m = ~np.isnan(o) & ~np.isnan(p)
        if m.sum() < 5:
            continue
        xs.append(corr[idx][m] - corr[idx][m].mean())
        ys_own.append(o[m] - o[m].mean())
        ys_opp.append(p[m] - p[m].mean())
    c, a, b = np.concatenate(xs), np.concatenate(ys_own), np.concatenate(ys_opp)
    print("2. does the critic's board correction track them, within equal counts?")
    print(f"   board correction: sd {corr.std():.3f} logits, mean {corr.mean():+.3f}")
    print(f"   within-count correlation with clean_own {np.corrcoef(c, a)[0, 1]:+.3f}, "
          f"with clean_opp {np.corrcoef(c, b)[0, 1]:+.3f} (n = {len(c)})")


if __name__ == "__main__":
    main()

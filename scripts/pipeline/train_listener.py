"""Distil a local listener from cached LLM rankings.

Fits the conditional logit described in codenames/listener_features.py: a
LightGBM model scores every candidate word, a softmax over the board turns
those scores into a choice, and the teacher's observed pick is the response.
Each position contributes `k` choice events -- pick 1 from all n words, pick 2
from the remaining n-1, and so on to the announced clue number, which is
exactly as deep as the game ever reads a ranking (measured: 100% of turns read
at most 4 words, 58.5% read one).

**Split is by board seed, never by row.** One board yields many positions, one
position yields k groups, and one group yields n rows, so a random row split
leaks the answer three ways over. The seed is not stored on a cached response,
so it is recovered by matching the candidate set against regenerated boards --
positions that cannot be resolved are dropped rather than guessed, because a
mis-assigned seed silently contaminates the eval set.

Usage:
    python scripts/pipeline/train_listener.py
    python scripts/pipeline/train_listener.py --learning-curve
    python scripts/pipeline/train_listener.py --model claude-sonnet-5+effort=medium   # another teacher
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from codenames.listener_features import FEATURE_NAMES, N_FEATURES
from codenames.listener_training import (
    CACHE, DB, DEFAULT_MODEL, FEATURE_BLOCKS,
    accuracy_on, build_groups, decoy_group_mask, first_step_mask, load_positions, load_soft_labels,
    mcfadden_on, step_weights, train,
)
from codenames.local_lm import DEFAULT_LM


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="which teacher's cached rankings to distil")
    ap.add_argument("--max-seed", type=int, default=60, help="arena board seeds to try when resolving")
    ap.add_argument("--collected", type=int, default=10000,
                    help="generated-board seeds to try (scripts/data/collect_listener_data.py)")
    ap.add_argument("--val-frac", type=float, default=0.25, help="fraction of BOARD SEEDS held out")
    ap.add_argument("--rounds", type=int, default=8000,
                    help="upper bound; early stopping decides. Raised with the "
                         "learning rate drop -- lr=0.01 wants ~2300 trees, and a "
                         "cap that binds would silently look like convergence.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learning-curve", action="store_true")
    ap.add_argument("--refresh-features", action="store_true",
                    help="re-extract features at every step instead of reusing the "
                         "full-board rows; see load_positions")
    ap.add_argument("--blocks", default="all",
                    help="comma-separated feature blocks to keep: "
                         + ",".join(FEATURE_BLOCKS) + " (default: all)")
    ap.add_argument("--decoys", type=Path, default=None,
                    help="jsonl from scripts/data/collect_decoy_data.py. Mixes "
                         "vocabulary words into the candidate list, which is what "
                         "makes the absolute level identifiable -- see decoy_positions")
    ap.add_argument("--decoy-frac", type=float, default=1.0,
                    help="fraction of TRAIN decoy boards to keep, for a learning curve "
                         "over decoy data. Validation and the board half are untouched, "
                         "so the numbers across fractions describe one fixed test set -- "
                         "subsampling the whole decoy file instead shrinks val too, and "
                         "then R2 falls as data GROWS purely because the test got bigger.")
    ap.add_argument("--soft-labels", type=Path, default=None,
                    help="cache/lm_distributions.db: train on a local model's full distribution "
                         "at every step instead of the teacher's one sampled pick (see "
                         "scripts/data/collect_lm_distributions.py). Positions and groups are "
                         "unchanged; only the labels are replaced")
    ap.add_argument("--lm", default=DEFAULT_LM, help="which local model's distributions")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="applied to the soft labels; 0 keeps only their argmax (hard-label control)")
    ap.add_argument("--out", type=Path, default=CACHE / "listener_gbt.txt")
    args = ap.parse_args()

    t0 = time.time()
    soft = load_soft_labels(args.soft_labels, args.lm, args.temperature) if args.soft_labels else None
    positions, dropped = load_positions(args.db, args.model, args.max_seed, args.collected,
                                        refresh_features=args.refresh_features,
                                        decoys=args.decoys, soft_labels=soft)
    print(f"teacher: {args.model}" + (f"   labels: {args.lm} at T={args.temperature:g}" if soft else ""))
    print(f"usable positions: {len(positions)}   dropped: {dropped}")
    if not positions:
        raise SystemExit("no usable positions")

    # Board and decoy seeds are drawn SEPARATELY, so the board split is
    # byte-identical whether or not --decoys is passed. Pooling them would make
    # every board accuracy incomparable with the run it is supposed to be
    # measured against: adding seeds changes the draw for all of them, and a
    # -0.005 difference is then partly a different validation set.
    seeds = sorted({p["seed"] for p in positions if not p.get("decoy")})
    rng = np.random.default_rng(args.seed)
    val_seeds = set(rng.choice(seeds, size=max(1, int(len(seeds) * args.val_frac)), replace=False).tolist())
    dec_seeds = sorted({p["seed"] for p in positions if p.get("decoy")})
    if dec_seeds:
        rng_d = np.random.default_rng(args.seed + 1)
        val_seeds |= set(rng_d.choice(dec_seeds, size=max(1, int(len(dec_seeds) * args.val_frac)),
                                      replace=False).tolist())
        seeds = seeds + dec_seeds
    tr_pos = [p for p in positions if p["seed"] not in val_seeds]
    va_pos = [p for p in positions if p["seed"] in val_seeds]
    if args.decoy_frac < 1.0:
        dec_tr = sorted({p["seed"] for p in tr_pos if p.get("decoy")})
        keep = set(np.random.default_rng(args.seed + 2).choice(
            dec_tr, size=max(1, int(len(dec_tr) * args.decoy_frac)), replace=False).tolist())
        before = len(tr_pos)
        tr_pos = [p for p in tr_pos if not p.get("decoy") or p["seed"] in keep]
        print(f"decoy-frac {args.decoy_frac:.0%}: {before} -> {len(tr_pos)} train positions "
              f"({len(keep)}/{len(dec_tr)} decoy boards); val unchanged")

    print(f"boards: {len(seeds)} total -> {len(seeds)-len(val_seeds)} train / {len(val_seeds)} val")
    print(f"positions: {len(tr_pos)} train / {len(va_pos)} val")

    Xtr, ytr, gtr, _ = build_groups(tr_pos)
    Xva, yva, gva, _ = build_groups(va_pos)
    Xva_full = Xva
    print(f"choice events: {len(gtr)} train / {len(gva)} val   rows: {len(Xtr)} / {len(Xva)}")

    # Feature ablation. Columns are dropped AFTER extraction so every run sees
    # identical rows, groups and split -- the only thing that differs is what
    # the trees are allowed to look at.
    if args.blocks == "all":
        names = list(FEATURE_NAMES)
    else:
        want = [b.strip() for b in args.blocks.split(",") if b.strip()]
        bad = [b for b in want if b not in FEATURE_BLOCKS]
        if bad:
            raise SystemExit(f"unknown block(s) {bad}; known: {list(FEATURE_BLOCKS)}")
        keep = {n for b in want for n in FEATURE_BLOCKS[b]}
        names = [n for n in FEATURE_NAMES if n in keep]
        cols = [FEATURE_NAMES.index(n) for n in names]
        Xtr, Xva = Xtr[:, cols], Xva[:, cols]
        print(f"blocks: {','.join(want)} -> {len(names)}/{N_FEATURES} features")

    # A block selection can exclude the baseline column itself; fall back to
    # the full matrix so an ablation arm still reports a comparable baseline.
    nb = (Xva[:, names.index("z_numberbatch")] if "z_numberbatch" in names
          else Xva_full[:, FEATURE_NAMES.index("z_numberbatch")])
    step1 = first_step_mask(va_pos)
    is_decoy = decoy_group_mask(va_pos)
    base_all = accuracy_on(nb, gva, yva)
    base_s1 = accuracy_on(nb, gva, yva, step1)
    print(f"\nbaseline (numberbatch z alone): all steps {base_all:.4f}   step-1 only {base_s1:.4f}")
    base = base_all

    if args.learning_curve:
        print("\nlearning curve (train boards subsampled; val fixed):")
        tr_seeds = sorted({p["seed"] for p in tr_pos})
        for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
            keep = set(tr_seeds[: max(1, int(len(tr_seeds) * frac))])
            sub = [p for p in tr_pos if p["seed"] in keep]
            Xs, ys, gs, _ = build_groups(sub)
            if args.blocks != "all":
                Xs = Xs[:, cols]
            _, acc = train(Xs, ys, gs, Xva, yva, gva, args.rounds, args.seed, names,
                           step_weights(sub))
            print(f"  {frac:4.0%}  {len(sub):5d} positions, {len(gs):5d} events -> val top-1 {acc:.4f}"
                  f"   (baseline {base:.4f})")
        return

    booster, acc = train(Xtr, ytr, gtr, Xva, yva, gva, args.rounds, args.seed, names,
                         step_weights(tr_pos))
    preds = booster.predict(Xva, raw_score=True)
    m_all, m_s1 = accuracy_on(preds, gva, yva), accuracy_on(preds, gva, yva, step1)
    print(f"model    : all steps {m_all:.4f} ({m_all-base_all:+.4f})   "
          f"step-1 only {m_s1:.4f} ({m_s1-base_s1:+.4f})")

    # Scored apart so the board number stays comparable with every run before
    # decoys existed -- pooling them would silently redefine the headline.
    if is_decoy.any():
        print(f"\nR2  boards {mcfadden_on(preds, gva, yva, ~is_decoy):.4f}"
              f"   decoys {mcfadden_on(preds, gva, yva, is_decoy):.4f}"
              f"   pooled {mcfadden_on(preds, gva, yva):.4f}")
        print(f"    board accuracy {accuracy_on(preds, gva, yva, ~is_decoy):.4f}"
              f"   ({int((~is_decoy).sum())} board events, {int(is_decoy.sum())} decoy)")

        # Is the fitted scale a property of the clue, or of how many decoys were
        # mixed in? `level` is the mean decoy score minus the best board score:
        # deliberately a mean, since the softmax denominator's total decoy mass
        # grows like log(D) whatever the model does, and a logsumexp is further
        # dominated by the largest of D draws. Neither is a defect in the fit,
        # and a statistic carrying either cannot test invariance. `n_decoys` is
        # not a feature, so nothing here forces the columns to agree.
        va_dec = [q for q in va_pos if q.get("decoy") and q["decoy_rows"]]
        if va_dec:
            print("\n  D     n   level (nats)    predicted   observed")
            for d in sorted({q["n_decoys"] for q in va_dec}):
                lv, pr, ob = [], [], []
                for q in (x for x in va_dec if x["n_decoys"] == d):
                    rows = q["x"] if args.blocks == "all" else q["x"][:, cols]
                    sc = booster.predict(rows, raw_score=True)
                    dr = np.asarray(q["decoy_rows"], dtype=int)
                    board = np.setdiff1d(np.arange(q["n"]), dr)
                    lv.append(sc[dr].mean() - sc[board].max())
                    e = np.exp(sc - sc.max())
                    pr.append(e[dr].sum() / e.sum())
                    ob.append(float(q["targets"][0] in set(dr.tolist())))
                print(f" {d:2d} {len(lv):5d} {np.mean(lv):9.2f} +-{np.std(lv)/np.sqrt(len(lv)):.2f}"
                      f" {np.mean(pr):11.3f} {np.mean(ob):10.3f}")

    imp = sorted(zip(names, booster.feature_importance("gain")), key=lambda t: -t[1])
    print("\nfeature importance (gain):")
    for name, g in imp:
        print(f"  {name:22s} {g:12.1f}")
    booster.save_model(str(args.out))
    print(f"\nsaved -> {args.out}   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()

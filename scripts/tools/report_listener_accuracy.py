"""Every listener model on one table, for the notebook (notebooks/
progress_since_0925.ipynb): McFadden R² pooled and by pick (1, 2, 3, 4+) on
the training positions and on the held-out eval set, written to
cache/report/listener_accuracy.csv.

**The eval set** is `held-out words, generated, Sonnet 5.5`: positions made
exactly like the training positions (collect_listener_data.py's clue mix,
reveals and k) on boards of the 150 held-out words, ranked by Sonnet 5.5. It
is the standing measure of a listener model (docs/log.md, "The Sonnet 5.5
generated held-out set"). `held-out words, generated` (the same positions
ranked by gpt-oss, the guesser every model was trained on) is kept as a
second column.

**Training R²** is on `train`, the positions every booster was fitted on
(the incumbent's fit also used val). For the neural correction, the GBT
scores it adds to are the cross-fitted ones it was trained on.

Rows are models, grouped by what they change:
- features: one booster per feature set, every pick scored from the same
  scores over the words left (the frozen Plackett-Luce model);
- later picks: how picks 2+ are modelled, on one base feature set (the
  incumbent recipe's 44, or the assoc booster's 56);
- neural: a learned correction over raw embeddings added to the 44 GBT.

Each row's paired difference from the incumbent is over the same events,
with a 95% bootstrap over boards.

    python scripts/tools/report_listener_accuracy.py
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE = Path(__file__).resolve().parents[2] / "cache"
OUT = CACHE / "report" / "listener_accuracy.csv"
PICK_TEMPERATURES = [1.0, 1.1, 1.3, 1.5]          # pick_temperature_listener's, picks 1, 2, 3, 4+
NETS = ["resid_no_attention", "resid_no_attention_s1", "resid_no_attention_s2"]
REF = "incumbent"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=None, help="load_sets names (default: train, Sonnet 5.5, gpt-oss)")
    args = ap.parse_args()

    import lightgbm as lgb
    import torch

    from codenames.listener_features import FEATURE_NAMES
    from codenames.listener_net import WordVectors, load_listener_net
    from codenames.sequential_listener import SequentialParams
    from eval_listener_accuracy import with_all_columns
    from train_history_listener import ARMS, PairTables, history_groups
    from train_joint_listener import Events
    from train_listener_net import (BASE_OUT, GENERATED, GENERATED_SONNET55, Tensors, event_logp, gbt_scores,
                                    load_sets, pkey)
    from train_pick_index_listener import pick_groups

    vw = WordVectors()
    pt = PairTables()
    booster = lambda f: lgb.Booster(model_file=str(CACHE / f))            # noqa: E731
    seq = lambda f: SequentialParams.from_dict(json.loads((CACHE / f).read_text()))  # noqa: E731
    net_base = BASE_OUT.with_suffix(".gbt.txt")
    cross_fit = pickle.loads(BASE_OUT.read_bytes())

    def models(ps: list[dict], e, train: bool):
        """(group, base, label, nll, null) per model on these positions."""
        ones = np.ones(len(e.y))
        rows = []

        def frozen(f, alpha=None):
            sc = gbt_scores(booster(f), ps)
            flat = np.concatenate([sc[i][keep] for i, keep in zip(e.pos_of, e.keep_of)])
            return sc, flat, e.nll(flat, ones if alpha is None else alpha, 0 * ones)

        for label, f in (("incumbent", "listener_gbt.txt"), ("44 recipe (train only)", "listener_gbt_control44.txt"),
                         ("association counts as a target", "listener_gbt_assoc_w0.3.txt"),
                         ("+ is-a (WordNet)", "listener_gbt_isa.txt"), ("+ ConceptNet", "listener_gbt_conceptnet.txt"),
                         ("+ free associations (assoc)", "listener_gbt_assoc_features.txt"),
                         ("+ association profile", "listener_gbt_assoc_profile.txt")):
            rows.append(("features", "", label, *frozen(f)[2]))

        # Later picks. Every model here keeps pick 1 the base booster's.
        _, _, r = frozen("listener_gbt.txt", 1 / np.asarray(PICK_TEMPERATURES)[e.row_step])
        rows.append(("later picks", "incumbent", "fixed per-pick temperatures (pick_temperature_listener)", *r))
        sc, flat, _ = frozen("listener_gbt_conceptnet.txt")
        rows.append(("later picks", "conceptnet", "within-turn (within_turn_listener)",
                     *e.nll(flat, *e.transform(seq("sequential_listener.json"), e.drop(sc)))))
        for base, f, cols_from in (("44", "control44", "listener_gbt_control44.txt"),
                                   ("assoc", "assoc", "listener_gbt_assoc_features.txt")):
            bf = cols_from
            sc, flat, r = frozen(bf)
            rows.append(("later picks", base, "frozen", *r))
            for label, pf in (("+ per-pick temperatures", f"sequential_listener_{f}_temperature.json"),
                              ("+ within-turn", f"sequential_listener_{f}.json")):
                rows.append(("later picks", base, label, *e.nll(flat, *e.transform(seq(pf), e.drop(sc)))))
            if base == "44":
                for label, jf in (("joint refit + within-turn", "joint44"),
                                  ("joint refit + per-pick temperatures", "joint44_temperature")):
                    jsc, jflat, _ = frozen(f"listener_gbt_{jf}.txt")
                    rows.append(("later picks", base, label,
                                 *e.nll(jflat, *e.transform(seq(f"sequential_listener_{jf}.json"), e.drop(jsc)))))
            cols = [FEATURE_NAMES.index(n) for n in booster(bf).feature_name()]
            Xp, _, gp, _ = pick_groups(ps, cols, None)
            assert len(gp) == len(e.step)
            for label, pf in (("pick index, depth k", f"listener_gbt_pick_index{base}_depthk.txt"),
                              ("pick index, depth 9", f"listener_gbt_pick_index{base}_depth9.txt")):
                if (CACHE / pf).exists():
                    rows.append(("later picks", base, label, *e.nll(booster(pf).predict(Xp, raw_score=True), ones, 0 * ones)))
            for arm, feats in ARMS.items():
                Xh, _, g, _ = history_groups(ps, cols, None, feats, vw, pt)
                assert len(g) == len(e.step)
                hb = booster(f"listener_gbt_history{base}_{arm.replace(' ', '_')}.txt")
                rows.append(("later picks", base, f"pick index + history, {arm}",
                             *e.nll(hb.predict(Xh, raw_score=True), ones, 0 * ones)))

        # The assoc profile booster: no pick-index or history models on its columns.
        sc, flat, r = frozen("listener_gbt_assoc_profile.txt")
        rows.append(("later picks", "profile", "frozen", *r))
        rows.append(("later picks", "profile", "+ per-pick temperatures (profile_temperature_listener)",
                     *e.nll(flat, *e.transform(seq("sequential_listener_assoc_profile_temperature.json"), e.drop(sc)))))

        # Neural correction: the net reads the first 44 columns, and adds to the
        # GBT scores it was trained with (cross-fitted on train, the full GBT elsewhere).
        if torch.cuda.is_available():
            full = gbt_scores(lgb.Booster(model_file=str(net_base)), ps)
            # The net's base is the 44 recipe booster, so "+ neural correction" reads against that row.
            assert all(np.allclose(a, b) for a, b in zip(full, gbt_scores(booster("listener_gbt_control44.txt"), ps)))
            base = cross_fit if train else {pkey(p): s for p, s in zip(ps, full)}
            for name in NETS:
                net, ck = load_listener_net(CACHE / f"listener_net_{name}.pt", vw, "cuda")
                F = len(ck["mean"])
                D = Tensors([{**p, "x": p["x"][:, :F]} for p in ps], vw, ck["mean"], ck["sd"], base, "cuda")
                assert D.dropped == 0 and D.n_events == len(e.step) and (np.minimum(D.ev_step, 3) == e.step).all()
                lp = event_logp(D, net)
                nll = -lp[np.arange(D.n_events), D.ev_tgt.cpu().numpy()]
                seed = name.rpartition("_s")[2] if name[-3:-1] == "_s" else "0"
                rows.append(("neural", "44", f"44 recipe + neural correction (seed {seed})", nll, D.null))
        return rows

    sets = load_sets()
    rng = np.random.default_rng(0)
    out = []
    for s in args.sets or ("train", GENERATED_SONNET55, GENERATED):
        ps = with_all_columns(sets[s])
        e = Events(ps, vw)
        _, bid = np.unique(e.board, return_inverse=True)
        nb = bid.max() + 1
        draws = np.stack([np.bincount(d, minlength=nb) for d in rng.integers(0, nb, (1000, nb))])
        rows = models(ps, e, s == "train")
        ref = next(r for r in rows if r[2] == REF)
        print(f"\n{s}: {len(ps)} positions, {len(e.step)} events, {nb} boards")
        for group, base, label, nll, null in rows:
            r2 = lambda m: 1 - nll[m].sum() / null[m].sum()                       # noqa: E731
            boot = 1 - (draws @ np.bincount(bid, nll, nb)) / (draws @ np.bincount(bid, null, nb))
            d = np.bincount(bid, ref[3] - nll, nb)
            db = (draws @ d) / (draws @ np.bincount(bid, null, nb))
            row = {"set": s, "group": group, "base": base, "model": label, "events": len(nll),
                   "r2": r2(np.ones(len(nll), bool)), "r2_lo": np.percentile(boot, 2.5), "r2_hi": np.percentile(boot, 97.5),
                   **{f"pick_{j + 1}": r2(e.step == j) for j in range(4)},
                   "vs_incumbent": d.sum() / null.sum(), "vs_lo": np.percentile(db, 2.5), "vs_hi": np.percentile(db, 97.5)}
            out.append(row)
            print(f"  {group:12s} {base:10s} {label:58s} {row['r2']:.4f}  "
                  + " ".join(f"{row[f'pick_{j + 1}']:.4f}" for j in range(4)) + f"  {row['vs_incumbent']:+.4f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()

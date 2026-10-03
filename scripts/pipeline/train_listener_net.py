"""Train the neural listener (codenames/listener_net.py) and compare it with
the LightGBM one on the same choice events.

    python scripts/pipeline/train_listener_net.py base          # GBT baseline + cross-fitted logits
    python scripts/pipeline/train_listener_net.py fit --name full
    python scripts/pipeline/train_listener_net.py fit --name feats_only --no-vectors
    python scripts/pipeline/train_listener_net.py fit --name resid --residual

**Data and splits** (all gpt-oss-120b at low effort unless named):
- train / val: the positions the incumbent was fitted on, boards below the
  clean-holdout range, split by board seed exactly as train_listener.py does
  (`split_positions`, 25%, seed 0). Val is used for early stopping.
- `new boards`: the clean holdout of docs/log.md ("The clean holdout"),
  collected boards from seed 1,040,000 on, never trained or selected on.
  Training words only, so this measures new boards, not new words.
- `held-out words`: the frozen eval suites' boards, built only from the 150
  held-out words, with gpt-oss's rankings from the holdout_v1_gptoss games.
  No candidate on them occurs in any training board.
- `held-out words, Sonnet`: the same boards with Sonnet's rankings from
  holdout_v1, the guesser nothing here was trained on.
- `held-out words, generated`: positions generated exactly like `new boards`
  (collect_listener_data.py's clue mix, reveals and k) but on boards of the
  held-out words (`--vocab holdout`). The two sets above are rankings of
  clues a spymaster chose, which are far easier to predict, so only this
  set's R² is comparable with `new boards`: the gap between them is the
  cost of new vocabulary. Added to the cached sets without rebuilding the
  others, so their numbers stay comparable with earlier runs. **This is the
  project's headline listener-accuracy test**, reported by pick.
- `held-out words, generated, Sonnet`: the first 913 of those positions
  (seeds 5,000,000 to 5,001,049), ranked by Sonnet at medium effort, the
  guesser nothing here was trained on (docs/log.md, "A Sonnet copy of the
  generated held-out set"). About 8.5% of its rankings are board-order
  backfill from answers that did not parse (docs/log.md, "Unparsed LLM
  rankings are stored as board order").
- `held-out words, generated, Sonnet 5.5`: all of the generated positions
  ranked by Sonnet 5.5 at medium effort, collected after the guesser was
  fixed to retry unusable answers instead of storing them.

Metric: McFadden R^2 per choice event, pooled and at step 1, as
listener_training.mcfadden_on computes it.

**The GBT baseline is refitted here** on train, early-stopped on val, with the
incumbent's recipe (`listener_training.train`), so both models see the same
rows. `base` also writes cross-fitted GBT logits for the training positions
(4 folds by board seed, each fold scored by a model that never saw it), which
the residual variant needs: an in-sample GBT logit is overconfident on its own
training rows, and a correction learned on top of it would learn to undo
that rather than anything about the board.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

import codenames.listener_training as T
from codenames.listener_net import ListenerNet, WordVectors, save_listener_net

CACHE = T.CACHE
BASE_OUT = CACHE / "training_data" / "listener_net_base.pkl"
SONNET = "claude-sonnet-5+effort=medium"
NEW_BOARDS = 1_040_000
N_MAX = 25
K_MAX = 9


SETS_CACHE = CACHE / "training_data" / "listener_net_sets.pkl"
GENERATED = "held-out words, generated"
GENERATED_BOARDS = 5000          # seeds searched when resolving them (2,200 collected)
GENERATED_SONNET = "held-out words, generated, Sonnet"
GENERATED_SONNET55 = "held-out words, generated, Sonnet 5.5"
SONNET55 = "claude-sonnet-5-5+effort=medium"


def load_sets(reload: bool = False) -> dict[str, list[dict]]:
    """The five sets below, cached after the first load so that parallel fits
    do not each spend a minute resolving boards. `--reload` rebuilds it (after
    new rankings are collected)."""
    if SETS_CACHE.exists() and not reload:
        sets = pickle.loads(SETS_CACHE.read_bytes())
    else:
        sets = _load_sets()
        SETS_CACHE.write_bytes(pickle.dumps(sets, protocol=pickle.HIGHEST_PROTOCOL))
    for name, model in ((GENERATED, T.DEFAULT_MODEL), (GENERATED_SONNET, SONNET), (GENERATED_SONNET55, SONNET55)):
        if name not in sets:
            sets[name] = _load_generated_holdout(model)
            SETS_CACHE.write_bytes(pickle.dumps(sets, protocol=pickle.HIGHEST_PROTOCOL))
    for k, v in sets.items():
        print(f"  {k:24s} {len(v):6d} positions {sum(min(p['k'], p['n'] - 1) for p in v):6d} events")
    return sets


def _load_sets() -> dict[str, list[dict]]:
    pos, dropped = T.load_positions(T.DB, T.DEFAULT_MODEL, 60, 45000, holdout_boards=100)
    print(f"gpt-oss positions {len(pos)}, dropped {dropped}")
    rest = [p for p in pos if p["seed"] < NEW_BOARDS]
    tr, va = T.split_positions(rest, 0.25, 0)
    son, _ = T.load_positions(T.DB, SONNET, 0, 0, holdout_boards=100)
    sets = {
        "train": tr,
        "val": va,
        "new boards": [p for p in pos if NEW_BOARDS <= p["seed"] < T.HOLDOUT_BOARD_BASE],
        "held-out words": [p for p in pos if T.HOLDOUT_BOARD_BASE <= p["seed"] < T.HOLDOUT_COLLECTED_BASE],
        "held-out words, Sonnet": [p for p in son if p["seed"] >= T.HOLDOUT_BOARD_BASE],
    }
    return sets


def _load_generated_holdout(model: str) -> list[dict]:
    pos, dropped = T.load_positions(T.DB, model, 60, 45000, holdout_boards=100,
                                    holdout_collected=GENERATED_BOARDS,
                                    seed_filter=lambda s: s >= T.HOLDOUT_COLLECTED_BASE)
    print(f"held-out words, generated ({model}): {len(pos)} positions, dropped {dropped}")
    return pos


def pkey(p: dict) -> tuple:
    """A position's identity for the stored GBT scores. The row order is part
    of it: rows are shuffled starting from the guesser's own ranking, so two
    guessers given the same clue on the same board share (seed, key) but not
    the order of their rows, and keying without it let one overwrite the
    other's scores."""
    return (p["seed"], p["key"], tuple(p["words"]))


# ---------------------------------------------------------------- GBT base


def gbt_scores(booster, positions: list[dict]) -> list[np.ndarray]:
    """Per position, the GBT's raw score for every row. Features are the
    full-board rows at every step (load_positions without refresh), so one
    vector per position serves all its steps, as in build_groups."""
    from codenames.listener_features import booster_columns

    X = np.vstack([p["x"] for p in positions])
    s = booster.predict(X[:, booster_columns(booster)], raw_score=True)
    return np.split(s, np.cumsum([p["n"] for p in positions])[:-1])


def r2_of_scores(scores: list[np.ndarray], positions: list[dict]) -> tuple[float, float]:
    _, y, g, _ = T.build_groups(positions)
    preds = []
    for s, p in zip(scores, positions):
        taken: list[int] = []
        for j in range(min(p["k"], p["n"] - 1)):
            preds.append(np.delete(s, taken))
            taken.append(p["targets"][j])
    preds = np.concatenate(preds)
    return T.mcfadden_on(preds, g, y), T.mcfadden_on(preds, g, y, T.first_step_mask(positions))


def cmd_base(args) -> None:
    sets = load_sets()
    tr, va = sets["train"], sets["val"]
    Xva, yva, gva, _ = T.build_groups(va)

    def fit(train_pos):
        X, y, g, _ = T.build_groups(train_pos)
        t0 = time.time()
        b, _ = T.train(X, y, g, Xva, yva, gva, 8000, 0, None, T.step_weights(train_pos))
        print(f"    {len(train_pos)} positions -> {b.best_iteration} trees, {time.time() - t0:.0f}s", flush=True)
        return b

    print("full GBT (train, early-stopped on val):")
    full = fit(tr)
    full.save_model(str(BASE_OUT.with_suffix(".gbt.txt")))
    out = {}
    for name, ps in sets.items():
        if name == "train":
            continue
        sc = gbt_scores(full, ps)
        out.update({pkey(p): s for p, s in zip(ps, sc)})
        print(f"  GBT {name:24s} R2 {r2_of_scores(sc, ps)[0]:.4f}  step-1 {r2_of_scores(sc, ps)[1]:.4f}")
    seeds = sorted({p["seed"] for p in tr})
    fold = {s: i % args.folds for i, s in enumerate(np.random.default_rng(7).permutation(seeds))}
    for f in range(args.folds):
        print(f"cross-fit fold {f + 1}/{args.folds}:")
        b = fit([p for p in tr if fold[p["seed"]] != f])
        held = [p for p in tr if fold[p["seed"]] == f]
        out.update({pkey(p): s for p, s in zip(held, gbt_scores(b, held))})
    held = [out[pkey(p)] for p in tr]
    print(f"  cross-fitted GBT on train R2 {r2_of_scores(held, tr)[0]:.4f}")
    BASE_OUT.write_bytes(pickle.dumps(out))
    print(f"saved -> {BASE_OUT}")


# ---------------------------------------------------------------- tensors


class Tensors:
    """Positions as padded GPU tensors, and their choice events.

    A position is (x, clue, cand, present, base); an event is (position, the
    words still available, the target, its step weight). Step j's available
    words are the position's minus the teacher's first j picks, so a later
    pick is scored by re-running the network on the words that remain."""

    def __init__(self, positions: list[dict], vw: WordVectors, mean: np.ndarray, sd: np.ndarray,
                 base: dict | None, device: str):
        keep = []
        for p in positions:
            rows = vw.rows(p["words"])
            clue = vw.index.get(p["clue"].lower())
            if rows is None or clue is None or (base is not None and pkey(p) not in base):
                continue
            keep.append((p, rows, clue))
        self.dropped = len(positions) - len(keep)
        P = len(keep)
        x = np.zeros((P, N_MAX, len(mean)), np.float32)
        cand = np.zeros((P, N_MAX), np.int64)
        present = np.zeros((P, N_MAX), bool)
        b = np.zeros((P, N_MAX), np.float32)
        clue = np.zeros(P, np.int64)
        ev_pos, ev_avail, ev_tgt, ev_w, ev_step = [], [], [], [], []
        for i, (p, rows, c) in enumerate(keep):
            n = p["n"]
            x[i, :n] = np.nan_to_num((np.asarray(p["x"], np.float64) - mean) / sd)
            cand[i, :n], present[i, :n], clue[i] = rows, True, c
            if base is not None:
                b[i, :n] = base[pkey(p)]
            avail = present[i].copy()
            for j in range(min(p["k"], n - 1)):
                t = p["targets"][j]
                ev_pos.append(i); ev_avail.append(avail.copy()); ev_tgt.append(t)
                ev_w.append(T.STEP_DECAY ** j); ev_step.append(j)
                avail[t] = False
        t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
        self.x, self.cand, self.clue, self.base = t(x), t(cand), t(clue), t(b)
        self.ev_pos, self.ev_avail, self.ev_tgt = t(np.array(ev_pos)), t(np.array(ev_avail)), t(np.array(ev_tgt))
        self.ev_w, self.ev_step = t(np.array(ev_w, np.float32)), np.array(ev_step)
        self.null = np.log(np.array(ev_avail).sum(1))
        self.n_events = len(ev_pos)

    def batch(self, idx: torch.Tensor):
        p = self.ev_pos[idx]
        return self.x[p], self.clue[p], self.cand[p], self.ev_avail[idx], self.base[p], self.ev_tgt[idx]


def event_nll(net, D: Tensors, bs: int = 2048) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for a in range(0, D.n_events, bs):
            idx = torch.arange(a, min(a + bs, D.n_events), device=D.x.device)
            x, c, w, av, b, tg = D.batch(idx)
            lp = net(x, c, w, av, b).float().log_softmax(-1)
            out.append(-lp.gather(1, tg[:, None]).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def r2(nll: np.ndarray, D: Tensors, step1: bool = False) -> float:
    m = D.ev_step == 0 if step1 else np.ones(len(nll), bool)
    return 1.0 - nll[m].mean() / D.null[m].mean()


def cmd_fit(args) -> None:
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    sets = load_sets(args.reload)
    vw = WordVectors()
    base = pickle.loads(BASE_OUT.read_bytes()) if args.residual else None
    Xtr = np.vstack([p["x"] for p in sets["train"]]).astype(np.float64)
    mean, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0)
    sd = np.where(sd > 0, sd, 1.0)
    dev = "cuda"
    data = {k: Tensors(v, vw, mean, sd, base, dev) for k, v in sets.items()}
    for k, D in data.items():
        if D.dropped:
            print(f"  {k}: {D.dropped} positions without vectors or base, dropped")
    net = ListenerNet(vw.vecs, len(mean), d=args.d, layers=args.layers, heads=args.heads,
                      sim_heads=args.sim_heads, rank=args.rank, dropout=args.dropout,
                      use_vectors=not args.no_vectors, residual=args.residual).to(dev)
    proj = [p for n, p in net.named_parameters() if n in ("A", "B")]
    other = [p for n, p in net.named_parameters() if n not in ("A", "B")]
    opt = torch.optim.AdamW([{"params": other, "weight_decay": args.wd},
                             {"params": proj, "weight_decay": args.proj_wd}], lr=args.lr)
    tr = data["train"]
    best, best_state, bad = -1e9, None, 0
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(tr.n_events, device=dev)
        tot = 0.0
        for a in range(0, tr.n_events, args.batch):
            idx = perm[a:a + args.batch]
            x, c, w, av, b, tg = tr.batch(idx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = net(x, c, w, av, b).float().log_softmax(-1)
            loss = -(lp.gather(1, tg[:, None]).squeeze(1) * tr.ev_w[idx]).sum() / tr.ev_w[idx].sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        v = r2(event_nll(net, data["val"]), data["val"])
        flag = ""
        if v > best:
            best, bad, flag = v, 0, " *"
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
        print(f"  epoch {ep + 1:2d} train loss {tot / tr.n_events:.4f}  val R2 {v:.4f}{flag}  "
              f"{time.time() - t0:.0f}s", flush=True)
        if bad >= args.patience:
            break
    net.load_state_dict(best_state)
    report = {"name": args.name, "args": vars(args) | {"out": str(args.out)}}
    print(f"\n{args.name}:")
    for k, D in data.items():
        nll = event_nll(net, D)
        report[k] = {"r2": r2(nll, D), "r2_step1": r2(nll, D, True), "events": D.n_events}
        print(f"  {k:24s} R2 {report[k]['r2']:.4f}  step-1 {report[k]['r2_step1']:.4f}  ({D.n_events} events)")
    out = args.out or CACHE / f"listener_net_{args.name}.pt"
    save_listener_net(out, net, mean, sd, report)
    with (CACHE / "training_data" / "listener_net_runs.jsonl").open("a") as fh:
        fh.write(json.dumps(report) + "\n")
    print(f"saved -> {out}")


# ---------------------------------------------------------------- comparison

CAL_BINS = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-9])


def event_logp(D: Tensors, net=None, bs: int = 2048) -> np.ndarray:
    """(events, N_MAX) log-probabilities, -inf off the available words. With
    no net, the GBT's: its per-position scores (D.base) softmaxed over the
    words still available, which is how build_groups scores it."""
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=net is not None):
        for a in range(0, D.n_events, bs):
            idx = torch.arange(a, min(a + bs, D.n_events), device=D.x.device)
            x, c, w, av, b, _ = D.batch(idx)
            s = b.masked_fill(~av, float("-inf")) if net is None else net(x, c, w, av, b)
            out.append(s.float().log_softmax(-1).cpu().numpy())
    return np.concatenate(out)


def metrics(logp: np.ndarray, D: Tensors) -> dict:
    """Every listener is scored by this one function on the same events.

    - r2, r2_step1: McFadden, 1 - mean NLL / mean log(n), as mcfadden_on.
    - acc, acc_step1: top-1, ties credited 1/(number tied), as accuracy_on.
    - nll: mean negative log-likelihood of the pick (nats).
    - ece_all: calibration over every (event, available word) probability,
      against whether that word was the one picked: sum over probability bins
      of (share of predictions in the bin) x |mean prediction - pick rate|.
    - ece_top: the same over each event's top word only (predicted confidence
      against top-1 hit rate).
    - reliability: the ece_all bins, for the table.
    """
    tgt = D.ev_tgt.cpu().numpy()
    avail = D.ev_avail.cpu().numpy()
    E = len(tgt)
    nll = -logp[np.arange(E), tgt]
    s1 = D.ev_step == 0
    top = logp.max(1)
    tied = (logp >= top[:, None] - 1e-9).sum(1)
    hit = np.where(logp[np.arange(E), tgt] >= top - 1e-9, 1.0 / tied, 0.0)
    p = np.exp(logp)
    y = np.zeros_like(p)
    y[np.arange(E), tgt] = 1.0

    def ece(pp, yy):
        b = np.digitize(pp, CAL_BINS) - 1
        rows, tot = [], 0.0
        for i in range(len(CAL_BINS) - 1):
            m = b == i
            if m.any():
                rows.append({"bin": f"{CAL_BINS[i]:.2f}-{min(CAL_BINS[i + 1], 1):.2f}", "n": int(m.sum()),
                             "predicted": float(pp[m].mean()), "observed": float(yy[m].mean())})
                tot += m.mean() * abs(pp[m].mean() - yy[m].mean())
        return float(tot), rows

    ece_all, rel = ece(p[avail], y[avail])
    ece_top, _ = ece(np.exp(top), hit)
    return {"events": E, "r2": float(1 - nll.mean() / D.null.mean()),
            "r2_step1": float(1 - nll[s1].mean() / D.null[s1].mean()),
            "acc": float(hit.mean()), "acc_step1": float(hit[s1].mean()), "nll": float(nll.mean()),
            "ece_all": ece_all, "ece_top": ece_top, "reliability": rel}


def cmd_eval(args) -> None:
    from codenames.listener_net import load_listener_net

    sets = load_sets()
    base = pickle.loads(BASE_OUT.read_bytes())
    vw = WordVectors()
    names = ["GBT"] + [Path(n).stem.removeprefix("listener_net_") for n in args.nets]
    nets = [None] + [load_listener_net(Path(n) if n.endswith(".pt") else CACHE / f"listener_net_{n}.pt",
                                       vw, "cuda") for n in args.nets]
    report = {}
    for set_name in [k for k in sets if k != "train"]:
        # One Tensors per set, standardised with the first net's statistics;
        # every net's own mean/sd is re-applied below, so rows match exactly.
        report[set_name] = {}
        print(f"\n{set_name}")
        print(f"  {'model':28s} {'R2':>7s} {'R2 s1':>7s} {'acc':>7s} {'acc s1':>7s} {'NLL':>7s} "
              f"{'ECE all':>8s} {'ECE top':>8s}")
        for name, nc in zip(names, nets):
            if nc is None:
                F = sets[set_name][0]["x"].shape[1]
                D = Tensors(sets[set_name], vw, np.zeros(F), np.ones(F), base, "cuda")
                m = metrics(event_logp(D), D)
            else:
                net, ck = nc
                D = Tensors(sets[set_name], vw, ck["mean"], ck["sd"], base, "cuda")
                m = metrics(event_logp(D, net), D)
            report[set_name][name] = m
            print(f"  {name:28s} {m['r2']:7.4f} {m['r2_step1']:7.4f} {m['acc']:7.4f} {m['acc_step1']:7.4f} "
                  f"{m['nll']:7.4f} {m['ece_all']:8.4f} {m['ece_top']:8.4f}   ({m['events']} events)")
    if args.reliability:
        for set_name in args.reliability:
            print(f"\nreliability, {set_name} (predicted / observed pick rate, n)")
            for name in names:
                rel = report[set_name][name]["reliability"]
                print(f"  {name:28s} " + "  ".join(f"{r['predicted']:.3f}/{r['observed']:.3f}" for r in rel))
            print(f"  {'n (' + names[0] + ')':28s} " + "  ".join(f"{r['n']:>11d}" for r in report[set_name][names[0]]["reliability"]))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
        print(f"saved -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("base")
    b.add_argument("--folds", type=int, default=4)
    f = sub.add_parser("fit")
    f.add_argument("--name", required=True)
    f.add_argument("--out", type=Path, default=None)
    f.add_argument("--residual", action="store_true")
    f.add_argument("--no-vectors", action="store_true")
    f.add_argument("--d", type=int, default=64)
    f.add_argument("--layers", type=int, default=2)
    f.add_argument("--heads", type=int, default=4)
    f.add_argument("--sim-heads", type=int, default=4)
    f.add_argument("--rank", type=int, default=16)
    f.add_argument("--dropout", type=float, default=0.1)
    f.add_argument("--wd", type=float, default=0.01)
    f.add_argument("--proj-wd", type=float, default=0.01)
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--batch", type=int, default=512)
    f.add_argument("--epochs", type=int, default=60)
    f.add_argument("--patience", type=int, default=6)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--reload", action="store_true", help="rebuild the cached position sets")
    e = sub.add_parser("eval", help="score the GBT and saved nets on the same events, same metrics")
    e.add_argument("nets", nargs="+", help="run names (cache/listener_net_<name>.pt) or .pt paths")
    e.add_argument("--reliability", nargs="*", default=["held-out words", "held-out words, Sonnet"])
    e.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    {"base": cmd_base, "fit": cmd_fit, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()

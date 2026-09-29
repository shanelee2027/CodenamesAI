"""Train the clue policy to imitate the incumbent: cache/policy_imitation.pt.

    python scripts/pipeline/train_imitation_policy.py --epochs 12

Data from scripts/data/collect_imitation_data.py (train and val splits,
disjoint board seeds). Always: cross-entropy of pi(clue | board), over the
legal pool, on the incumbent's pick. Then, by `--head`:

- outcome (the per-turn policies, docs/log.md "gptoss_reward_policy:
  design"): MSE between the outcome head's expected reward for each k and
  the incumbent's, over its 200 shortlisted clues (weight --value-weight);
- k (win_actor_critic, docs/log.md "win_actor_critic: design"):
  cross-entropy of pi(k | board, clue) on the incumbent's number, at the
  incumbent's clue (weight --value-weight). Only the incumbent's picks are
  imitated, never its expected values.

Reported on val each epoch: how often the policy's greedy clue is the
incumbent's, how often it is inside the incumbent's shortlist, the
incumbent-valued regret of the policy's (clue, k) where it can be priced,
agreement on k, and the value MSE. The checkpoint kept is the epoch with the
lowest val cross-entropy.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from codenames.clue_policy import (
    POLICY_MAX_NUMBER,
    DeviceFeatures,
    masked_log_k,
    PolicyFeatures,
    build_net,
    expected_rewards,
    masked_log_policy,
    outcome_log_probs,
    save_policy,
)
from codenames.eval_suite import file_content_hash
from codenames.similarity import DEFAULT_CACHE_DIR

DATA = DEFAULT_CACHE_DIR / "training_data"
OUT = DEFAULT_CACHE_DIR / "policy_imitation.pt"


def load_split(path) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def losses(net, gf: DeviceFeatures, d: dict, idx: np.ndarray, value_weight: float):
    pair, word, roles, present, legal, kmax = gf.batch(d["words"][idx], d["roles"][idx], d["present"][idx],
                                                       max_number=net.max_number)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, outcome, _ = net(pair, word, roles, present)
    logits, outcome = logits.float(), outcome.float()
    logp = masked_log_policy(logits, legal)
    pick = torch.as_tensor(d["pick"][idx], device=gf.device, dtype=torch.long)
    ce = -logp.gather(1, pick[:, None]).squeeze(1)
    if net.head == "k":
        rows = torch.arange(len(idx), device=gf.device)
        logk = masked_log_k(outcome[rows, pick], kmax)                          # (B, M)
        number = torch.as_tensor(d["number"][idx], device=gf.device, dtype=torch.long)
        k_ce = -logk.gather(1, (number - 1)[:, None]).squeeze(1)
        return ce.mean() + value_weight * k_ce.mean(), ce, k_ce.mean(), logp, outcome

    er = expected_rewards(outcome_log_probs(outcome, kmax), kmax)            # (B, P, M)
    short = torch.as_tensor(d["shortlist"][idx], device=gf.device, dtype=torch.long)
    target = torch.as_tensor(d["net"][idx], device=gf.device)[:, :, : net.max_number]
    got = er.gather(1, short.clamp(min=0)[:, :, None].expand(-1, -1, net.max_number))
    ok = torch.isfinite(target) & (short >= 0)[:, :, None] & torch.isfinite(got)
    # where, not a multiply: got is -inf beyond K_max, and -inf * 0 is nan.
    err = torch.where(ok, got - target.nan_to_num(), torch.zeros_like(got))
    mse = (err ** 2).sum() / ok.sum().clamp(min=1)
    return ce.mean() + value_weight * mse, ce, mse, logp, er


@torch.no_grad()
def evaluate(net, gf, d, batch: int, value_weight: float) -> dict:
    net.eval()
    n = len(d["pick"])
    ce_all, mse_all, same, inside, k_same, regrets, k_given_clue = [], [], 0, 0, 0, [], []
    for s in range(0, n, batch):
        idx = np.arange(s, min(n, s + batch))
        _, ce, mse, logp, er = losses(net, gf, d, idx, value_weight)
        ce_all.append(ce.cpu())
        mse_all.append(float(mse) * len(idx))
        g = logp.argmax(1).cpu().numpy()
        rows = torch.arange(len(idx), device=gf.device)
        if net.head == "k":
            # er holds the k logits here: mask k > K_max as training does,
            # then take the argmax.
            kmax = torch.as_tensor(np.minimum(((d["roles"][idx] == 0) & d["present"][idx]).sum(1),
                                              net.max_number), device=gf.device)
            k = masked_log_k(er[rows, torch.as_tensor(g, device=gf.device)], kmax).argmax(-1).cpu().numpy() + 1
            k_at_pick = masked_log_k(er[rows, torch.as_tensor(d["pick"][idx], device=gf.device)],
                                     kmax).argmax(-1).cpu().numpy() + 1
            k_given_clue.extend((k_at_pick == d["number"][idx]).tolist())
        else:
            k = er[rows, torch.as_tensor(g, device=gf.device)].argmax(-1).cpu().numpy() + 1
        for i, row in enumerate(idx):
            same += int(g[i] == d["pick"][row])
            k_same += int(g[i] == d["pick"][row] and k[i] == d["number"][row])
            hit = np.flatnonzero(d["shortlist"][row] == g[i])
            inside += int(hit.size > 0)
            if hit.size and net.head == "outcome":
                v = d["net"][row, hit[0], k[i] - 1]
                regrets.append(float(d["score"][row] - v) if np.isfinite(v) else np.nan)
    net.train()
    out = {"ce": float(torch.cat(ce_all).mean()), "head_loss": sum(mse_all) / n,
           "same_clue": same / n, "same_clue_and_k": k_same / n, "in_shortlist": inside / n}
    if net.head == "k":
        out["k_right_at_incumbent_clue"] = float(np.mean(k_given_clue))
    else:
        out["regret_where_priced"] = float(np.nanmean(regrets)) if regrets else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--trunk", type=int, default=128)
    ap.add_argument("--head", choices=["outcome", "k"], default="outcome",
                    help="what picks the number: see the module docstring")
    ap.add_argument("--max-number", type=int, default=POLICY_MAX_NUMBER,
                    help="highest number the policy may announce (the data must reach it)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--train-file", default=str(DATA / "policy_imitation_train_uncapped.npz"))
    ap.add_argument("--val-file", default=str(DATA / "policy_imitation_val_uncapped.npz"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    feats = PolicyFeatures.load(mmap=False)
    gf = DeviceFeatures(feats, "cuda")
    tr, va = load_split(args.train_file), load_split(args.val_file)
    print(f"train {len(tr['pick'])} positions, val {len(va['pick'])}")
    net = build_net(len(feats.pair_names), len(feats.word_names), width=args.width,
                    hidden=args.hidden, trunk=args.trunk, max_number=args.max_number,
                    head=args.head).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(tr["pick"]) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)

    best, history = math.inf, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(tr["pick"]))
        run = []
        for s in range(0, len(order), args.batch):
            loss, ce, mse, _, _ = losses(net, gf, tr, order[s: s + args.batch], args.value_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            run.append((float(ce.mean()), float(mse)))
        m = evaluate(net, gf, va, 32, args.value_weight)
        tr_ce, tr_mse = np.mean(run, axis=0)
        m.update(epoch=epoch, train_ce=float(tr_ce), train_head_loss=float(tr_mse), minutes=(time.time() - t0) / 60)
        history.append(m)
        print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()), flush=True)
        if m["ce"] < best:
            best = m["ce"]
            save_policy(net, args.out, {
                "stage": "imitation", "epoch": epoch, "val": m, "args": vars(args), "history": history,
                "features": file_content_hash(DEFAULT_CACHE_DIR / "policy_features_aux.npz"),
                "n_train": int(len(tr["pick"])), "n_val": int(len(va["pick"]))})
    print(f"best val ce {best:.4f}; saved {args.out}")


if __name__ == "__main__":
    main()

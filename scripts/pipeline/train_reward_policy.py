"""Fine-tune the clue policy on gpt-oss's reward: cache/policy_gptoss_reward.pt.

    python scripts/pipeline/train_reward_policy.py --iterations 20 --batch 64     # pilot
    python scripts/pipeline/train_reward_policy.py --iterations 300 --resume

A contextual bandit. Each iteration:

1. deals `--batch` fresh training positions (clue_policy.RL_SEEDS, training
   vocabulary, 0-8 cards pre-revealed, both sides);
2. samples one clue per board from the current pi(clue | board) and announces
   the k its outcome head prices best;
3. asks the guesser (default gpt-oss-120b, low effort, temperature 1) to rank
   the unrevealed words, with no number in the prompt, and reads the turn's
   reward for every k off the ranking (clue_policy.rollout);
4. takes one gradient step on
       -(r - V(board)) * log pi(clue)                  REINFORCE, r at the announced k
       + beta * KL(pi || pi_imitation)                 stay near the imitation policy
       + outcome_weight * -log p(observed outcome)     the outcome head, every k at once
   and (V - r)^2 for the baseline V, a small network on the board's role
   counts and two summaries of the current policy (its expected best reward
   under pi, and the best over the pool), which do not depend on the sampled
   clue, so the gradient stays unbiased.

The policy is not changed between sampling and the update, so every step is
on-policy. Every rollout is appended to
cache/training_data/policy_rl_rollouts.jsonl; the LLM answers themselves are
cached in cache/llm_store.db like any other guesser's. A checkpoint is written
every iteration, and `--resume` continues from it (same seed cursor, same
optimizer state).

Every `--val-every` iterations, the greedy policy (argmax pi, argmax k) is
scored on the first `--val-n` validation positions (clue_policy.VAL_SEEDS),
through the same guesser; a repeated clue on a position is a cache hit.
`--max-calls` stops the run when that many new guesser calls have been made.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from codenames.clue_policy import (
    RL_SEEDS,
    VAL_SEEDS,
    DeviceFeatures,
    PolicyFeatures,
    deal_position,
    expected_rewards,
    k_max,
    load_policy,
    masked_log_policy,
    outcome_log_probs,
    positions,
    rollout,
    save_policy,
)
from codenames.eval_suite import file_content_hash
from codenames.guessers.registry import build_guesser
from codenames.similarity import DEFAULT_CACHE_DIR

GUESSER = "deepinfra:openai/gpt-oss-120b:low:temperature=1.0"
INIT = DEFAULT_CACHE_DIR / "policy_imitation.pt"
OUT = DEFAULT_CACHE_DIR / "policy_gptoss_reward.pt"
ROLLOUTS = DEFAULT_CACHE_DIR / "training_data" / "policy_rl_rollouts.jsonl"
CHUNK = 16          # boards per GPU pass; --chunk
PAUSE = 0.0         # seconds idle after each pass; --gpu-pause


def yield_gpu() -> None:
    """Let the GPU go idle between passes, so the work of an iteration is spread
    out instead of arriving as one burst (another program sharing the card
    stutters during the bursts). Changes timing only: the update is the same
    sum over chunks either way."""
    if PAUSE > 0:
        torch.cuda.synchronize()
        time.sleep(PAUSE)


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def board_summary(logp, er, legal, roles, present):
    """Baseline inputs, all independent of the sampled clue."""
    best = er.max(-1).values.masked_fill(~legal, 0.0)                   # (B, P)
    e_pi = (logp.exp() * best).sum(1)
    top = er.max(-1).values.masked_fill(~legal, float("-inf")).max(1).values
    counts = torch.stack([((roles == r) & present).sum(1) for r in range(4)], dim=1).float() / 9.0
    return torch.cat([counts, e_pi[:, None], top[:, None]], dim=1).detach()


def forward(net, gf, boards, idx):
    pair, word, roles, present, legal, kmax = gf.batch(
        [boards[i].words for i in idx], [boards[i].roles for i in idx], [boards[i].present for i in idx])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, outcome, _ = net(pair, word, roles, present)
    logp = masked_log_policy(logits.float(), legal)
    olp = outcome_log_probs(outcome.float(), kmax)
    return logp, olp, expected_rewards(olp, kmax), legal, roles, present


class Counter:
    """Counts the guesser calls that miss every cache (the paid ones) and the
    tokens they use. Cost is priced from the measured $0.000102 per call at
    555 completion tokens (docs/log.md, "gpt-oss-120B as a cheap listener"),
    scaled by the completion tokens actually used."""

    PER_CALL, AT_TOKENS = 0.000102, 555

    def __init__(self, guesser):
        self.n, self.requests, self.prompt_tokens, self.completion_tokens = 0, 0, 0, 0
        inner = guesser._query
        create = guesser.client.chat.completions.create

        def counted(*a, **k):
            self.n += 1
            return inner(*a, **k)

        def metered(*a, **k):
            resp = create(*a, **k)
            self.requests += 1
            if resp.usage:
                self.prompt_tokens += resp.usage.prompt_tokens
                self.completion_tokens += resp.usage.completion_tokens
            return resp

        guesser._query = counted
        guesser.client.chat.completions.create = metered

    @property
    def dollars(self) -> float:
        return self.PER_CALL * self.completion_tokens / self.AT_TOKENS


def run_rollouts(guesser, pool: ThreadPoolExecutor, jobs: list[tuple]) -> list:
    """jobs: (view, clue) -> list of rollout dicts, or the exception raised."""
    def one(job):
        try:
            return rollout(guesser, *job)
        except Exception as exc:                                          # noqa: BLE001
            return exc
    return list(pool.map(one, jobs))


@torch.no_grad()
def greedy_picks(net, gf, feats, views) -> list[tuple[str, int]]:
    out = []
    boards = [feats.encode(v) for v in views]
    for s in range(0, len(boards), CHUNK):
        idx = list(range(s, min(len(boards), s + CHUNK)))
        logp, _, er, *_ = forward(net, gf, boards, idx)
        c = logp.argmax(1)
        k = er[torch.arange(len(idx)), c].argmax(-1) + 1
        out += [(feats.clue_words[int(ci)], int(ki)) for ci, ki in zip(c, k)]
        yield_gpu()
    return out


def validate(net, gf, feats, guesser, pool, val_views, it, log) -> dict:
    picks = greedy_picks(net, gf, feats, [v for _, v in val_views])
    res = run_rollouts(guesser, pool, [(v, c) for (_, v), (c, _) in zip(val_views, picks)])
    rewards, own, fails = [], [], 0
    for (seed, v), (clue, k), r in zip(val_views, picks, res):
        if isinstance(r, Exception):
            fails += 1
            continue
        rewards.append(r["rewards"][k - 1])
        log.write(json.dumps({"split": "val", "it": it, "seed": seed, "clue": clue, "k": k,
                              "outcome": r["outcome"], "rewards": r["rewards"], "ranking": r["ranking"]}) + "\n")
    rewards = np.array(rewards)
    return {"val_reward": float(rewards.mean()), "val_se": float(rewards.std(ddof=1) / math.sqrt(len(rewards))),
            "val_n": int(len(rewards)), "val_fail": fails,
            "val_mean_k": float(np.mean([k for _, k in picks]))}


def main() -> None:
    global CHUNK, PAUSE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--beta", type=float, default=0.05, help="KL weight")
    ap.add_argument("--outcome-weight", type=float, default=1.0)
    ap.add_argument("--guesser", default=GUESSER)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--val-n", type=int, default=300)
    ap.add_argument("--max-calls", type=int, default=40_000)
    ap.add_argument("--init", default=str(INIT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--rollouts", default=str(ROLLOUTS))
    ap.add_argument("--chunk", type=int, default=CHUNK, help="boards per GPU pass")
    ap.add_argument("--gpu-pause", type=float, default=0.0, help="seconds idle after each GPU pass")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    CHUNK, PAUSE = args.chunk, args.gpu_pause

    out = Path(args.out)
    state_path = out.with_suffix(".state.pt")
    torch.manual_seed(args.seed)
    feats = PolicyFeatures.load(mmap=False)
    gf = DeviceFeatures(feats, "cuda")
    ref, ref_meta = load_policy(Path(args.init), device="cuda")
    for p in ref.parameters():
        p.requires_grad_(False)
    net, _ = load_policy(out if args.resume and out.exists() else Path(args.init), device="cuda")
    net.train()
    baseline = Baseline().cuda()
    opt = torch.optim.Adam([{"params": net.parameters(), "lr": args.lr},
                            {"params": baseline.parameters(), "lr": 1e-3}])
    start, cursor, history = 0, RL_SEEDS[0], []
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cuda", weights_only=False)
        opt.load_state_dict(st["opt"])
        baseline.load_state_dict(st["baseline"])
        start, cursor, history = st["it"], st["cursor"], st["history"]
        opt.param_groups[0]["lr"] = args.lr            # --lr wins over the saved state
        print(f"resumed at iteration {start}, seed cursor {cursor}")

    guesser = build_guesser(args.guesser)
    calls = Counter(guesser)
    val_views = positions(VAL_SEEDS, args.val_n)
    Path(args.rollouts).parent.mkdir(parents=True, exist_ok=True)
    log = Path(args.rollouts).open("a")
    pool = ThreadPoolExecutor(args.threads)
    t0 = time.time()

    def checkpoint(it, record):
        save_policy(net, out, {"stage": "gptoss_reward", "iteration": it, "init": str(args.init),
                               "init_hash": file_content_hash(Path(args.init)), "guesser": args.guesser,
                               "args": vars(args), "history": history[-1:] if history else [],
                               "costs": "game (neutral 0.2, opponent 1, assassin 10)"})
        torch.save({"opt": opt.state_dict(), "baseline": baseline.state_dict(), "it": it,
                    "cursor": cursor, "history": history}, state_path)

    if start == 0 and args.val_every:
        v = validate(net, gf, feats, guesser, pool, val_views, 0, log)
        history.append({"it": 0, **v})
        print(f"it 0  {v}", flush=True)

    for it in range(start + 1, args.iterations + 1):
        if calls.n >= args.max_calls:
            print(f"stopping: {calls.n} guesser calls made (--max-calls {args.max_calls})")
            break
        views, seeds = [], []
        while len(views) < args.batch:
            v = deal_position(cursor)
            if v is not None:
                views.append(v)
                seeds.append(cursor)
            cursor += 1
        boards = [feats.encode(v) for v in views]

        # Sample, with the policy as it stands.
        clues, ks, lps, ents, greedy_hit = [], [], [], [], 0
        with torch.no_grad():
            for s in range(0, len(boards), CHUNK):
                idx = list(range(s, min(len(boards), s + CHUNK)))
                logp, _, er, *_ = forward(net, gf, boards, idx)
                c = torch.multinomial(logp.exp(), 1).squeeze(1)
                k = er[torch.arange(len(idx)), c].argmax(-1) + 1
                clues += c.tolist()
                ks += k.tolist()
                lps += logp[torch.arange(len(idx)), c].tolist()
                p = logp.exp()
                ents += (-(p * logp.nan_to_num(neginf=0.0)).sum(1)).tolist()
                greedy_hit += int((c == logp.argmax(1)).sum())
                yield_gpu()

        res = run_rollouts(guesser, pool, [(v, feats.clue_words[c]) for v, c in zip(views, clues)])
        ok = [i for i, r in enumerate(res) if not isinstance(r, Exception)]
        rewards = torch.tensor([res[i]["rewards"][ks[i] - 1] for i in ok], device="cuda")
        cats = torch.tensor([res[i]["outcome"] for i in ok], device="cuda")
        for i, r in enumerate(res):
            if isinstance(r, Exception):
                continue
            log.write(json.dumps({"split": "train", "it": it, "seed": seeds[i], "clue": feats.clue_words[clues[i]],
                                  "k": ks[i], "logp": lps[i], "outcome": r["outcome"], "rewards": r["rewards"],
                                  "ranking": r["ranking"]}) + "\n")
        log.flush()

        # One on-policy step.
        opt.zero_grad(set_to_none=True)
        tot = {"pg": 0.0, "kl": 0.0, "oc": 0.0, "v": 0.0, "adv_sd": []}
        for s in range(0, len(ok), CHUNK):
            sub = ok[s: s + CHUNK]
            rows = torch.arange(len(sub), device="cuda")
            logp, olp, er, legal, roles, present = forward(net, gf, boards, sub)
            with torch.no_grad():
                ref_logp, *_ = forward(ref, gf, boards, sub)
            c = torch.tensor([clues[i] for i in sub], device="cuda")
            r = rewards[s: s + len(sub)]
            V = baseline(board_summary(logp, er, legal, roles, present))
            adv = (r - V).detach()
            pg = -(adv * logp[rows, c])
            # Finite stand-ins off the legal set: where() alone would still
            # backpropagate the nan of (-inf) - (-inf) from the branch it drops.
            lp, rlp = logp.masked_fill(~legal, 0.0), ref_logp.masked_fill(~legal, 0.0)
            kl = (lp.exp() * legal * (lp - rlp)).sum(1)
            oc = -olp[rows, c, cats[s: s + len(sub)]]
            loss = (pg + args.beta * kl + args.outcome_weight * oc + (V - r) ** 2).sum() / len(ok)
            loss.backward()
            tot["pg"] += float(pg.sum().detach())
            tot["kl"] += float(kl.sum().detach())
            tot["oc"] += float(oc.sum().detach())
            tot["v"] += float(((V - r) ** 2).sum().detach())
            tot["adv_sd"] += adv.tolist()
            yield_gpu()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        n = max(len(ok), 1)
        rec = {"it": it, "n": len(ok), "fail": len(res) - len(ok), "reward": float(rewards.mean()),
               "mean_k": float(np.mean([ks[i] for i in ok])), "kl": tot["kl"] / n, "outcome_nll": tot["oc"] / n,
               "baseline_mse": tot["v"] / n, "adv_sd": float(np.std(tot["adv_sd"])), "entropy": float(np.mean(ents)),
               "greedy_share": greedy_hit / len(boards), "calls": calls.n, "dollars": calls.dollars,
               "minutes": (time.time() - t0) / 60}
        if args.val_every and it % args.val_every == 0:
            rec.update(validate(net, gf, feats, guesser, pool, val_views, it, log))
            log.flush()
        history.append(rec)
        print(" ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
        checkpoint(it, rec)

    pool.shutdown()
    log.close()
    print(f"done: {calls.n} guesser calls ({calls.requests} requests with retries), "
          f"{calls.prompt_tokens:,} prompt / {calls.completion_tokens:,} completion tokens, "
          f"~${calls.dollars:.2f}, in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

"""win_actor_critic: play real games against the learned listener, fit the
critic, and train the actor on the change in win probability (docs/log.md,
"win_actor_critic: design").

    # games with a fixed policy (the critic pilot, and the final comparison)
    python scripts/pipeline/train_win_actor_critic.py play --policy cache/win_policy_init.pt \\
        --seeds train --n 1200 --out cache/training_data/win_games_pilot.jsonl --budget 1.2
    python scripts/pipeline/train_win_actor_critic.py play --policy cache/win_policy_init.pt \\
        --seeds val --n 300 --greedy --out cache/training_data/win_games_val_init.jsonl --budget 0.4

    # the critic, from any set of game files
    python scripts/pipeline/train_win_actor_critic.py fit-critic cache/training_data/win_games_pilot.jsonl

    # actor-critic, resumable
    python scripts/pipeline/train_win_actor_critic.py train --budget 5

The only reward anywhere is the game's result. Nothing reads the learned
listener's probabilities: it is only the opponent. The guesser for both
sides is gpt-oss-120b at temperature 1.

Seeds: games use TRAIN_SEEDS in order (agent is team A on even seeds, B on
odd), and a run resumes where the last one stopped. VAL_SEEDS are only
played greedily, for the before/after comparison, and never trained on.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from codenames.board import Role
from codenames.clue_policy import (
    ROLE_ID,
    BudgetExhausted,
    DeviceFeatures,
    GuesserMeter,
    PolicyFeatures,
    load_policy,
    masked_log_k,
    masked_log_policy,
    save_policy,
)
from codenames.eval_suite import file_content_hash
from codenames.guessers.registry import build_guesser
from codenames.similarity import DEFAULT_CACHE_DIR
from codenames.win_critic import (
    CriticFeatures,
    build_critic,
    lambda_returns,
    load_critic,
    relative_roles,
    save_critic,
)
from codenames.win_game import GameRecord, after_boards, make_board, other, play_game, view_for

GUESSER = "deepinfra:openai/gpt-oss-120b:low:temperature=1.0"
DATA = DEFAULT_CACHE_DIR / "training_data"
TRAIN_SEEDS = (22_000_000, 23_000_000)
VAL_SEEDS = (29_100_000, 29_200_000)
CRITIC = DEFAULT_CACHE_DIR / "win_critic.pt"
POLICY = DEFAULT_CACHE_DIR / "win_policy.pt"
TRAIN_GAMES = DATA / "win_games_train.jsonl"
_opp: dict = {}


# ---------------------------------------------------------------------------
# The opponent, in worker processes


def _opp_init() -> None:
    from codenames.similarity import SimilarityTensor
    from codenames.spymasters.registry import spymaster_spec

    torch.set_num_threads(1)
    cls, kwargs = spymaster_spec("learned_listener")
    _opp["sm"] = cls(**kwargs)
    _opp["sims"] = SimilarityTensor.load()


def _opp_move(job: tuple[int, tuple[int, ...], str]) -> tuple[str, int]:
    from codenames.spymasters.base import TurnContext

    seed, revealed, team = job
    board = make_board(seed, frozenset(revealed))
    ctx = TurnContext(board=view_for(board, team), turn_index=len(board.revealed))
    return _opp["sm"].give_clue(ctx, _opp["sims"])


# ---------------------------------------------------------------------------
# The agent


class Actor:
    """Proposals for one position from a policy network on the GPU."""

    def __init__(self, net, feats: PolicyFeatures, gf: DeviceFeatures, greedy: bool, branches: int):
        self.net, self.feats, self.gf = net, feats, gf
        self.greedy, self.branches = greedy, branches
        self.lock = threading.Lock()

    def encode(self, seed: int, revealed, team: str):
        return self.feats.encode(view_for(make_board(seed, frozenset(revealed)), team))

    @torch.no_grad()
    def __call__(self, seed: int, revealed, team: str) -> list[dict]:
        b = self.encode(seed, revealed, team)
        with self.lock:
            pair, word, roles, present, legal, kmax = self.gf.batch(
                [b.words], [b.roles], [b.present], max_number=self.net.max_number)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, klog, _ = self.net(pair, word, roles, present)
            logp = masked_log_policy(logits.float(), legal)[0]
            if self.greedy:
                clues = [int(logp.argmax())]
            else:
                clues = torch.multinomial(logp.exp(), self.branches, replacement=True).tolist()
            out = []
            for c in clues:
                logk = masked_log_k(klog[0, c].float()[None], kmax)[0]
                k = int(logk.argmax()) if self.greedy else int(torch.multinomial(logk.exp(), 1))
                out.append({"clue": self.feats.clue_words[c], "pool": c, "k": k + 1,
                            "logp": float(logp[c]), "kmax": int(kmax[0])})
            return out


class IncumbentAgent:
    """The learned listener playing the agent's side: its own clue and number,
    but ranked the way the agent's turns are, with no number announced. Against
    the listener itself this measures what that ranking costs a side
    (docs/log.md, "win_actor_critic: critic pilot")."""

    def __init__(self, opp_pool):
        self.pool = opp_pool

    def __call__(self, seed: int, revealed, team: str) -> list[dict]:
        clue, number = self.pool.submit(_opp_move, (seed, tuple(sorted(revealed)), team)).result()
        return [{"clue": clue, "k": int(number), "pool": -1, "kmax": int(number), "logp": 0.0}]


def play_games(seeds: list[int], actor: Actor, guesser, opp_pool, threads: int, log_path: Path,
               extra: dict | None = None) -> list[GameRecord]:
    """Play one game per seed, concurrently; append each finished game to
    `log_path` as it ends. Returns them in seed order."""
    rank_pool = ThreadPoolExecutor(threads * 2)

    def rank_many(jobs):
        return list(rank_pool.map(lambda j: guesser.rank_candidates(j[0], j[1], None, number=None), jobs))

    def opponent(seed, revealed, team):
        return opp_pool.submit(_opp_move, (seed, tuple(sorted(revealed)), team)).result()

    lock = threading.Lock()
    games: dict[int, GameRecord] = {}
    with log_path.open("a") as log, ThreadPoolExecutor(threads) as pool:
        def one(seed):
            g = play_game(seed, "A" if seed % 2 == 0 else "B", actor, opponent, guesser, rank_many)
            with lock:
                log.write(json.dumps({**dataclasses.asdict(g), **(extra or {})}) + "\n")
                log.flush()
                games[seed] = g
        list(pool.map(one, seeds))
    rank_pool.shutdown()
    return [games[s] for s in seeds]


def read_games(paths) -> list[dict]:
    out = []
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p) as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


def next_seeds(n: int, seed_range: tuple[int, int], used: set[int]) -> list[int]:
    """The first `n` seeds of the range not in `used`."""
    out, s = [], seed_range[0]
    while len(out) < n:
        if s >= seed_range[1]:
            raise ValueError("seed range exhausted")
        if s not in used:
            out.append(s)
        s += 1
    return out


# ---------------------------------------------------------------------------
# Positions as critic inputs


class Positions:
    """Critic inputs for positions (seed, revealed, side to move, agent
    flag), with the pair features computed on the GPU and kept on the CPU."""

    def __init__(self, feats: PolicyFeatures, gf: DeviceFeatures):
        self.feats, self.gf = feats, gf
        self.cf = CriticFeatures(feats, gf)
        self._boards: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def board(self, seed: int):
        if seed not in self._boards:
            b = self.feats.encode(make_board(seed))            # team A's view, nothing revealed
            self._boards[seed] = (b.words, b.roles)
        return self._boards[seed]

    def build(self, rows: list[tuple[int, frozenset[int] | list[int], str, bool]], batch: int = 64) -> dict:
        words, roles, present, agent = [], [], [], []
        for seed, revealed, mover, is_agent in rows:
            w, r = self.board(seed)
            pres = np.ones(25, dtype=bool)
            pres[list(revealed)] = False
            words.append(w)
            roles.append(relative_roles(r, mover))
            present.append(pres)
            agent.append(is_agent)
        words, roles, present = np.stack(words), np.stack(roles), np.stack(present)
        edges = torch.cat([self.cf.edges(words[s: s + batch], roles[s: s + batch], present[s: s + batch]).cpu()
                           for s in range(0, len(rows), batch)])
        return {"words": torch.as_tensor(words), "roles": torch.as_tensor(roles),
                "present": torch.as_tensor(present), "agent": torch.as_tensor(np.array(agent)), "edges": edges}


def cat_positions(a: dict | None, b: dict) -> dict:
    return b if a is None else {k: torch.cat([a[k], b[k]]) for k in b}


def critic_logits(critic, gf: DeviceFeatures, P: dict, idx=None, batch: int = 512, grad: bool = False):
    idx = torch.arange(len(P["words"])) if idx is None else torch.as_tensor(idx)
    outs = []
    for s in range(0, len(idx), batch):
        i = idx[s: s + batch]
        w = P["words"][i].to(gf.device)
        with torch.set_grad_enabled(grad):
            outs.append(critic(gf.word[w], P["roles"][i].to(gf.device), P["present"][i].to(gf.device),
                               P["edges"][i].to(gf.device), P["agent"][i].to(gf.device)))
    return torch.cat(outs)


class GameStates:
    """Every finished game's positions (the critic's training data), and
    where each game's run of positions starts."""

    def __init__(self, positions: Positions):
        self.pos = positions
        self.P: dict | None = None
        self.games: list[dict] = []              # {seed, start, T, last_won, agent_won}

    def add(self, games: list[dict]) -> None:
        rows = []
        n = 0 if self.P is None else len(self.P["words"])
        for g in games:
            if g.get("error") or g.get("winner") is None or not g["turns"]:
                continue
            T = len(g["turns"])
            for t in g["turns"]:
                rows.append((g["seed"], t["revealed"], t["mover"], t["mover"] == g["agent"]))
            self.games.append({"seed": g["seed"], "start": n, "T": T,
                               "last_won": float(g["winner"] == g["turns"][-1]["mover"]),
                               "agent_won": float(g["winner"] == g["agent"]),
                               "first_agent": g["turns"][0]["mover"] == g["agent"]})
            n += T
        if rows:
            self.P = cat_positions(self.P, self.pos.build(rows))

    def targets(self, v: np.ndarray, lam: float, games=None) -> np.ndarray:
        """TD(lambda) targets for every position, from values `v` =
        P(mover wins) under the current critic."""
        g_all = np.empty(len(v))
        for g in (games or self.games):
            s, T = g["start"], g["T"]
            g_all[s: s + T] = lambda_returns(v[s + 1: s + T], g["last_won"], lam)
        return g_all

    def outcomes(self) -> np.ndarray:
        """1 if the position's mover went on to win: the Monte-Carlo target."""
        z = np.empty(len(self.P["words"]))
        for g in self.games:
            s, T = g["start"], g["T"]
            z[s: s + T] = lambda_returns(np.zeros(T - 1), g["last_won"], 1.0)
        return z


def fit_critic_steps(critic, opt, gf, S: GameStates, idx: np.ndarray, lam: float, steps: int,
                     batch: int, rng) -> float:
    """`steps` minibatch steps of BCE against TD(lambda) targets recomputed
    once with the critic as it stands."""
    critic.eval()
    with torch.no_grad():
        v = torch.sigmoid(critic_logits(critic, gf, S.P)).float().cpu().numpy()
    target = torch.as_tensor(S.targets(v, lam), dtype=torch.float32)
    critic.train()
    losses = []
    for _ in range(steps):
        i = rng.choice(idx, size=min(batch, len(idx)), replace=False)
        logit = critic_logits(critic, gf, S.P, i, grad=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target[i].to(gf.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    critic.eval()
    return float(np.mean(losses))


def count_features(S: GameStates, idx) -> np.ndarray:
    roles, pres = S.P["roles"][idx].numpy(), S.P["present"][idx].numpy()
    cols = [((roles == r) & pres).sum(1) for r in range(4)]
    return np.stack(cols + [S.P["agent"][idx].numpy().astype(int)], 1)


def critic_report(critic, gf, S: GameStates, val_idx: np.ndarray, train_idx: np.ndarray) -> dict:
    """Held-out log loss and Brier score against real results, beside a
    constant and a counts-only model, plus calibration and the martingale
    check (mean change in the mover's win probability over one half-turn)."""
    import lightgbm as lgb

    z = S.outcomes()
    with torch.no_grad():
        v = torch.sigmoid(critic_logits(critic, gf, S.P)).float().cpu().numpy()
    eps = 1e-6

    def scores(p, y):
        p = np.clip(p, eps, 1 - eps)
        return {"log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                "brier": float(np.mean((p - y) ** 2))}

    Xtr, Xva = count_features(S, train_idx), count_features(S, val_idx)
    gbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                             verbose=-1).fit(Xtr, z[train_idx])
    out = {"n_val_positions": int(len(val_idx)), "n_train_positions": int(len(train_idx)),
           "critic": scores(v[val_idx], z[val_idx]),
           "counts_only": scores(gbm.predict_proba(Xva)[:, 1], z[val_idx]),
           "constant": scores(np.full(len(val_idx), z[train_idx].mean()), z[val_idx])}
    bins = np.clip((v[val_idx] * 10).astype(int), 0, 9)
    out["calibration"] = [{"bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}", "n": int((bins == b).sum()),
                           "predicted": float(v[val_idx][bins == b].mean()),
                           "won": float(z[val_idx][bins == b].mean())}
                          for b in range(10) if (bins == b).sum() >= 10]
    val_set = set(val_idx.tolist())
    drift = []
    for g in S.games:
        s, T = g["start"], g["T"]
        if s not in val_set:
            continue
        for t in range(T - 1):
            drift.append((1 - v[s + t + 1]) - v[s + t])
        drift.append((g["last_won"]) - v[s + T - 1])
    out["martingale_mean_change"] = float(np.mean(drift))
    out["martingale_se"] = float(np.std(drift) / math.sqrt(len(drift)))
    return out


def split_games(S: GameStates, val_frac: float = 0.2):
    """Held-out games by seed (a fixed hash, so reruns agree)."""
    tr, va = [], []
    for g in S.games:
        rows = range(g["start"], g["start"] + g["T"])
        (va if (g["seed"] * 2654435761) % 1000 < val_frac * 1000 else tr).extend(rows)
    return np.array(tr), np.array(va)


# ---------------------------------------------------------------------------
# Commands


def setup(policy_path: Path | None):
    feats = PolicyFeatures.load(mmap=False)
    gf = DeviceFeatures(feats, "cuda")
    net = load_policy(policy_path, device="cuda")[0] if policy_path else None
    return feats, gf, net


def make_guesser(budget: float, already_spent: float = 0.0):
    guesser = build_guesser(GUESSER)
    return guesser, GuesserMeter(guesser, limit_dollars=budget, already_spent=already_spent)


def cmd_play(args) -> None:
    if args.agent == "policy" and not args.policy:
        raise SystemExit("--policy is required unless --agent incumbent")
    feats, gf, net = setup(Path(args.policy) if args.agent == "policy" else None)
    guesser, meter = make_guesser(args.budget)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {g["seed"] for g in read_games([out])} if out.exists() else set()
    if args.seeds == "val":
        # The same first n validation seeds for every policy: a paired comparison.
        seeds = [s for s in range(VAL_SEEDS[0], VAL_SEEDS[0] + args.n) if s not in done]
    else:
        # Fresh training seeds, never played by any earlier game file.
        used = {g["seed"] for g in read_games(list(DATA.glob("win_games_*.jsonl")))}
        seeds = next_seeds(max(0, args.n - len(done)), TRAIN_SEEDS, used)
    print(f"{len(seeds)} games to play ({len(done)} already in {out})", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.opp_workers, initializer=_opp_init) as opp:
        actor = Actor(net, feats, gf, greedy=args.greedy, branches=1) if args.agent == "policy" else IncumbentAgent(opp)
        for s in range(0, len(seeds), args.chunk):
            games = play_games(seeds[s: s + args.chunk], actor, guesser, opp, args.threads, out,
                               {"policy": str(args.policy) if args.agent == "policy" else "incumbent",
                                "greedy": args.greedy})
            report_games(games, meter, t0, s + len(games))
            if meter.limit is not None and meter.dollars >= meter.limit:
                print("budget spent; stopping")
                break


def report_games(games, meter, t0, n_done):
    ok = [g for g in games if not g.error]
    won = sum(g.winner == g.agent for g in ok)
    errs = [g.error for g in games if g.error]
    turns = np.mean([len(g.turns) for g in ok]) if ok else float("nan")
    print(f"  {n_done} games: this chunk won {won}/{len(ok)}, {len(errs)} errors "
          f"{sorted(set(e[:40] for e in errs))[:3]}, half-turns {turns:.1f}; calls {meter.n} "
          f"${meter.dollars:.3f}; {(time.time() - t0) / 60:.1f} min", flush=True)


def cmd_fit_critic(args) -> None:
    feats, gf, _ = setup(None)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    S = GameStates(Positions(feats, gf))
    t0 = time.time()
    S.add(read_games(args.games))
    print(f"{len(S.games)} games, {len(S.P['words'])} positions; features in {time.time() - t0:.0f}s", flush=True)
    tr, va = split_games(S)
    z = S.outcomes()
    results, best_epochs = {}, {}

    def new_critic(board: bool):
        critic = build_critic(len(feats.word_names), d=args.d, layers=args.layers, board=board,
                              dropout=args.dropout).cuda()
        return critic, torch.optim.AdamW(critic.param_groups(1e-4, args.board_wd), lr=args.lr)

    for variant, lam in [(v, l) for v in args.variants for l in args.lam]:
        critic, opt = new_critic(variant == "board")
        # Early stopping on held-out log loss against the real results: the
        # critic is small data (a few thousand games) and overfits.
        best_ll, best_state, best_epoch = math.inf, None, 0
        for epoch in range(1, args.epochs + 1):
            loss = fit_critic_steps(critic, opt, gf, S, tr, lam, max(1, len(tr) // args.batch), args.batch, rng)
            with torch.no_grad():
                p = torch.sigmoid(critic_logits(critic, gf, S.P, va)).float().cpu().numpy().clip(1e-6, 1 - 1e-6)
            ll = float(-np.mean(z[va] * np.log(p) + (1 - z[va]) * np.log(1 - p)))
            if ll < best_ll:
                best_ll, best_epoch = ll, epoch
                best_state = {k: v.detach().clone() for k, v in critic.state_dict().items()}
        critic.load_state_dict(best_state)
        rep = critic_report(critic, gf, S, va, tr)
        rep["best_epoch"] = best_epoch
        results[variant, lam], best_epochs[variant, lam] = (rep, critic), best_epoch
        print(f"{variant} lambda {lam}: best epoch {best_epoch}, last train loss {loss:.4f}  held-out "
              f"{json.dumps({k: rep[k] for k in ('critic', 'counts_only', 'constant')})}  "
              f"drift {rep['martingale_mean_change']:+.4f} ± {rep['martingale_se']:.4f}", flush=True)
    # The critic the actor uses reads the board: the counts-only variant is
    # there to show what the board adds, not to be chosen.
    keep = [k for k in results if k[0] == "board"] or list(results)
    best = min(keep, key=lambda k: results[k][0]["critic"]["log_loss"])
    rep, critic = results[best]
    print(f"best {best}; calibration:")
    for row in rep["calibration"]:
        print(f"   {row}")
    if args.final:
        # Refit on every game with the chosen lambda for the held-out-chosen
        # number of epochs (scaled for the extra data), for the actor to use.
        critic, opt = new_critic(best[0] == "board")
        every = np.arange(len(S.P["words"]))
        for epoch in range(best_epochs[best]):
            fit_critic_steps(critic, opt, gf, S, every, best[1], max(1, len(tr) // args.batch), args.batch, rng)
    save_critic(critic, Path(args.out), {"lam": best[1], "variant": best[0], "report": rep,
                                         "board_wd": args.board_wd, "games": [str(p) for p in args.games],
                                         "refit_on_all": bool(args.final), "args": vars(args) | {"games": None}})
    Path(args.out).with_suffix(".report.json").write_text(json.dumps(
        {f"{v} {l}": r for (v, l), (r, _) in results.items()} | {"best": list(best)}, indent=1))
    print(f"saved {args.out}")


def agent_samples(games: list[GameRecord], agent_positions: Positions, critic, gf, net):
    """Every agent turn's proposals with Q(k) = P(agent wins) after playing
    the first k of that proposal's real ranking, for k = 1..K_max."""
    items, rows, where = [], [], []
    for g in games:
        if g.error == "timeout":
            continue
        for t in g.turns:
            if not t.agent or not t.branches:
                continue
            item = {"seed": g.seed, "revealed": t.revealed, "team": t.mover, "branches": []}
            for b in t.branches:
                if "ranking" not in b:
                    continue
                afters = after_boards(g.seed, frozenset(t.revealed), t.mover, b["ranking"], b["kmax"])
                q = []
                for revealed, winner in afters:
                    if winner is not None:
                        q.append(1.0 if winner == t.mover else 0.0)
                    else:
                        q.append(None)
                        rows.append((g.seed, sorted(revealed), other(t.mover), False))
                        where.append((len(items), len(item["branches"]), len(q) - 1))
                item["branches"].append({"pool": b["pool"], "k": b["k"], "q": q})
            if len(item["branches"]):
                items.append(item)
    if rows:
        P = agent_positions.build(rows)
        with torch.no_grad():
            v_opp = torch.sigmoid(critic_logits(critic, gf, P)).float().cpu().numpy()
        for (i, b, k), v in zip(where, v_opp):
            items[i]["branches"][b]["q"][k] = 1.0 - float(v)
    return items


def cmd_train(args) -> None:
    feats, gf, _ = setup(None)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    init = Path(args.init)
    ref, _ = load_policy(init, device="cuda")
    for p in ref.parameters():
        p.requires_grad_(False)
    out = Path(args.out)
    state_path, games_path = out.with_suffix(".state.pt"), Path(args.games_out)
    resume = args.resume and state_path.exists()
    net, _ = load_policy(out if resume else init, device="cuda")
    net.train()
    critic, cmeta = load_critic(Path(args.critic), device="cuda")
    lam = cmeta["lam"] if args.lam is None else args.lam
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    copt = torch.optim.AdamW(critic.param_groups(1e-4, cmeta.get("board_wd", 0.05)), lr=args.critic_lr)
    start, history = 0, []
    if resume:
        st = torch.load(state_path, map_location="cuda", weights_only=False)
        opt.load_state_dict(st["opt"])
        critic.load_state_dict(st["critic"])
        copt.load_state_dict(st["copt"])
        start, history = st["it"], st["history"]
        print(f"resumed at iteration {start}")

    positions = Positions(feats, gf)
    S = GameStates(positions)
    pilot = [Path(p) for p in args.critic_games]
    S.add(read_games(pilot + ([games_path] if games_path.exists() else [])))
    print(f"critic replay: {len(S.games)} games, {len(S.P['words'])} positions", flush=True)
    used = {g["seed"] for g in read_games(list(DATA.glob("win_games_*.jsonl")) + [games_path] + pilot)}
    guesser, meter = make_guesser(args.budget, history[-1]["dollars"] if history else 0.0)
    actor = Actor(net, feats, gf, greedy=False, branches=args.branches)
    t0 = time.time()

    def checkpoint(it):
        save_policy(net, out, {"stage": "win_actor_critic", "iteration": it, "init": str(init),
                                  "init_hash": file_content_hash(init), "guesser": GUESSER,
                                  "critic": str(args.critic), "args": vars(args),
                                  "history": history[-1:] if history else []})
        save_critic(critic, out.with_suffix(".critic.pt"), {"lam": lam, "iteration": it})
        torch.save({"opt": opt.state_dict(), "critic": critic.state_dict(), "copt": copt.state_dict(),
                    "it": it, "history": history}, state_path)

    with ProcessPoolExecutor(args.opp_workers, initializer=_opp_init) as opp:
        for it in range(start + 1, args.iterations + 1):
            if meter.dollars >= meter.limit:
                print(f"stopping: budget ${meter.limit:.2f} spent")
                break
            seeds = next_seeds(args.games, TRAIN_SEEDS, used)
            used |= set(seeds)
            net.eval()
            games = play_games(seeds, actor, guesser, opp, args.threads, games_path, {"iteration": it})
            finished = [g for g in games if not g.error]
            if meter.dollars >= meter.limit and len(finished) < 0.9 * len(seeds):
                # The budget ran out mid-iteration. The games that finished are
                # the short ones, a biased sample, so no update is made from
                # them (the first short run's last iteration did: 42 of 112
                # games, 57% won).
                print(f"stopping: budget ${meter.limit:.2f} spent mid-iteration; "
                      f"its {len(finished)} finished games are not trained on")
                break
            if not finished:
                print("no game finished (errors); stopping")
                break

            # Actor targets first: Q(k) for every proposal from its real
            # ranking, scored by a critic that has not yet trained on these
            # games. Otherwise the branch that was played would be pulled
            # toward its game's actual result and the others would not.
            items = agent_samples(finished, positions, critic, gf, net)

            # Critic: add the new games, then a few steps on the replay.
            S.add([dataclasses.asdict(g) for g in finished])
            c_loss = fit_critic_steps(critic, copt, gf, S, np.arange(len(S.P["words"])), lam,
                                      args.critic_steps, 256, rng)
            net.train()
            stats = actor_step(net, ref, opt, gf, feats, items, args)
            won = np.mean([g.winner == g.agent for g in finished])
            rec = {"it": it, "games": len(finished), "won": float(won), "critic_loss": c_loss, **stats,
                   "calls": meter.n, "dollars": meter.dollars, "minutes": (time.time() - t0) / 60}
            history.append(rec)
            print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
            checkpoint(it)
    print("done")


def actor_step(net, ref, opt, gf, feats, items, args) -> dict:
    """One optimiser step over this iteration's agent positions.

    Per position with proposals b = 1..B (all sampled from the policy as it
    was): Q_b = sum_k pi(k | clue_b) Q_b(k), the clue's win probability under
    the current k policy on its real ranking. The clue term is REINFORCE with
    a leave-one-out baseline (the mean Q of the other proposals; the critic's
    V(position) when there is only one). The k term is the exact gradient of
    sum_k pi(k) Q(k): no sampling noise from k. A KL penalty to the
    imitation policy keeps the clue distribution near its start."""
    import random

    random.Random(0).shuffle(items)
    n = sum(len(it["branches"]) for it in items)
    tot = {"adv_abs": 0.0, "kl": 0.0, "q": 0.0, "q_best_k": 0.0, "k": 0.0, "entropy": 0.0}
    # `args.steps` optimiser steps, each on its own share of the positions:
    # more updates per round of games, while every sample is still used once.
    groups = [items[g::args.steps] for g in range(args.steps)]
    for group in groups:
        if group:
            _actor_group(net, ref, opt, gf, feats, group, args, tot)
    npos = max(1, len(items))
    return {"positions": len(items), "proposals": n,
            "distinct_per_position": float(np.mean([len({b["pool"] for b in it["branches"]}) for it in items])) if items else 0.0,
            "q": tot["q"] / max(1, n), "q_best_k": tot["q_best_k"] / max(1, n), "mean_k": tot["k"] / max(1, n),
            "adv_abs": tot["adv_abs"] / max(1, n), "kl": tot["kl"] / npos, "entropy": tot["entropy"] / npos}


def _actor_group(net, ref, opt, gf, feats, items, args, tot) -> None:
    opt.zero_grad(set_to_none=True)
    for s in range(0, len(items), args.chunk):
        chunk = items[s: s + args.chunk]
        boards = [feats.encode(view_for(make_board(it["seed"], frozenset(it["revealed"])), it["team"])) for it in chunk]
        pair, word, roles, present, legal, kmax = gf.batch([b.words for b in boards], [b.roles for b in boards],
                                                           [b.present for b in boards], max_number=net.max_number)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, klog, _ = net(pair, word, roles, present)
            with torch.no_grad():
                rlogits, rklog, _ = ref(pair, word, roles, present)
        logp = masked_log_policy(logits.float(), legal)
        rlogp = masked_log_policy(rlogits.float(), legal)
        lp, rlp = logp.masked_fill(~legal, 0.0), rlogp.masked_fill(~legal, 0.0)
        kl = (lp.exp() * legal * (lp - rlp)).sum(1)
        ent = -(lp.exp() * legal * lp).sum(1)
        loss = args.beta * kl.sum()
        for i, it in enumerate(chunk):
            B = len(it["branches"])
            qs, terms = [], []
            for b in it["branches"]:
                c = b["pool"]
                logk = masked_log_k(klog[i, c].float()[None], kmax[i: i + 1])[0]
                rlogk = masked_log_k(rklog[i, c].float()[None], kmax[i: i + 1])[0]
                K = len(b["q"])
                q = torch.tensor(b["q"], device=gf.device)
                pk = logk[:K].exp()
                qs.append(float((pk.detach() * q).sum()))
                okk = torch.isfinite(logk[:K])
                k_kl = (pk * (logk[:K] - rlogk[:K]).masked_fill(~okk, 0.0)).sum()
                terms.append((c, pk, q, k_kl))
                tot["q_best_k"] += float(q.max())
                tot["k"] += float((pk.detach() * torch.arange(1, K + 1, device=gf.device)).sum())
            for j, (c, pk, q, k_kl) in enumerate(terms):
                base = (sum(qs) - qs[j]) / (B - 1) if B > 1 else it.get("v", qs[j])
                adv = qs[j] - base
                loss = loss - adv * args.adv_scale * logp[i, c] / B
                loss = loss - (pk * q).sum() * args.k_scale / B + args.beta * k_kl / B
                tot["adv_abs"] += abs(adv)
                tot["q"] += qs[j]
            tot["kl"] += kl[i].item()
            tot["entropy"] += ent[i].item()
        (loss / max(1, len(items))).backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    opt.step()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play")
    p.add_argument("--policy")
    p.add_argument("--agent", choices=["policy", "incumbent"], default="policy")
    p.add_argument("--seeds", choices=["train", "val"], required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=float, required=True, help="dollars of guesser calls, at most")
    p.add_argument("--threads", type=int, default=64, help="games at once")
    p.add_argument("--chunk", type=int, default=128, help="games per progress line")
    p.add_argument("--opp-workers", type=int, default=10)

    p = sub.add_parser("fit-critic")
    p.add_argument("games", nargs="+", type=Path)
    p.add_argument("--lam", type=float, nargs="+", default=[1.0, 0.8, 0.5])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--board-wd", type=float, default=0.05, help="weight decay on the board correction")
    p.add_argument("--variants", nargs="+", choices=["board", "counts"], default=["board", "counts"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--final", action="store_true", help="refit the chosen lambda on every game before saving")
    p.add_argument("--out", default=str(CRITIC))

    p = sub.add_parser("train")
    p.add_argument("--init", default=str(DEFAULT_CACHE_DIR / "win_policy_init.pt"))
    p.add_argument("--critic", default=str(CRITIC))
    p.add_argument("--critic-games", nargs="*", default=[str(DATA / "win_games_pilot.jsonl")])
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--games", type=int, default=48, help="games per iteration")
    p.add_argument("--branches", type=int, default=4, help="clues proposed per agent position")
    p.add_argument("--threads", type=int, default=48)
    p.add_argument("--opp-workers", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--critic-steps", type=int, default=40)
    p.add_argument("--lam", type=float, default=None, help="default: the critic's fitted lambda")
    p.add_argument("--beta", type=float, default=0.05, help="KL penalty to the imitation policy")
    p.add_argument("--adv-scale", type=float, default=10.0,
                   help="advantages are win-probability differences, ~0.01-0.1; scaled up so the clue term is not swamped by the KL")
    p.add_argument("--k-scale", type=float, default=10.0)
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--steps", type=int, default=1, help="optimiser steps per iteration")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", default=str(POLICY), help="the policy; its state and critic are saved beside it")
    p.add_argument("--games-out", default=str(TRAIN_GAMES))

    args = ap.parse_args()
    {"play": cmd_play, "fit-critic": cmd_fit_critic, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()

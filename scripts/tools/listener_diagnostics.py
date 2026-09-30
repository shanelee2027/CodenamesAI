"""Where is the listener's unexplained variance? Three free checks
(docs/log.md, "Where the listener's unexplained variance is").

1. **Backfill in the labels.** A guesser that names only some words gets the
   rest appended in prompt order (LLMGuesser._parse_ranking), and the store
   keeps only the parsed ranking. So a backfilled tail is a suffix of the
   ranking in prompt order. A genuine ranking ends in a run of length L in
   prompt order with probability 1/L!, so a run of 4 or more is taken as
   backfill. Reported: how often it reaches into the first k picks, which are
   the ones the listener trains on.
2. **Prompt position.** The guesser sees the unrevealed words in board order,
   and board order is random with respect to meaning. So any link between a
   word's position in the prompt and whether it is picked is a pure order
   effect, not confounded with the clue. Reported: pick share by position
   against the uniform share, per pick. Then an additive position term
   (position fraction, first, last; separately per pick) on top of the GBT
   score, fitted on val by maximum likelihood and scored on the test sets.
3. **Error analysis.** The refitted GBT's (train_listener_net.py base) R² by
   pick, number, clue rarity, clue part of speech, and multi-word target. Then
   the worst first picks: clue, target and the model's favourite.

    python scripts/tools/listener_diagnostics.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from train_listener_net import BASE_OUT, load_sets, pkey  # noqa: E402

STEPS = [(0, "pick 1"), (1, "pick 2"), (2, "pick 3"), (3, "pick 4+")]


def prompt_pos(p: dict) -> np.ndarray:
    """Row r's position in the list the guesser was shown, 0-based."""
    order = {w.lower(): i for i, w in enumerate(p["key"][1])}
    return np.array([order[w.lower()] for w in p["words"]])


def ranking(p: dict) -> list[int]:
    """The stored ranking as prompt positions."""
    pos = prompt_pos(p)
    return [int(pos[t]) for t in p["targets"]]


def ordered_suffix(r: list[int]) -> int:
    L = 1
    while L < len(r) and r[-L - 1] < r[-L]:
        L += 1
    return L


def events(ps: list[dict], base: dict):
    """Per choice event: GBT scores, prompt positions, availability, target, step."""
    E = []
    for p in ps:
        s, pos, n = base[pkey(p)], prompt_pos(p), p["n"]
        avail = np.ones(n, bool)
        for j in range(min(p["k"], n - 1)):
            E.append((s, pos, avail.copy(), p["targets"][j], j, n, p))
            avail[p["targets"][j]] = False
    return E


def basis(pos: np.ndarray, n: int) -> np.ndarray:
    return np.stack([pos / max(n - 1, 1), pos == 0, pos == n - 1], 1).astype(np.float64)


def fit_position(E_fit, E_tests: dict) -> None:
    def tensors(E):
        N = 25
        S = np.full((len(E), N), -np.inf)
        X = np.zeros((len(E), N, 3))
        T = np.zeros(len(E), np.int64)
        J = np.zeros(len(E), np.int64)
        null = np.zeros(len(E))
        for i, (s, pos, av, t, j, n, _) in enumerate(E):
            S[i, :n] = np.where(av, s, -np.inf)
            X[i, :n] = basis(pos, n)
            T[i], J[i], null[i] = t, min(j, 3), np.log(av.sum())
        return torch.tensor(S), torch.tensor(X), torch.tensor(T), torch.tensor(J), null

    def nll(beta, S, X, T, J):
        z = S + (X * beta[J][:, None, :]).sum(-1)
        return -z.log_softmax(-1).gather(1, T[:, None]).squeeze(1)

    S, X, T, J, _ = tensors(E_fit)
    beta = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([beta], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nll(beta, S, X, T, J).mean()
        loss.backward()
        return loss

    opt.step(closure)
    b = beta.detach()
    print("  fitted on val (logits added per pick: position fraction, first word, last word):")
    for j, name in STEPS:
        print(f"    {name:8s} {b[j, 0]:+.3f} {b[j, 1]:+.3f} {b[j, 2]:+.3f}")
    print(f"  {'':24s} {'R2 GBT':>8s} {'+ position':>11s}   by pick (GBT -> + position)")
    for name, E in E_tests.items():
        S, X, T, J, null = tensors(E)
        a = nll(torch.zeros(4, 3, dtype=torch.float64), S, X, T, J).numpy()
        c = nll(b, S, X, T, J).numpy()
        Jn = J.numpy()
        per = "  ".join(f"{1 - a[Jn == j].mean() / null[Jn == j].mean():.3f}->"
                        f"{1 - c[Jn == j].mean() / null[Jn == j].mean():.3f}" for j, _ in STEPS if (Jn == j).any())
        print(f"  {name:24s} {1 - a.mean() / null.mean():8.4f} {1 - c.mean() / null.mean():11.4f}   {per}")


def pos_of_clue(clue: str) -> str:
    from nltk.corpus import wordnet as wn

    counts = {}
    for s in wn.synsets(clue):
        k = {"s": "a"}.get(s.pos(), s.pos())
        counts[k] = counts.get(k, 0) + sum(l.count() for l in s.lemmas()) + 1
    return {"n": "noun", "v": "verb", "a": "adj", "r": "adv"}.get(max(counts, key=counts.get), "?") if counts else "none"


def main() -> None:
    from codenames.clue_stats import ClueStats
    from codenames.similarity import DEFAULT_CACHE_DIR

    sets = load_sets()
    base = pickle.loads(BASE_OUT.read_bytes())
    gpt = {k: v for k, v in sets.items() if k != "held-out words, Sonnet"}

    print("\n1. Backfill: stored rankings ending in a run of >= 4 words in prompt order")
    for name, ps in sets.items():
        runs = [(ordered_suffix(ranking(p)), p) for p in ps]
        tail = [(L, p) for L, p in runs if L >= 4]
        into = sum(1 for L, p in tail if p["n"] - L < min(p["k"], p["n"] - 1))
        print(f"  {name:24s} {len(tail) / len(ps):6.1%} of {len(ps)} rankings;"
              f" reaching the trained picks: {into} ({into / len(ps):.2%})")

    def backfilled(p):
        return p["n"] - ordered_suffix(ranking(p)) < min(p["k"], p["n"] - 1)

    print("  GBT R2 on positions whose trained picks are backfilled, against the rest:")
    for name, ps in sets.items():
        if name == "train":
            continue
        bad = [p for p in ps if backfilled(p)]
        good = [p for p in ps if not backfilled(p)]
        if not bad:
            continue
        def r2(q):
            E = events(q, base)
            nll = [-(np.where(av, s_, -np.inf) - np.logaddexp.reduce(np.where(av, s_, -np.inf)))[t]
                   for s_, pos, av, t, j, n, _ in E]
            return 1 - np.mean(nll) / np.mean([np.log(av.sum()) for _, _, av, *_ in E])
        print(f"    {name:24s} backfilled {len(bad):4d}: R2 {r2(bad):.3f}   rest: R2 {r2(good):.3f}")
    clean = {k: [p for p in v if not backfilled(p)] for k, v in sets.items()}

    print("\n2. Prompt position (backfilled positions removed)")
    sets = clean
    print("  pick share by position fifth (first fifth of the list ... last), as a ratio to uniform:")
    for name in ("val", "held-out words", "held-out words, Sonnet"):
        E = events(sets[name], base)
        for j, label in STEPS[:3]:
            got, exp = np.zeros(5), np.zeros(5)
            for s, pos, av, t, jj, n, _ in E:
                if jj != j:
                    continue
                q = np.minimum((pos * 5) // n, 4)
                got[q[t]] += 1
                np.add.at(exp, q[av], 1.0 / av.sum())
            print(f"    {name:24s} {label}: " + "  ".join(f"{g / e:.2f}" for g, e in zip(got, exp)))
    fit_position(events(sets["val"], base), {k: events(v, base) for k, v in sets.items() if k != "train"})

    print("\n3. Error analysis (refitted GBT; val and new boards pooled, all out of sample)")
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    rar = {w.lower(): float(r) for w, r in zip(stats.clue_words, stats.rarity_percentile)}
    E = events(gpt["val"] + gpt["new boards"], base)
    rows = []
    for s, pos, av, t, j, n, p in E:
        z = np.where(av, s, -np.inf)
        lp = z - z.max() - np.log(np.exp(z - z.max()).sum())
        rows.append({"nll": -lp[t], "null": np.log(av.sum()), "step": min(j, 3), "k": p["k"],
                     "rarity": rar.get(p["clue"].lower(), np.nan), "clue": p["clue"], "p": p,
                     "multi": " " in p["words"][t] or "-" in p["words"][t], "top": int(np.argmax(z)),
                     "t": t, "pt": float(np.exp(lp[t])), "ptop": float(np.exp(lp.max()))})
    pos_cache = {}
    for r in rows:
        c = r["clue"].lower()
        if c not in pos_cache:
            pos_cache[c] = pos_of_clue(c)
        r["pos"] = pos_cache[c]

    def table(title, key, order=None):
        groups = {}
        for r in rows:
            groups.setdefault(key(r), []).append(r)
        print(f"  by {title}:")
        for g in (order or sorted(g for g in groups if g is not None)):
            rs = groups.get(g, [])
            if len(rs) < 30:
                continue
            nll = np.mean([r["nll"] for r in rs])
            null = np.mean([r["null"] for r in rs])
            print(f"    {str(g):14s} n={len(rs):6d}   R2 {1 - nll / null:.3f}")

    table("pick", lambda r: r["step"] + 1)
    table("announced number (pick 1 only)", lambda r: r["k"] if r["step"] == 0 else None, [1, 2, 3, 4])
    table("clue rarity percentile (pick 1)",
          lambda r: None if r["step"] or np.isnan(r["rarity"]) else f"{int(r['rarity'] // 2.5) * 2.5:.1f}-",
          None)
    table("clue part of speech (pick 1)", lambda r: r["pos"] if r["step"] == 0 else None,
          ["noun", "verb", "adj", "adv", "none"])
    table("multi-word target (pick 1)", lambda r: r["multi"] if r["step"] == 0 else None, [False, True])

    print("\n  the 25 worst first picks (clue, number: the guesser's pick p=..., the model's favourite p=...):")
    worst = sorted((r for r in rows if r["step"] == 0), key=lambda r: -r["nll"])[:25]
    for r in worst:
        w = r["p"]["words"]
        print(f"    {r['clue']:14s} {r['k']}: picked {w[r['t']]} p={r['pt']:.3f}; "
              f"model's favourite {w[r['top']]} p={r['ptop']:.2f}")


if __name__ == "__main__":
    main()

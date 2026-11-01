"""How the history-aware guesser goes back to earlier clues: an additive
history term on top of the incumbent's listener (docs/log.md, "Plan after
probe 3", option B).

**Data.** Turns of probe_history_rule.py's `rule_medium` guesser: the 2,000
from collect_history_turns.py to fit, the probe's 300 to test (different
games).

**What the guesser can see,** and so all the term reads: the earlier clues of
its spymaster and, for each, how many of that turn's guesses were its own
words. An earlier clue c is *unfinished* while number(c) exceeds that count.
For each word w still on the board and each unfinished clue c, the
incumbent's listener gives P_c(w), the probability of w for c over the words
now on the board (its softmax at c's number). The term reads the strongest
such link: `link` = max_c log P_c(w), and whether w is c's top word now.
Words with no unfinished clue get `link` = log(1 / n), the uniform level, and
top = 0.

**Model.** One conditional-logit event per listed word (pick j among the
words not yet listed; stopping is not modelled here):
    logit(w) = a * s(w) + h(w),  h(w) = b1 * (link(w) - log(1/n)) + b2 * top(w),
with s the incumbent's listener score for the current clue. The baseline is
the same with h = 0 (only a fitted). Reported: per-event log-likelihood gain
and the R² of both against the uniform choice, on the test turns, and the same
for the +1 picks only (the picks beyond the number).

    python scripts/tools/fit_history_pickup.py
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

import probe_history_guesser as H  # noqa: E402
import probe_history_rule as R  # noqa: E402

TURNS = PROJECT_ROOT / "cache" / "training_data" / "history_turns.pkl"


def probe_turns(db: Path) -> list[dict]:
    """The probe's 300 positions with their stored rule_medium answers."""
    pos, _ = H.sample(H.positions(db), 300, 0)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = []
    for p in pos:
        key = json.dumps([f"deepinfra/{H.MODEL}+effort=medium", R.VERSION, R.prompt_for(p, "rule_medium")])
        row = conn.execute("SELECT guesses FROM turn_probe WHERE cache_key = ?", (key,)).fetchone()
        if row is not None:
            out.append({**p, "guesses": json.loads(row[0])})
    return out


def history_columns(p: dict, sm, sims) -> tuple[np.ndarray, np.ndarray]:
    """(link measured from the uniform level, top) over p["words"]."""
    words, us = p["words"], p["team"]
    n = len(words)
    link = np.full(n, np.log(1 / n))
    top = np.zeros(n)
    for t in p["past"]:
        if t["team"] != us or not t["clue"]:
            continue
        if sum(r == "own" for _, r in t["guesses"]) >= int(t["number"]):
            continue
        sc = sm._listener_scores(t["clue"], words, int(t["number"]), sims)
        if sc is None:
            continue
        link = np.maximum(link, sc - (sc.max() + np.log(np.exp(sc - sc.max()).sum())))
        top[int(np.argmax(sc))] = 1.0
    return link - np.log(1 / n), top


def left_alone(turns: list[dict], sm, sims, beyond: np.ndarray, w: np.ndarray) -> np.ndarray:
    """For each pending word a re-clue targeted: the chance the guesser would
    have spent the +1 on it had the clue left it alone. The clue is taken to
    target only its other words (all found), the +1 to be taken with the
    fitted logistic, and the word chosen by the beyond-the-number weights
    (the current clue's term dropped: it is near 0 there, and the
    counterfactual clue is not known). An upper bound on the turn's pickup:
    in play the +1 also needs the clue's words to be found first."""
    out = []
    for p in turns:
        clued = set(p["targets"]) & set(p["pending"])
        rest = [t for t in p["targets"] if t not in clued]
        if not clued or not rest:
            continue
        link, top = history_columns(p, sm, sims)
        keep = np.array([x not in rest for x in p["words"]])
        z = beyond[1] * link[keep] + beyond[2] * top[keep]
        prob = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        take = 1 / (1 + np.exp(-(w[0] + w[1] * link[keep].max() + w[2] * len(rest))))
        kept = [x for x, k in zip(p["words"], keep) if k]
        out += [take * prob[kept.index(x)] for x in clued]
    return np.asarray(out)


def events(turns: list[dict], sm, sims, per_turn: list | None = None) -> list[dict]:
    """Per listed word: the current clue's scores over the words not yet
    listed, the history columns, the chosen row and whether it is a +1 pick.
    `per_turn`, if given, gets each turn's events (an empty list if none)."""
    out = []
    for p in turns:
        start = len(out)
        if per_turn is not None:
            per_turn.append([])
        words = p["words"]
        s = sm._listener_scores(p["clue"], words, p["number"], sims)
        if s is None or not p["guesses"]:
            continue
        n = len(words)
        link, top = history_columns(p, sm, sims)
        link = link + np.log(1 / n)
        index = {w: i for i, w in enumerate(words)}
        left = np.ones(n, dtype=bool)
        for j, g in enumerate(p["guesses"]):
            i = index[g]
            out.append({"s": s[left], "link": link[left] - np.log(1 / n), "top": top[left],
                        "y": int(np.flatnonzero(left).tolist().index(i)), "plus1": j >= p["number"],
                        "pending": g in p["pending"] and g not in p["targets"]})
            left[i] = False
        if per_turn is not None:
            per_turn[-1] = out[start:]
    return out


def plus_one_rows(turns: list[dict], ev_by_turn: list[list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    """Turns that listed at least the number: whether they went on to the +1,
    against the strongest earlier-clue link among the words left after the
    first `number` (0 when no earlier clue is unfinished) and the number."""
    X, y = [], []
    for p, ev in zip(turns, ev_by_turn):
        if len(p["guesses"]) < p["number"] or len(ev) < p["number"]:
            continue
        if len(ev) > p["number"]:
            left = ev[p["number"]]["link"]
        else:
            last = ev[p["number"] - 1]
            left = np.delete(last["link"], last["y"])
        X.append([1.0, float(left.max()) if len(left) else 0.0, p["number"]])
        y.append(len(p["guesses"]) > p["number"])
    return np.asarray(X), np.asarray(y, dtype=float)


def nll(theta: np.ndarray, ev: list[dict], history: bool, split: bool = False) -> np.ndarray:
    """theta = (a, b1, b2), or with `split` (a, b1, b2) within the number and
    (a', b1', b2') for the picks beyond it."""
    res = []
    for e in ev:
        t = theta[3:6] if split and e["plus1"] else theta[:3]
        a = t[0]
        b1, b2 = (t[1], t[2]) if history else (0.0, 0.0)
        z = a * e["s"] + b1 * e["link"] + b2 * e["top"]
        m = z.max()
        res.append(m + np.log(np.exp(z - m).sum()) - z[e["y"]])
    return np.asarray(res)


def main() -> None:
    from scipy.optimize import minimize

    from codenames.similarity import SimilarityTensor
    from codenames.spymasters.registry import spymaster_spec

    db = PROJECT_ROOT / "cache" / "llm_store.db"
    sims = SimilarityTensor.load()
    cls, kw = spymaster_spec(H.INCUMBENT)
    sm = cls(**kw)
    # Paired arena games often reach the same board with the same history and
    # the same incumbent clue, so many collected prompts are word for word the
    # same (one stored answer). One turn per distinct prompt, and none that is
    # also a test prompt.
    test_turns = probe_turns(db)
    seen = {R.prompt_for(p, "rule_medium") for p in test_turns}
    fit_turns = []
    for p in pickle.loads(TURNS.read_bytes()):
        k = R.prompt_for(p, "rule_medium")
        if k not in seen:
            seen.add(k)
            fit_turns.append(p)
    pt_fit, pt_test = [], []
    ev_fit, ev_test = events(fit_turns, sm, sims, pt_fit), events(test_turns, sm, sims, pt_test)
    print(f"fit: {len(fit_turns)} turns, {len(ev_fit)} picks ({sum(e['plus1'] for e in ev_fit)} beyond the number); "
          f"test: {len(test_turns)} turns, {len(ev_test)} picks ({sum(e['plus1'] for e in ev_test)} beyond)")

    opts = {"maxiter": 8000, "xatol": 1e-5, "fatol": 1e-7}
    base = minimize(lambda t: nll(np.r_[t, 0, 0], ev_fit, False).mean(), np.array([1.0]), method="Nelder-Mead")
    full = minimize(lambda t: nll(t, ev_fit, True).mean(), np.array([base.x[0], 0.0, 0.0]), method="Nelder-Mead",
                    options=opts)
    split = minimize(lambda t: nll(t, ev_fit, True, True).mean(), np.r_[full.x, full.x], method="Nelder-Mead",
                     options=opts)
    print(f"baseline a = {base.x[0]:.3f};  history a = {full.x[0]:.3f}, b_link = {full.x[1]:.3f}, "
          f"b_top = {full.x[2]:.3f}")
    print("split: within the number a, b_link, b_top = " + ", ".join(f"{v:.3f}" for v in split.x[:3])
          + ";  beyond it " + ", ".join(f"{v:.3f}" for v in split.x[3:]))
    for name, ev in (("fit", ev_fit), ("test", ev_test)):
        null = np.array([np.log(len(e["s"])) for e in ev])
        n0, n1, n2 = nll(np.r_[base.x, 0, 0], ev, False), nll(full.x, ev, True), nll(split.x, ev, True, True)
        plus = np.array([e["plus1"] for e in ev])
        pend = np.array([e["pending"] for e in ev])
        r2 = lambda v, m: 1 - v[m].sum() / null[m].sum()               # noqa: E731
        every = np.ones(len(ev), dtype=bool)
        print(f"  {name:4s} all picks  R2 {r2(n0, every):.4f} -> {r2(n1, every):.4f} -> split {r2(n2, every):.4f}   "
              f"+1 picks R2 {r2(n0, plus):.4f} -> {r2(n1, plus):.4f} -> {r2(n2, plus):.4f}   "
              f"pending-word picks: mean -log p {n0[pend].mean():.2f} -> {n1[pend].mean():.2f} -> "
              f"{n2[pend].mean():.2f} ({pend.sum()})")

    # Whether the +1 is taken: logistic on the strongest leftover link.
    Xf, yf = plus_one_rows(fit_turns, pt_fit)
    Xt, yt = plus_one_rows(test_turns, pt_test)
    def lnll(w, X, y):
        z = X @ w
        return np.logaddexp(0, z) - y * z
    w0 = minimize(lambda w: lnll(np.r_[w, 0, 0], Xf, yf).mean(), np.zeros(1)).x
    w = minimize(lambda w: lnll(w, Xf, yf).mean(), np.zeros(3)).x
    for name, X, y in (("fit", Xf, yf), ("test", Xt, yt)):
        a, b = lnll(np.r_[w0, 0, 0], X, y), lnll(w, X, y)
        print(f"  +1 taken, {name}: {y.mean():.1%} of {len(y)} turns; log-loss {a.mean():.4f} (rate only) -> "
              f"{b.mean():.4f} (R2 {1 - b.sum() / a.sum():.3f}); w = " + ", ".join(f"{v:.3f}" for v in w))

    # What re-clueing buys under this guesser: the incumbent's re-clued
    # pending words, had the clue left them alone.
    reclue = [p for p in fit_turns if p.get("kind") == "reclue"]
    pick = left_alone(reclue, sm, sims, split.x[3:], w)
    print(f"  re-clued pending words ({len(pick)} in {len(reclue)} re-clue turns): chance of a +1 pickup had the clue "
          f"left them alone, mean {pick.mean():.3f}, median {np.median(pick):.3f}, above 0.3 {(pick > 0.3).mean():.1%}")


if __name__ == "__main__":
    main()

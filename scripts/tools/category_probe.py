"""Does the listener tell category members from things merely associated with
the category? (docs/log.md, "Category clues: members against associates").

The worry: on a board with Tiger, Whale, Octopus (ours) and Africa (not ours),
a human reads "animal 3" as exactly the three animals, but Africa sits close
to "animal" in embedding space, so the listener might rate it a likely pick,
price the clue as risky, and never give it.

Each case is a category clue, three members of it as our words, and one
associate of the category that is not a member (Tiger and Russia are not on
the board list, so the user's two examples use Bear and Germany). Boards are
built around them: the members as three of our nine words, the associate as a
neutral, an opponent or the assassin, and 21 fillers drawn from board words
outside every case's theme (EXCLUDE). The control is the same board with the
associate replaced by a filler of the same role, so the difference between
the two is the associate's effect alone.

Reported per case and role, averaged over boards:
- p(assoc): the listener's first-pick probability for the associate, and for
  its control filler; p(members): the three members' total.
- P(3 members): the frozen-score (Plackett-Luce) probability that the first
  three picks are exactly the members, which is what "clue 3" is scored on.
- value 3: the incumbent's reward for the clue at number 3 (gain - penalty).
- rank: the clue's place among all clues the incumbent scored on that board
  (1 = it gives this clue), and how often stage one (the Gaussian shortlist)
  let it through at all.

With --llm N, gpt-oss also ranks the words for the first N boards of each
case (associate neutral), a few cents at most, to show what the real guesser
does with the same clue.

    python scripts/tools/category_probe.py [--boards 20] [--llm 3]
"""

from __future__ import annotations

import argparse
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

# (clue, members, associate)
CASES = [
    ("animal", ["Bear", "Whale", "Octopus"], "Africa"),
    ("animal", ["Bear", "Whale", "Octopus"], "Forest"),
    ("country", ["America", "China", "Germany"], "Field"),
    ("country", ["America", "China", "India"], "Berlin"),
    ("planet", ["Mercury", "Saturn", "Jupiter"], "Moon"),
    ("fruit", ["Orange", "Lemon", "Apple"], "Pie"),
    ("instrument", ["Piano", "Flute", "Organ"], "Concert"),
    ("metal", ["Iron", "Gold", "Copper"], "Mine"),
    ("city", ["London", "Tokyo", "Rome"], "England"),
    ("continent", ["Europe", "Africa", "Antarctica"], "Egypt"),
]

# Words in (or strongly tied to) any case's theme, never used as filler.
EXCLUDE = set("""
Africa Alps Amazon America Antarctica Atlantis Australia Aztec Beijing Berlin Bermuda Canada Capital China Czech
Egypt England Europe France Germany Greece Himalayas Hollywood India London Mexico Moscow New_york Olympus Rome
Tokyo Washington Embassy State Field Forest Park Beach Cliff Ground Grass
Bat Bear Buck Buffalo Calf Cat Chick Crane Cricket Dinosaur Dog Dragon Duck Eagle Fish Fly Hawk Horse Kangaroo Lion
Mammoth Mole Mouse Octopus Penguin Platypus Rabbit Robin Scorpion Seal Shark Slug Spider Turkey Unicorn Whale Worm
Bug Centaur Phoenix Leprechaun Loch_ness Ivory Tail Web
Jupiter Mercury Moon Saturn Space Star Satellite Telescope Alien Ray Light
Apple Berry Carrot Kiwi Lemon Olive Orange Pumpkin Pie Honey Jam Ketchup Ham Chocolate Ice_cream Nut Maple Palm Mint
Bugle Concert Conductor Flute Horn Organ Piano Opera Band Note Scale Sound Beat Theater Dance Pitch
Copper Gold Iron Lead Mine Diamond Marble Ring Plastic Crown Glass
""".replace("_", " ").split())

ROLE_NAMES = ("neutral", "opponent", "assassin")
_W: dict = {}


def build(case_i: int, rep: int, assoc_role: str, control: bool):
    from codenames.board import Board, Card, Role, load_wordlist

    clue, members, assoc = CASES[case_i]
    rng = random.Random(1000 * case_i + rep)
    pool = [w for w in load_wordlist() if w not in EXCLUDE and w not in members and w != assoc]
    fill = rng.sample(pool, 22)
    # fill[0] stands in for the associate on the control board.
    fourth = fill[0] if control else assoc
    counts = {Role.OWN: 9 - 3, Role.NEUTRAL: 7, Role.OPPONENT: 8, Role.ASSASSIN: 1}
    counts[Role[assoc_role.upper()]] -= 1
    roles = [r for r, n in counts.items() for _ in range(n)]
    cards = [Card(w, Role.OWN) for w in members] + [Card(fourth, Role[assoc_role.upper()])]
    cards += [Card(w, r) for w, r in zip(fill[1:], roles)]
    rng.shuffle(cards)
    return Board(cards=tuple(cards), seed=-1), clue, members, fourth


def _init() -> None:
    from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
    from codenames.spymasters.registry import spymaster_spec

    _W["sims"] = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    cls, kw = spymaster_spec("learned_listener")
    _W["sm"] = cls(**kw)


def _run(job) -> dict:
    from codenames.board import Role, clue_number_cap
    from codenames.pl_reward import gain_and_penalty
    from codenames.spymasters.base import TurnContext

    case_i, rep, assoc_role, control = job
    board, clue, members, fourth = build(*job)
    sm, sims = _W["sm"], _W["sims"]
    got = sm.listen(board, clue, sims)
    words, roles, s = got["words"], got["roles"], got["scores"]
    p = np.exp(s - s.max())
    p /= p.sum()
    idx = {w: i for i, w in enumerate(words)}
    # P(first three picks are exactly the members), frozen scores.
    e = np.exp(s - s.max())
    tot, pm = e.sum(), 0.0
    from itertools import permutations
    for perm in permutations([idx[m] for m in members]):
        left, q = tot, 1.0
        for i in perm:
            q *= e[i] / left
            left -= e[i]
        pm += q
    n_own = sum(r is Role.OWN for r in roles)
    costs = np.array([sm.costs[r] for r in roles[n_own:]])
    gain, pen = gain_and_penalty(s[None, :n_own], s[None, n_own:], costs, clue_number_cap(n_own, sm.max_number))
    value3 = float((gain - pen)[0, 2])
    best_n, scores, _ = sm._score_all_clues(board, sims)
    ci = next((i for i, w in enumerate(sims.clue_words) if w == clue), None)
    fin = np.isfinite(scores)
    shortlisted = ci is not None and bool(fin[ci])
    rank = int((scores[fin] > scores[ci]).sum()) + 1 if shortlisted else None
    top = sm.top_clues(TurnContext(board=board, turn_index=0), sims, 3)
    return {"case": case_i, "rep": rep, "role": assoc_role, "control": control,
            "p_assoc": float(p[idx[fourth]]), "p_members": float(sum(p[idx[m]] for m in members)),
            "p_members_min": float(min(p[idx[m]] for m in members)),
            "assoc_rank": int((p > p[idx[fourth]]).sum()) + 1,
            "pm3": pm, "value3": value3, "shortlisted": shortlisted, "rank": rank,
            "best_n": int(best_n[ci]) if shortlisted else None,
            "top": [(c, int(k), round(float(v), 2)) for c, k, v in top]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boards", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--llm", type=int, default=0)
    args = ap.parse_args()

    jobs = [(c, r, role, ctl) for c in range(len(CASES)) for role in ROLE_NAMES
            for r in range(args.boards) for ctl in (False, True)]
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        rows = list(ex.map(_run, jobs, chunksize=4))

    print(f"{args.boards} boards per case and role; 'ctl' = the associate swapped for a filler of the same role")
    print(f"{'case':34s} {'assoc as':9s} {'p(assoc)':>9s} {'ctl':>6s} {'its rank':>8s} {'p(members)':>10s} {'ctl':>6s} "
          f"{'P(3 mem)':>8s} {'ctl':>6s} {'value 3':>8s} {'ctl':>6s} {'shortlist':>9s} {'rank':>9s} {'ctl':>9s} {'gives it':>8s}")
    for c, (clue, members, assoc) in enumerate(CASES):
        for role in ROLE_NAMES:
            a = [r for r in rows if r["case"] == c and r["role"] == role and not r["control"]]
            b = [r for r in rows if r["case"] == c and r["role"] == role and r["control"]]
            m = lambda rs, k: np.mean([r[k] for r in rs])
            med = lambda rs: np.median([r["rank"] if r["rank"] else 999 for r in rs])
            gives = sum(r["top"][0][0] == clue for r in a)
            label = f"{clue} ({'/'.join(members)} | {assoc})"
            print(f"{label:34s} {role:9s} {m(a, 'p_assoc'):9.3f} {m(b, 'p_assoc'):6.3f} {m(a, 'assoc_rank'):8.1f} "
                  f"{m(a, 'p_members'):10.3f} {m(b, 'p_members'):6.3f} {m(a, 'pm3'):8.3f} {m(b, 'pm3'):6.3f} "
                  f"{m(a, 'value3'):8.2f} {m(b, 'value3'):6.2f} {sum(r['shortlisted'] for r in a):4d}/{len(a):<4d} "
                  f"{med(a):9.0f} {med(b):9.0f} {gives:5d}/{len(a)}")
    print("\nwhat the incumbent gives instead (associate neutral, first 3 boards per case):")
    for c, (clue, members, assoc) in enumerate(CASES):
        a = [r for r in rows if r["case"] == c and r["role"] == "neutral" and not r["control"]][:3]
        print(f"  {clue:10s} | {assoc:8s}: " + "   ".join(" ".join(f"{t[0]} {t[1]}" for t in r["top"][:2]) for r in a))

    if args.llm:
        from codenames.env import load_env
        from codenames.guessers.registry import build_guesser

        load_env()
        g = build_guesser("deepinfra:openai/gpt-oss-120b")
        items = []
        for c in range(len(CASES)):
            for r in range(args.llm):
                board, clue, members, fourth = build(c, r, "neutral", False)
                words = list(board.words)
                items.append((c, clue, members, fourth, words))
        with ThreadPoolExecutor(16) as ex:
            ranks = list(ex.map(lambda it: g.rank_candidates(it[1], it[4], None, number=3), items))
        print("\ngpt-oss (the eval suite's spec) on the same boards, associate neutral, told the number 3:")
        for (c, clue, members, fourth, _), rk in zip(items, ranks):
            pos = rk.index(fourth) + 1
            print(f"  {clue:10s} top 3 = members: {set(rk[:3]) == set(members)!s:5s}  "
                  f"{fourth} ranked {pos:2d}   top 4: {', '.join(rk[:4])}")


if __name__ == "__main__":
    main()

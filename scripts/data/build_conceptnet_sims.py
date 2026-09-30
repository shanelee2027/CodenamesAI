"""ConceptNet relations and compounds between every clue and every board word.

**Why, when numberbatch is already in the tensor.** Numberbatch is built FROM
ConceptNet, but as an embedding: it blurs every edge into a similarity. Two
things it cannot say sharply are:
- *which* relation holds (Porsche IsA sports car, wheel PartOf car); and
- whether clue and word form a **phrase**: Donald + Duck, fire + Fly
  ("firefly"), Marilyn + Monroe.

Phrases are how most of the clues WordNet does not know work: first names,
brands and abbreviations (docs/log.md, "Where the listener's unexplained
variance is": R² 0.335 on them). ConceptNet's 1.2M English terms include
Wiktionary's multi-word entries and many names.

Tables (clue rows x board-word columns), all from English-English edges of
ConceptNet 5.7 (`data/conceptnet/raw/en_edges.csv.gz`, filtered from the
public assertions dump, see the command below):
- `cn_isa`: the board word IsA the clue. 1 for a direct edge, 0.5 through one
  intermediate term. Kept apart from WordNet's `isa` because ConceptNet's
  IsA is crowd-sourced and noisy ("moon IsA planet" is in it); the trees
  decide how much to trust each.
- `cn_isa_rev`: the clue IsA the board word (porsche IsA car).
- `cn_part`: PartOf, HasA or MadeOf, either way round (wheel/car).
- `cn_typed`: the number of distinct specific relations joining them
  directly, either way (UsedFor, AtLocation, CapableOf, ...; everything but
  the generic and lexical ones in GENERIC).
- `cn_any`: log(1 + max weight) of any direct edge, generic ones included.
- `cmp_cw`, `cmp_wc`: "clue word"/"clueword" (resp. "word clue"/"wordclue")
  is a ConceptNet term: Donald Duck, firefly.

NaN where the clue or the word is not a ConceptNet term at all, so "no
evidence" never reads as "unrelated". The phrase tables need only the
PHRASE to exist, so they are 0/1 whenever both words are known.

    zcat data/conceptnet/raw/conceptnet-assertions-5.7.0.csv.gz \\
      | awk -F'\\t' '$3 ~ /^\\/c\\/en\\// && $4 ~ /^\\/c\\/en\\//' | gzip -1 > data/conceptnet/raw/en_edges.csv.gz
    python scripts/data/build_conceptnet_sims.py
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from codenames.clue_stats import ClueStats  # noqa: E402
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor  # noqa: E402

EDGES = PROJECT_ROOT / "data" / "conceptnet" / "raw" / "en_edges.csv.gz"
OUT = PROJECT_ROOT / "cache" / "conceptnet_sims.npz"
PART = {"/r/PartOf", "/r/HasA", "/r/MadeOf"}
# Relations that say "these are related" or "these are forms of one word"
# without saying how: counted by cn_any, not by cn_typed.
GENERIC = {"/r/RelatedTo", "/r/FormOf", "/r/DerivedFrom", "/r/EtymologicallyRelatedTo",
           "/r/EtymologicallyDerivedFrom", "/r/HasContext", "/r/Synonym", "/r/Antonym",
           "/r/DistinctFrom", "/r/SimilarTo", "/r/ExternalURL"}


def term(uri: str) -> str:
    return uri.split("/")[3].replace("_", " ")


def load(path: Path):
    terms: set[str] = set()
    isa: dict[str, set[str]] = defaultdict(set)
    part: dict[str, set[str]] = defaultdict(set)          # symmetric
    typed: dict[tuple[str, str], set[str]] = defaultdict(set)
    weight: dict[tuple[str, str], float] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            _, rel, s, e, meta = line.rstrip("\n").split("\t")
            a, b = term(s), term(e)
            terms.add(a)
            terms.add(b)
            if a == b:
                continue
            w = float(json.loads(meta).get("weight", 1.0))
            key = (a, b) if a < b else (b, a)
            weight[key] = max(weight.get(key, 0.0), w)
            if rel in ("/r/IsA", "/r/InstanceOf"):
                isa[a].add(b)
            if rel in PART:
                part[a].add(b)
                part[b].add(a)
            if rel not in GENERIC:
                typed[key].add(rel)
    return terms, isa, part, typed, weight


def isa_score(below: str, above: str, isa: dict) -> float:
    up = isa.get(below, ())
    if above in up:
        return 1.0
    return 0.5 if any(above in isa.get(x, ()) for x in up) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges", type=Path, default=EDGES)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-rarity", type=float, default=10.0)
    args = ap.parse_args()

    terms, isa, part, typed, weight = load(args.edges)
    print(f"{len(terms):,} English terms, {sum(map(len, isa.values())):,} IsA edges")
    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    board = sorted(sims.board_index, key=lambda w: sims.board_index[w])
    pool_rows = np.flatnonzero(stats.rarity_percentile <= args.max_rarity)
    clues = [stats.clue_words[i].lower() for i in pool_rows]
    bw = [w.lower().replace("-", " ") for w in board]

    names = ["cn_isa", "cn_isa_rev", "cn_part", "cn_typed", "cn_any", "cmp_cw", "cmp_wc"]
    T = {k: np.full((len(clues), len(board)), np.nan, dtype=np.float32) for k in names}
    for i, c in enumerate(clues):
        if c not in terms:
            continue
        for j, w in enumerate(bw):
            if w not in terms:
                continue
            key = (c, w) if c < w else (w, c)
            T["cn_isa"][i, j] = isa_score(w, c, isa)
            T["cn_isa_rev"][i, j] = isa_score(c, w, isa)
            T["cn_part"][i, j] = float(w in part.get(c, ()))
            T["cn_typed"][i, j] = len(typed.get(key, ()))
            T["cn_any"][i, j] = np.log1p(weight.get(key, 0.0))
            ws = w.replace(" ", "")
            T["cmp_cw"][i, j] = float(f"{c} {w}" in terms or c + ws in terms)
            T["cmp_wc"][i, j] = float(f"{w} {c}" in terms or ws + c in terms)
        if i % 2000 == 0:
            print(f"  {i:,}/{len(clues):,}", flush=True)

    known = np.isfinite(T["cn_any"])
    print(f"clues in ConceptNet {np.isfinite(T['cn_any']).any(1).mean():.1%}   "
          f"board words {np.isfinite(T['cn_any']).any(0).mean():.1%}")
    for k in names:
        print(f"  {k:10s} nonzero {np.mean(T[k][known] > 0):.3%}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **T, clue_rows=pool_rows.astype(np.int32), board_words=np.array(board))
    print(f"saved -> {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""Surface-form and definitional links between clues and board words.

Two axes nothing else in the feature set can see.

**Orthography.** Real Codenames clues work on word form: MICRO for MICROSCOPE
and MICROWAVE, BERRY for BLUEBERRY and STRAWBERRY, FIRE for FIREFLY. Every
other feature here reads a word as a semantic token and is blind to its
spelling. The one caveat is fastText, which is subword-based and may already
carry some of this -- which makes this a real test rather than a certainty.

**Glosses.** WordNet's taxonomy says how far apart two words sit in the is-a
tree. Its *definitions* say something different: FROST and ICE need not be
close in the hierarchy while each one's gloss mentions the other. Definitional
overlap is not hierarchy distance, and the data is already on disk.

Longest-common-substring was considered and dropped in favour of character
trigram overlap: the DP is O(n*m) per pair over 4.5M pairs, while trigram sets
precompute once per word and reduce each pair to a set intersection. It catches
the same cases -- MICRO/MICROSCOPE share every trigram of the shorter word.

Output `cache/lexical_sims.npz`, laid out like cache/extra_sims.npz.

Usage:
    python scripts/data/build_lexical_sims.py --workers 14
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from codenames.clue_stats import ClueStats
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

OUT = PROJECT_ROOT / "cache" / "lexical_sims.npz"
STOP = {
    "a", "an", "the", "of", "or", "and", "to", "in", "on", "for", "with", "by",
    "is", "are", "was", "be", "as", "that", "which", "it", "its", "from", "at",
    "any", "some", "one", "used", "esp", "especially", "usually", "often",
}
_D: dict = {}


def trigrams(w: str) -> set[str]:
    p = f"^{w}$"
    return {p[i : i + 3] for i in range(len(p) - 2)} or {p}


def gloss_words(word: str) -> set[str]:
    from nltk.corpus import wordnet as wn

    out: set[str] = set()
    for s in wn.synsets(word):
        for tok in s.definition().lower().replace(",", " ").replace(";", " ").split():
            t = tok.strip("().'\"-")
            if len(t) > 2 and t not in STOP:
                out.add(t)
    return out


def chunk(bounds: tuple[int, int]) -> tuple[int, np.ndarray]:
    lo, hi = bounds
    cl, bd = _D["clue"], _D["board"]
    nb = len(bd)
    out = np.zeros((hi - lo, nb, 7), dtype=np.float32)
    for i in range(lo, hi):
        cw, ctri, cglo = cl[i]
        for j in range(nb):
            bw, btri, bglo = bd[j]
            m = min(len(cw), len(bw))
            # containment, as a length ratio so MICRO/MICROSCOPE beats A/APPLE
            contains = (m / max(len(cw), len(bw))) if (cw in bw or bw in cw) else 0.0
            pre = 0
            while pre < m and cw[pre] == bw[pre]:
                pre += 1
            suf = 0
            while suf < m and cw[-1 - suf] == bw[-1 - suf]:
                suf += 1
            inter = len(ctri & btri)
            tri = inter / (len(ctri) + len(btri) - inter) if (ctri or btri) else 0.0
            gi = len(cglo & bglo)
            out[i - lo, j] = (
                contains, pre / m if m else 0.0, suf / m if m else 0.0, tri,
                1.0 if cw in bglo else 0.0,
                1.0 if bw in cglo else 0.0,
                gi / (len(cglo) + len(bglo) - gi) if (cglo or bglo) else 0.0,
            )
    return lo, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-rarity", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    board_words = sorted(sims.board_index, key=lambda w: sims.board_index[w])
    pool_rows = np.flatnonzero(stats.rarity_percentile <= args.max_rarity)
    pool_words = [stats.clue_words[i].lower() for i in pool_rows]

    print(f"precomputing trigrams and glosses for {len(pool_words):,} clues "
          f"+ {len(board_words)} board words...", flush=True)
    _D["board"] = [(w, trigrams(w), gloss_words(w)) for w in board_words]
    _D["clue"] = [(w, trigrams(w), gloss_words(w)) for w in pool_words]
    gl = sum(1 for _, _, g in _D["clue"] if g) / len(pool_words)
    print(f"  clues with a gloss: {gl:.1%}", flush=True)

    n = len(pool_words)
    step = max(1, n // (args.workers * 4))
    chunks = [(i, min(i + step, n)) for i in range(0, n, step)]
    all_ = np.zeros((n, len(board_words), 7), dtype=np.float32)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for lo, part in ex.map(chunk, chunks):
            all_[lo : lo + len(part)] = part
    print(f"scored {n*len(board_words):,} pairs in {time.time()-t0:.0f}s")

    names = ["orth_contains", "orth_prefix", "orth_suffix", "orth_trigram",
             "gloss_c_in_w", "gloss_w_in_c", "gloss_jaccard"]
    payload = {k: all_[:, :, i] for i, k in enumerate(names)}
    for k in names:
        print(f"  {k:16s} nonzero {np.mean(payload[k] > 0):6.2%}  max {payload[k].max():.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, clue_rows=pool_rows.astype(np.int32),
                        board_words=np.array(board_words), **payload)
    print(f"saved -> {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""Check that extending the board vocabulary left every existing feature alone.

scripts/data/extend_board_vocab.py appends board words after the base game's
400, and the side tables are then rebuilt by their own scripts. The rebuilt
tables are only safe if the old columns are exactly what they were: the
boosters were fitted on those values, and every recorded result was played
on them. This compares each table in cache/ against its copy in the backup
directory taken before the extension, array by array: the old board words'
slice of the new array must equal the old array (NaN equal to NaN), and
board-independent arrays must be unchanged. No API calls.

    python scripts/tools/check_board_extension.py --backup cache/backups/pre_duet_2026-10-30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE = PROJECT_ROOT / "cache"
TABLES = ["extra_sims.npz", "entity_sims.npz", "lexical_sims.npz", "isa_sims.npz", "conceptnet_sims.npz",
          "wordnet_sims.npz", "wordnet_senses.npz", "assoc_sims.npz", "word_norms.npz", "lm_pmi.npz",
          "word_stats.npz", "swow.npz", "assoc_profile.npz"]


def same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc":
        return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))
    return bool(np.array_equal(a, b))


def compare(old: np.ndarray, new: np.ndarray, n_old: int, n_new: int) -> str:
    """'same' or a description of the difference, comparing the old board
    words' slice when one axis is the board axis."""
    if old.shape == new.shape:
        return "same" if same(old, new) else "CHANGED"
    axes = [i for i, (a, b) in enumerate(zip(old.shape, new.shape)) if a != b]
    if old.ndim != new.ndim or len(axes) != 1 or old.shape[axes[0]] != n_old or new.shape[axes[0]] != n_new:
        return f"CHANGED shape {old.shape} -> {new.shape}"
    head = np.take(new, np.arange(n_old), axis=axes[0])
    if old.dtype.kind == "U":
        return "same" if np.array_equal(old, head) else "CHANGED"
    return "same" if same(old, head) else "CHANGED in the old columns"


def csr(d, prefix: str):
    import scipy.sparse as sp

    return sp.csr_matrix((d[f"{prefix}_data"], d[f"{prefix}_indices"], d[f"{prefix}_indptr"]),
                         shape=tuple(d[f"{prefix}_shape"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backup", type=Path, required=True)
    args = ap.parse_args()

    n_old = len(json.loads((args.backup / "board_vocab.json").read_text()))
    n_new = len(json.loads((CACHE / "board_vocab.json").read_text()))
    print(f"board words {n_old} -> {n_new}")
    t_old = np.load(args.backup / "similarity_tensor.npy", mmap_mode="r")
    t_new = np.load(CACHE / "similarity_tensor.npy", mmap_mode="r")
    print(f"similarity_tensor.npy: {compare(np.asarray(t_old), np.asarray(t_new), n_old, n_new)}")

    bad = 0
    for name in TABLES:
        o, n = np.load(args.backup / name, allow_pickle=False), np.load(CACHE / name, allow_pickle=False)
        if set(o.files) != set(n.files):
            print(f"{name}: KEYS CHANGED {sorted(set(o.files) ^ set(n.files))}")
            bad += 1
            continue
        out = {}
        prefixes = sorted({k.rsplit("_", 1)[0] for k in o.files if k.endswith("_indptr")})
        for p in prefixes:
            # SWOW's sparse tables: forward ones are clue x board, reverse ones
            # board x clue, so the board axis is whichever one grew.
            a, b = csr(o, p).toarray(), csr(n, p).toarray()
            out[p] = compare(a, b, n_old, n_new)
        for k in o.files:
            if any(k.startswith(p + "_") for p in prefixes):
                continue
            if name == "assoc_profile.npz" and k == "word_lists":
                lo, ln = json.loads(str(o[k])), json.loads(str(n[k]))
                out[k] = "same" if all(ln.get(w) == v for w, v in lo.items()) else "CHANGED"
                out[k] += f" ({len(lo)} -> {len(ln)} words)"
                continue
            out[k] = compare(o[k], n[k], n_old, n_new)
        bad += sum("CHANGED" in v for v in out.values())
        print(f"{name}: " + ", ".join(f"{k} {v}" for k, v in out.items()))
    print("\nall old values unchanged" if bad == 0 else f"\n{bad} arrays changed")


if __name__ == "__main__":
    main()

"""Two-hop association: how close each board word is to the words gpt-oss
associates with the clue, as a clue x board-word table.

Why (docs/log.md, 2026-11-06, "HONEYMOON"): every listener feature reads the
direct link from clue to board word. HONEYMOON reaches HEART only through
LOVE and HAWAII through VACATION; gpt-oss's own association lists for it name
neither board word, and the listener put HEART at 0.03 as the second pick
where gpt-oss took it 19 times in 30.

Source: cache/associations.db (scripts/data/collect_associations.py --pool),
5 lists per clue. The mediators are assoc_profile.mediators: words named in
at least 2 of the 5 lists, at most 10, forms of the clue left out. Each must
be in the similarity tensor's clue vocabulary.

- `med_<space>` for the tensor's three spaces: the board word's best z over
  the mediators, each mediator's similarity z-scored over the online game's
  800 words (cache/clue_stats_online.npz), as the online footing's z_* are.
  A mediator that is the board word itself (ROMANCE -> "heart" for HEART)
  is left out of that word's max: it would be the word's similarity to
  itself, and a direct hit is what assoc_share already measures. So the
  block is the second hop only.

NaN for a clue with no lists or no mediator in the tensor. Rows: the clue
pool at rarity <= --max-rarity, like the other tables. The mediators are
stored beside the table (`mediated_sims.mediators.json`), so a row can be
explained.

    python scripts/data/build_mediated_sims.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from codenames.assoc_profile import clue_forms, mediators  # noqa: E402
from codenames.clue_stats import ClueStats  # noqa: E402
from codenames.listener_training import ASSOCIATIONS, _assoc_norm  # noqa: E402
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor  # noqa: E402

OUT = PROJECT_ROOT / "cache" / "mediated_sims.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=ASSOCIATIONS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-rarity", type=float, default=10.0)
    args = ap.parse_args()

    lists: dict[str, list[list[str]]] = {}
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    for clue, words in conn.execute("SELECT clue, words FROM lists"):
        lst = [_assoc_norm(w) for w in json.loads(words)]
        if lst:
            lists.setdefault(clue.lower(), []).append(lst)
    conn.close()

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    pool = ClueStats.load(DEFAULT_CACHE_DIR)
    online = ClueStats.load(DEFAULT_CACHE_DIR, stem="clue_stats_online")
    board = sorted(sims.board_index, key=lambda w: sims.board_index[w])
    # A mediator is matched to the tensor's clue vocabulary by the lists' own
    # normalisation; the first spelling wins.
    tensor_of: dict[str, int] = {}
    for i, w in enumerate(sims.clue_words):
        tensor_of.setdefault(_assoc_norm(w), i)
    # The board columns each list word is a form of (plural or singular).
    columns_of: dict[str, list[int]] = {}
    for j, w in enumerate(board):
        for f in clue_forms(_assoc_norm(w)):
            columns_of.setdefault(f, []).append(j)
    pool_rows = np.flatnonzero(pool.rarity_percentile <= args.max_rarity)
    clues = [pool.clue_words[i].lower() for i in pool_rows]

    out = np.full((len(clues), len(board), len(sims.spaces)), np.nan, dtype=np.float32)
    used: list[list[str]] = []
    for i, c in enumerate(clues):
        med = [m for m in mediators(lists.get(c, []), c) if m in tensor_of]
        used.append(med)
        if not med:
            continue
        rows = np.array([tensor_of[m] for m in med])
        s = np.asarray(sims.tensor[rows], dtype=np.float32)                 # (mediators, board, spaces)
        sd = np.where(online.std[rows] > 0, online.std[rows], 1.0)
        z = (s - online.mean[rows][:, None, :]) / sd[:, None, :]
        for k, m in enumerate(med):
            z[k, columns_of.get(m, [])] = -np.inf
        best = z.max(axis=0)
        out[i] = np.where(np.isfinite(best), best, np.nan)
    n_med = np.array([len(m) for m in used])
    print(f"{len(clues):,} clues: {np.mean(n_med > 0):.1%} with a mediator, mean {n_med.mean():.1f} "
          f"(median {np.median(n_med):.0f})")
    np.savez_compressed(args.out, **{f"med_{s}": out[:, :, k] for k, s in enumerate(sims.spaces)},
                        clue_rows=pool_rows.astype(np.int32), board_words=np.array(board))
    args.out.with_suffix(".mediators.json").write_text(json.dumps(dict(zip(clues, used))))
    print(f"saved -> {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""Raw word vectors for the words a listener can ever see, in five spaces.

Every other table in cache/ stores *similarities* (clue x board word). A
neural listener that learns its own similarity (codenames/listener_net.py)
needs the vectors themselves, so this pulls them out of the downloaded files
once, for:

- every clue in the pool (`rarity_percentile <= --max-rarity`, as
  build_extra_space_sims.py uses),
- every clue any cached LLM response was given (a few bad clues sit outside
  the pool),
- the 400 board words.

That is about 12k words instead of the files' millions, so the output is
small enough to load whole.

**Lookups match the existing tables.** Board words go through
`resolve_board_vector` (native multi-word token, else the mean of the parts),
as the similarity tensor does. glove840 and fastText are cased, and the first
spelling in file order wins after lower-casing, as in build_extra_space_sims.py.

**Stored L2-normalised, float16.** Only direction is used. A vector's norm
tracks word frequency in several of these spaces, and frequency is already a
listener feature (`log_freq`), so dropping it removes a second route by which
a model could learn which particular words these are.

Output `cache/word_vectors.npz`: `words` (lower-case), `is_board`, and per
space `<space>` (n, 300) float16 plus `<space>_ok` (n,) bool.

Usage:
    python scripts/data/build_word_vectors.py
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _embedding_lib import SPACE_CONFIGS, resolve_board_vector, split_words  # noqa: E402

from codenames.clue_stats import ClueStats  # noqa: E402
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor  # noqa: E402

RAW = PROJECT_ROOT / "data" / "embeddings" / "raw"
OUT = PROJECT_ROOT / "cache" / "word_vectors.npz"
DB = PROJECT_ROOT / "cache" / "llm_store.db"
SPACES = ("glove", "numberbatch", "wikipedia2vec", "glove840", "fasttext")
ZIPPED = {
    "glove840": (RAW / "glove.840B.300d.zip", "glove.840B.300d.txt"),
    "fasttext": (RAW / "crawl-300d-2M.vec.zip", "crawl-300d-2M.vec"),
}


def scan(space: str, wanted: set[str]) -> dict[str, np.ndarray]:
    """{token: vector} for the wanted tokens, in one streaming pass."""
    out: dict[str, np.ndarray] = {}
    if space in ZIPPED:
        path, member = ZIPPED[space]
        with zipfile.ZipFile(path) as zf, zf.open(member) as fh:
            for raw in fh:
                head, _, rest = raw.decode("utf-8", "replace").partition(" ")
                w = head.lower()
                if w in wanted and w not in out:
                    v = np.fromstring(rest, dtype=np.float32, sep=" ")
                    if v.size == 300:
                        out[w] = v
        return out
    cfg = SPACE_CONFIGS[space]
    with cfg["opener"](PROJECT_ROOT / cfg["default_source"]) as f:
        if cfg["has_header"]:
            next(f)
        for line in f:
            tok, _, rest = line.partition(" ")
            if cfg["skip_prefixes"] and tok.startswith(cfg["skip_prefixes"]):
                continue
            if tok in wanted and tok not in out:
                out[tok] = np.fromstring(rest, dtype=np.float32, sep=" ")
    return out


def matrix(space: str, words: list[str], is_board: np.ndarray, wanted: set[str]) -> tuple[str, np.ndarray, np.ndarray]:
    vecs = scan(space, wanted)
    join = SPACE_CONFIGS[space]["multiword_join"] if space in SPACE_CONFIGS else None
    M = np.zeros((len(words), 300), dtype=np.float32)
    ok = np.zeros(len(words), dtype=bool)
    for i, w in enumerate(words):
        v = resolve_board_vector(w, vecs, join) if is_board[i] else vecs.get(w)
        if v is not None and np.linalg.norm(v) > 0:
            M[i], ok[i] = v / np.linalg.norm(v), True
    print(f"  {space}: clues {ok[~is_board].sum():,}/{(~is_board).sum():,}   "
          f"board {ok[is_board].sum()}/{is_board.sum()}", flush=True)
    return space, M.astype(np.float16), ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-rarity", type=float, default=10.0)
    args = ap.parse_args()

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    vocab = {w.lower() for w in stats.clue_words}
    clues = {stats.clue_words[i].lower() for i in np.flatnonzero(stats.rarity_percentile <= args.max_rarity)}
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    clues |= {c.lower() for (c,) in conn.execute("SELECT DISTINCT clue FROM responses")} & vocab
    conn.close()
    board = sorted(sims.board_index, key=lambda w: sims.board_index[w])
    clues -= set(board)
    words = sorted(clues) + board
    is_board = np.array([False] * len(clues) + [True] * len(board))
    wanted = set(words) | {p for w in board for p in split_words(w)} | {w.replace(" ", "_") for w in board}
    print(f"{len(clues):,} clue words + {len(board)} board words; scanning {len(SPACES)} files in parallel", flush=True)

    payload: dict[str, np.ndarray] = {"words": np.array(words), "is_board": is_board}
    with ProcessPoolExecutor(len(SPACES)) as ex:
        for space, M, ok in ex.map(matrix, SPACES, [words] * len(SPACES), [is_board] * len(SPACES),
                                   [wanted] * len(SPACES)):
            payload[space], payload[f"{space}_ok"] = M, ok
    np.savez_compressed(args.out, **payload)
    print(f"saved -> {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

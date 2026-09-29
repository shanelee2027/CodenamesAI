"""Build the clue policy's inputs: raw evidence for every (pool clue, board
word) pair, plus the per-word legality table (codenames/clue_policy.py).

    python scripts/data/build_policy_features.py

Writes
    cache/policy_features.npy       (400, n_pool, F) float16, word-major
    cache/policy_features_aux.npz   pool, per-word features, legality, names,
                                    and the standardisation used

**The pool** is the incumbent's: rarity percentile <= 10 and not an acronym,
10,674 clues. Every side table covers exactly the rarity-10 rows, so every
pool clue has a row in each.

**Transforms.** Each column is standardised over the whole table, and a
missing value becomes 0 afterwards, with a flag column beside it where
missingness is common enough to carry information:

- z-scores: the tensor's three spaces (per-clue mean and sd, as
  expected_words uses them), glove840 and fasttext (z-scored at build time).
  One flag for the 1% of pairs the extra spaces lack.
- SWOW: a sparse table where an absent entry in a present row means no
  recorded association, so it is log(x + 1e-5) - log(1e-5), zero when absent.
  A flag per direction says whether the clue has a row at all (59% forward,
  94% reverse); a missing row is no evidence, not no association.
- Entity similarity: 82% missing, so value and flag.
- PMI, and WordNet (Wu-Palmer, LCS depth; 7% missing, flagged).
- The seven spelling and gloss overlaps are left in [0, 1]. They are rare
  enough that standardising would put a full overlap ~60 sd out.

Standardised columns are clipped at +-8 sd.

Per word: the four Brysbaert norms with a missing flag (8.5% of words), the
sense count, and the word's mean and sd of similarity over the whole clue
vocabulary in each tensor space.
"""

from __future__ import annotations

import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp

from codenames.acronyms import load_acronym_mask
from codenames.board import is_legal_clue
from codenames.clue_policy import AUX_FILE, FEATURES_FILE
from codenames.clue_stats import ClueStats
from codenames.listener_features import WordStats
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

CACHE = DEFAULT_CACHE_DIR
MAX_RARITY = 10.0
SWOW_FLOOR = 1e-5
CLIP = 8.0


def _side(name: str, keys: list[str], pool: np.ndarray, board_words: list[str]) -> dict[str, np.ndarray]:
    """(n_pool, 400) arrays from a side table, rows and columns re-indexed to
    the pool and to the tensor's board-word order."""
    d = np.load(CACHE / name, allow_pickle=False)
    row_of = {int(c): i for i, c in enumerate(d["clue_rows"])}
    pos = {str(w).lower(): i for i, w in enumerate(d["board_words"])}
    rows = np.array([row_of[int(c)] for c in pool])
    cols = np.array([pos[w.lower()] for w in board_words])
    return {k: d[k][rows][:, cols].astype(np.float32) for k in keys}


def _swow(pool: np.ndarray, board_words: list[str]) -> dict[str, np.ndarray]:
    d = np.load(CACHE / "swow.npz", allow_pickle=False)
    pos = {str(w).lower(): i for i, w in enumerate(d["board_words"])}
    cols = np.array([pos[w.lower()] for w in board_words])
    out = {}
    for key in ("one", "two", "rone", "rtwo"):
        m = sp.csr_matrix((d[f"{key}_data"], d[f"{key}_indices"], d[f"{key}_indptr"]),
                          shape=tuple(d[f"{key}_shape"]))[pool]
        out[key] = np.asarray(m[:, cols].todense(), dtype=np.float32)
        out[key + "_row"] = (np.diff(m.indptr) > 0)
    return out


def _illegal_for_word(args) -> np.ndarray:
    word, clues = args
    return np.array([not is_legal_clue(c, [word]) for c in clues], dtype=bool)


def main() -> None:
    t0 = time.time()
    sims = SimilarityTensor.load(CACHE)
    stats = ClueStats.load(CACHE)
    acronym = load_acronym_mask(CACHE, stats.clue_words)
    pool = np.flatnonzero((stats.rarity_percentile <= MAX_RARITY) & ~acronym)
    board_words = sims.board_words
    n_pool, n_words = len(pool), len(board_words)
    print(f"pool {n_pool} clues x {n_words} board words")

    cols: dict[str, np.ndarray] = {}                      # name -> (n_pool, 400)
    raw = np.asarray(sims.tensor[pool], dtype=np.float32)  # (n_pool, 400, 3)
    z = (raw - stats.mean[pool][:, None, :]) / stats.std[pool][:, None, :]
    for s, name in enumerate(("glove", "numberbatch", "wiki2vec")):
        cols[f"z_{name}"] = z[:, :, s]
    extra = _side("extra_sims.npz", ["glove840", "fasttext"], pool, board_words)
    cols["z_glove840"], cols["z_fasttext"] = extra["glove840"], extra["fasttext"]
    cols["extra_missing"] = (np.isnan(extra["glove840"]) | np.isnan(extra["fasttext"])).astype(np.float32)

    sw = _swow(pool, board_words)
    for key, name in (("one", "swow_fwd1"), ("two", "swow_fwd2"), ("rone", "swow_rev1"), ("rtwo", "swow_rev2")):
        cols[name] = np.log(sw[key] + SWOW_FLOOR) - np.log(SWOW_FLOOR)
    cols["swow_fwd_row"] = np.repeat(sw["two_row"][:, None], n_words, axis=1).astype(np.float32)
    cols["swow_rev_row"] = np.repeat(sw["rtwo_row"][:, None], n_words, axis=1).astype(np.float32)

    ent = _side("entity_sims.npz", ["sims"], pool, board_words)["sims"]
    cols["entity"], cols["entity_has"] = ent, (~np.isnan(ent)).astype(np.float32)
    cols["pmi"] = _side("lm_pmi.npz", ["sims"], pool, board_words)["sims"]
    wn = _side("wordnet_sims.npz", ["wup", "lcs_depth"], pool, board_words)
    cols["wn_wup"], cols["wn_lcs_depth"] = wn["wup"], wn["lcs_depth"]
    cols["wn_missing"] = np.isnan(wn["wup"]).astype(np.float32)
    lex_keys = ["orth_contains", "orth_prefix", "orth_suffix", "orth_trigram",
                "gloss_c_in_w", "gloss_w_in_c", "gloss_jaccard"]
    cols.update(_side("lexical_sims.npz", lex_keys, pool, board_words))

    names = list(cols)
    F = len(names)
    means, sds = np.zeros(F, np.float32), np.ones(F, np.float32)
    out = np.lib.format.open_memmap(CACHE / FEATURES_FILE, mode="w+", dtype=np.float16,
                                    shape=(n_words, n_pool, F))
    for f, name in enumerate(names):
        a = cols[name]
        # Flags, and the spelling/gloss overlaps (rare values in [0, 1]:
        # standardised, a full overlap would sit ~60 sd out), are left as they are.
        if not name.endswith(("_missing", "_has", "_row")) and name not in lex_keys:
            ok = np.isfinite(a)
            means[f], sds[f] = float(a[ok].mean()), float(a[ok].std()) or 1.0
            a = np.clip(np.where(ok, (a - means[f]) / sds[f], 0.0), -CLIP, CLIP)
        out[:, :, f] = a.T.astype(np.float16)
        print(f"  {name:14s} mean {means[f]:+.3f} sd {sds[f]:.3f}")
    out.flush()
    del out

    # Per word.
    norms_d = np.load(CACHE / "word_norms.npz", allow_pickle=False)
    npos = {str(w).lower(): i for i, w in enumerate(norms_d["board_words"])}
    norms = norms_d["norms"][[npos[w.lower()] for w in board_words]].astype(np.float32)
    wstats = WordStats.load(CACHE / "word_stats.npz")
    wpos = {w.lower(): i for i, w in enumerate(wstats.board_words)}
    wi = [wpos[w.lower()] for w in board_words]
    word_cols = {"conc": norms[:, 0], "conc_sd": norms[:, 1], "pct_known": norms[:, 2],
                 "log_freq": norms[:, 3], "n_senses": norms[:, 4],
                 "norms_missing": np.isnan(norms[:, 0]).astype(np.float32)}
    for s, name in enumerate(("glove", "numberbatch", "wiki2vec")):
        word_cols[f"mean_sim_{name}"] = wstats.mean[wi, s].astype(np.float32)
        word_cols[f"sd_sim_{name}"] = wstats.sd[wi, s].astype(np.float32)
    word = np.zeros((n_words, len(word_cols)), np.float32)
    for f, (name, a) in enumerate(word_cols.items()):
        if name.endswith("_missing"):
            word[:, f] = a
            continue
        ok = np.isfinite(a)
        word[:, f] = np.where(ok, (a - a[ok].mean()) / (a[ok].std() or 1.0), 0.0)

    # Legality, word by word (is_legal_clue fails a clue if any one word does).
    clue_words = [sims.clue_words[c] for c in pool]
    with Pool(16) as p:
        illegal = np.stack(p.map(_illegal_for_word, [(w, clue_words) for w in board_words]))
    print(f"illegal pairs: {illegal.sum()} ({illegal.mean():.4%})")

    np.savez(CACHE / AUX_FILE, pool=pool, clue_words=np.array(clue_words), board_words=np.array(board_words),
             word=word, illegal=illegal, pair_names=np.array(names), word_names=np.array(list(word_cols)),
             pair_mean=means, pair_sd=sds)
    print(f"done in {time.time() - t0:.0f}s: {F} pair features, {word.shape[1]} word features")


if __name__ == "__main__":
    main()

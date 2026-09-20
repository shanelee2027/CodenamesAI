"""Per-board raw inputs, so a browser can rebuild features after every reveal.

The artifact currently ships one GBT scoring of the full board and renormalises
for the rest of the game; its error grows as the board empties (91.8% top-1
agreement with 18-20 words left, 71.7% with 4-6 -- docs/log.md). The deployed
spymaster has no such problem because it re-extracts every turn. To do the same
in a browser, ship the inputs rather than the outputs.

Seventeen of the 44 features depend on which words remain, and every one of
them is derivable from values that do not:

  per (clue, word), 19 values  z_glove z_nb z_wiki | swow1 swow2 swow_rev1
                               swow_rev2 | ent | pmi | g840 ft | wn_wup |
                               orth_contains orth_prefix orth_suffix
                               orth_trigram gloss_c_in_w gloss_w_in_c
                               gloss_jaccard
  per word, 7 values           word_mean_sim word_sd_sim conc conc_sd
                               pct_known log_freq n_senses
  per board, 25x25             the cohesion z-matrix: row t is candidate t
                               treated as a clue, standardised the same way
  per word, 1 clue word        the k=1 substitution (see below)

Everything else -- gaptop, ranks, peak_z, lead_margin, n_candidates,
p_max_sigma2, cohesion, the SWOW shares and asymmetry -- is computed in the
browser from whichever words are still standing.

Usage:
    python scripts/tools/export_board_inputs.py --out app/boards.json --n 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.board import Board, OpponentBoardView, Role
from codenames.clue_search import is_legal_clue
from codenames.clue_stats import ClueStats
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor
from codenames.spymasters.learned_listener import LearnedListenerSpymaster

ROLE = {Role.OWN: "a", Role.OPPONENT: "b", Role.NEUTRAL: "n", Role.ASSASSIN: "x"}
TARGET = 1e8        # integers carry ~8 significant digits after scaling


def scale_for(a):
    """One multiplier per column, so each keeps ~6 significant digits.

    A single fixed-point scale does not work here: the columns span many orders
    of magnitude. Two-hop SWOW strengths are sums of products of probabilities
    and are routinely below 1e-3, so a 1/1000 grid rounded them to zero, which
    silently emptied the denominator of `swow_rev2_share` and put `swow_asym`
    out by 0.5. Measured, not guessed -- the port's golden test found it.
    """
    m = np.nanmax(np.abs(np.asarray(a, dtype=np.float64)))
    if not np.isfinite(m) or m == 0:
        return 1.0
    return float(TARGET / m)


def q(a, sc):
    """Round to 1/sc and emit ints, NaN as null -- NaN is meaningful here (no
    SWOW row, no entity vector) and must survive the round trip."""
    out = []
    for v in np.asarray(a, dtype=np.float64).ravel():
        out.append(None if not np.isfinite(v) else int(round(v * sc)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--first-seed", type=int, default=101)
    ap.add_argument("--top", type=int, default=260, help="Gaussian shortlist per side")
    args = ap.parse_args()

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    sm = LearnedListenerSpymaster()
    b = sm.bundle
    ci_of = sm._clue_index
    boards = []

    for t in range(args.n):
        seed = args.first_seed + 7 * t
        board = Board.generate(seed=seed)
        words = [c.word for c in board.cards]
        wl = [w.lower() for w in words]
        idxs = [sims.board_index[w] for w in wl]

        pool = set()
        for view in (board, OpponentBoardView(board)):
            _, gs, _ = sm._first_stage._score_all_clues(view, sims)
            ok = np.flatnonzero(np.isfinite(gs))
            pool.update(ok[np.argsort(-gs[ok])[: args.top]].tolist())
        cand = sorted(i for i in pool if is_legal_clue(sims.clue_words[i], words))

        # ---- per (clue, word) ------------------------------------------------
        rows, keep = [], []
        bcols_sw = np.array([b.swow.board_pos.get(w, -1) for w in wl])
        ecols = np.array([b.entity.board_pos.get(w, -1) for w in wl])
        pcols = np.array([b.pmi.board_pos.get(w, -1) for w in wl])
        xcols = np.array([b.extra.board_pos.get(w, -1) for w in wl])
        ncols = np.array([b.wordnet.board_pos.get(w, -1) for w in wl])
        lcols = np.array([b.lexical.board_pos.get(w, -1) for w in wl])
        for ci in cand:
            sim = np.asarray(sims.tensor[ci, idxs, :], dtype=np.float64)
            mu = np.asarray(stats.mean[ci], dtype=np.float64)
            sd = np.asarray(stats.std[ci], dtype=np.float64)
            if not np.all(np.isfinite(sd)) or np.any(sd <= 0) or not np.all(np.isfinite(sim)):
                continue
            z = (sim - mu) / sd                                   # (25, 3)
            vals = [z[:, 0], z[:, 1], z[:, 2],
                    b.swow.row(1, ci, bcols_sw), b.swow.row(2, ci, bcols_sw),
                    b.swow.row("r1", ci, bcols_sw), b.swow.row("r2", ci, bcols_sw),
                    b.entity.row(ci, ecols), b.pmi.row(ci, pcols),
                    b.extra.row("glove840", ci, xcols), b.extra.row("fasttext", ci, xcols),
                    b.wordnet.row("wup", ci, ncols)]
            for key in ("orth_contains", "orth_prefix", "orth_suffix", "orth_trigram",
                        "gloss_c_in_w", "gloss_w_in_c", "gloss_jaccard"):
                vals.append(b.lexical.row(key, ci, lcols))
            rows.append(np.stack(vals, axis=1))                   # 25 x 19
            keep.append(ci)

        stack = np.stack(rows) if rows else np.zeros((0, len(words), 19))
        pair_scales = [scale_for(stack[:, :, j]) for j in range(stack.shape[2])]
        flat = []
        for r in stack:                                            # 25 x 19
            row = []
            for wi_ in range(r.shape[0]):
                for j in range(r.shape[1]):
                    v = r[wi_, j]
                    row.append(None if not np.isfinite(v) else int(round(v * pair_scales[j])))
            flat.append(row)
        rows = flat

        # ---- per word --------------------------------------------------------
        wi = np.asarray(idxs)
        nv = b.norms.rows(words)
        per_word = np.stack([np.asarray(b.word_stats.mean[wi, 1], dtype=np.float64),
                             np.asarray(b.word_stats.sd[wi, 1], dtype=np.float64),
                             nv[:, 0], nv[:, 1], nv[:, 2], nv[:, 3], nv[:, 4]], axis=1)

        # ---- cohesion matrix: candidate t as a clue, against every candidate --
        coh = np.full((len(words), len(words)), np.nan)
        for ti, w in enumerate(wl):
            cj = ci_of.get(w)
            if cj is None:
                continue
            s = float(stats.std[cj, 1])
            if not np.isfinite(s) or s <= 0:
                continue
            coh[ti] = (np.asarray(sims.tensor[cj, idxs, 1], dtype=np.float64)
                       - float(stats.mean[cj, 1])) / s

        # ---- k=1 substitution table -----------------------------------------
        # `LearnedListenerSpymaster._swap_k1` scans the *whole* clue pool by raw
        # numberbatch cosine, not the Gaussian shortlist, so the browser cannot
        # reproduce it from `cand`. It does not have to: legality is fixed by
        # the board, so the rule's answer is one clue word per board word and is
        # resolved exactly here. The substituted clue is only ever displayed --
        # a human guesses it -- so it needs no feature row.
        pool = np.flatnonzero(stats.rarity_percentile <= sm.max_rarity)
        k1 = []
        for ti in idxs:
            cos = np.asarray(sims.tensor[pool, ti, 1], dtype=np.float64)
            pick = None
            for r in np.argsort(-cos):
                if not np.isfinite(cos[r]):
                    continue
                cw = sims.clue_words[int(pool[r])]
                if is_legal_clue(cw, words):
                    pick = cw
                    break
            k1.append(pick)

        word_scales = [scale_for(per_word[:, j]) for j in range(per_word.shape[1])]
        word_flat = []
        for wi_ in range(per_word.shape[0]):
            for j in range(per_word.shape[1]):
                v = per_word[wi_, j]
                word_flat.append(None if not np.isfinite(v) else int(round(v * word_scales[j])))
        coh_scale = scale_for(coh)
        boards.append({"seed": seed, "words": words,
                       "roles": [ROLE[c.role] for c in board.cards],
                       "clues": [sims.clue_words[c] for c in keep],
                       "pair": rows, "word": word_flat, "coh": q(coh, coh_scale), "k1": k1,
                       "ps": pair_scales, "ws": word_scales, "cs": coh_scale})
        print(f"  seed {seed}: {len(keep)} clues", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"boards": boards}, separators=(",", ":")))
    print(f"{len(boards)} boards -> {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

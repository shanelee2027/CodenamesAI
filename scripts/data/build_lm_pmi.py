"""Pointwise mutual information between every clue and every board word, on GPU.

**Why PMI and not another embedding.** The measured pattern in docs/log.md is
that another view of distributional similarity buys nothing, while a different
*kind* of evidence buys a lot: SWOW forward +0.0139 nats, SWOW reverse +0.0294.
SWOW is human free-association -- which words come to mind together -- and its
weakness is coverage: 12,217 cues, so 41% of our clue pool has no row at all.

PMI from a language model measures the same syntagmatic axis from a corpus
instead of from people, and it covers every word in the vocabulary. It is the
dense stand-in for the sparse graph.

    PMI(clue, word) = log P(word | "The word <clue> reminds me of the word")
                    - log P(word | "The word reminds me of the word")

**Why PMI rather than the raw conditional.** The conditional alone is dominated
by word frequency and tokenisation: "platypus" is " plat" + "ypus", and the
second token is nearly certain given the first, so a length-normalised score
ranks it above "camera" as an associate of "nikon". Summing rather than
averaging keeps that bias, but the unconditional term carries exactly the same
bias and it cancels in the difference. Measured on a probe set, the raw
conditional got king/queen below king/stapler; PMI puts every related pair
above every unrelated one:

    doctor/nurse  5.62   king/queen 5.14   beach/sand 4.30   nikon/olympus 3.41
    schmidt/table 0.79   nikon/platypus 0.53   king/stapler -0.61

It even recovers schmidt -> scorpion (2.19 vs 0.79 for schmidt -> table), the
encyclopedic link the entity vectors were added for and largely failed to give.

**Cross-encoders were tried first and rejected.** cross-encoder/stsb-roberta-large
scores bare word pairs in a 0.009-0.075 band and ranks schmidt/table above
schmidt/scorpion: STS is trained for paraphrase similarity between sentences,
which is the wrong objective and out of distribution on single words. Sentence
templating widened the range but not the discrimination.

Output `cache/lm_pmi.npz`, laid out like cache/entity_sims.npz so the feature
extractor treats it the same way.

Usage:
    python scripts/data/build_lm_pmi.py --model gpt2-large
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codenames.clue_stats import ClueStats
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor

OUT = PROJECT_ROOT / "cache" / "lm_pmi.npz"

# Several prompts, averaged. One template is one arbitrary way of asking, and
# on a 14-pair probe no single template cleanly separates related from
# unrelated pairs -- each gets some hard pair backwards, but a different one.
# Averaging the PMI across templates cancels the prompt-specific part and keeps
# what they agree on. These three had the widest mean gap between related and
# unrelated probe pairs; "{} and" and "{} makes me think of" were measured and
# dropped (gaps 1.67 and 0.70 against 3.1-3.8 for these).
#
# Each template carries its own null, which is the same sentence with the cue
# word removed. The null need not be graceful English -- it only has to put the
# target word in the same syntactic slot, so that the frequency and
# tokenisation bias in the conditional cancels exactly.
TEMPLATES = [
    ("Things related to {}:", "Things related to:"),
    ("The word {} reminds me of the word", "The word reminds me of the word"),
    ("Word association. {} ->", "Word association. ->"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--model", default="gpt2-large")
    ap.add_argument("--max-rarity", type=float, default=10.0)
    ap.add_argument("--limit-clues", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sims = SimilarityTensor.load(DEFAULT_CACHE_DIR)
    stats = ClueStats.load(DEFAULT_CACHE_DIR)
    board_words = sorted(sims.board_index, key=lambda w: sims.board_index[w])
    pool_rows = np.flatnonzero(stats.rarity_percentile <= args.max_rarity)
    if args.limit_clues:
        pool_rows = pool_rows[: args.limit_clues]
    pool_words = [stats.clue_words[i].lower() for i in pool_rows]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    print(f"{args.model} on {dev}: {len(pool_words):,} clues x {len(board_words)} words", flush=True)

    # Board words tokenised once, right-padded to a common length. Every
    # sequence shares the same prompt, so the word tokens always start at the
    # same index and the gather below is a fixed slice.
    wtok = [tok(" " + w, add_special_tokens=False).input_ids for w in board_words]
    wlen = torch.tensor([len(t) for t in wtok], device=dev)
    lmax = int(wlen.max())
    pad = tok.eos_token_id or 0
    wpad = torch.tensor([t + [pad] * (lmax - len(t)) for t in wtok], device=dev)
    nb = len(board_words)

    @torch.no_grad()
    def score(prompt: str) -> np.ndarray:
        """Summed log P(word tokens | prompt) for every board word."""
        pids = torch.tensor(tok(prompt, add_special_tokens=False).input_ids, device=dev)
        plen = len(pids)
        ids = torch.cat([pids.expand(nb, plen), wpad], dim=1)
        mask = torch.arange(lmax, device=dev)[None, :] < wlen[:, None]
        attn = torch.cat([torch.ones(nb, plen, device=dev, dtype=torch.long), mask.long()], dim=1)
        logits = mdl(ids, attention_mask=attn).logits[:, plen - 1 : -1].float()
        lp = F.log_softmax(logits, dim=-1).gather(2, wpad[:, :, None]).squeeze(2)
        return (lp * mask).sum(dim=1).cpu().numpy()

    out = np.zeros((len(pool_words), nb), dtype=np.float32)
    t0 = time.time()
    for ti, (cue, null) in enumerate(TEMPLATES, 1):
        uncond = score(null)
        for i, clue in enumerate(pool_words):
            out[i] += (score(cue.format(clue)) - uncond) / len(TEMPLATES)
            if (i + 1) % 2000 == 0:
                done = (ti - 1) * len(pool_words) + i + 1
                total = len(TEMPLATES) * len(pool_words)
                rate = done / (time.time() - t0)
                print(f"  template {ti}/{len(TEMPLATES)}  {i+1:,}/{len(pool_words):,}  "
                      f"eta {(total-done)/rate/60:.1f} min", flush=True)

    print(f"done in {(time.time()-t0)/60:.1f} min   PMI range "
          f"[{out.min():.2f}, {out.max():.2f}]  mean {out.mean():.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, sims=out, clue_rows=pool_rows.astype(np.int32),
                        board_words=np.array(board_words))
    print(f"saved -> {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

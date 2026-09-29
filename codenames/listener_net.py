"""A neural listener: which unrevealed word a guesser picks for a clue.

Same job and same loss as the LightGBM listener (codenames/listener_training.py):
one score per candidate, a softmax over the candidates, the teacher's pick as
the target. Two things the trees cannot do are added.

**Raw embeddings, used relationally.** For each embedding space s the clue and
every candidate are projected by learned maps, and their dot product is a
learned similarity, sim_sh(c, w) = <A_sh c, B_sh w> / sqrt(r), for H low-rank
heads per space. It learns which directions of each space predict a guesser's
pick (topical versus grammatical versus phrase-like). A word's vector is never
fed in on its own: the vocabulary of training board words is 250, and a
network given the vector itself could learn which of those 250 get picked,
which is no use on the 150 held-out words (docs/design-decisions.md,
"Held-out board words"). The vectors are frozen; only the projections learn.

**The rest of the board.** Candidates attend to one another, with an attention
bias from their pairwise cosine in each space. So the score of a word can
depend on what else is on the board: "this fits, but that fits better". The
trees score each word from its own row, and board context enters them only
through hand-built columns. That is why the GBT listener violates IIA only by
accident and cannot model it (removing one word flips its favourite 20% of the
time, docs/log.md). Here a later pick is scored by re-running the network on
the words that remain, so the choice among the rest can change.

**Optional residual.** With `residual=True` the output layer starts at zero and
the score is `base + correction`, where `base` is the GBT's logit. At
initialisation the network is exactly the GBT, and training measures what the
new inputs add, the same device as the win critic's board correction.

Input per candidate: its 44 listener features (standardised), the learned
similarities, the plain cosines per space, and both again minus the best
available candidate's value (the analogue of the `gaptop_*` columns).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORD_VECTORS = PROJECT_ROOT / "cache" / "word_vectors.npz"
SPACES = ("glove", "numberbatch", "wikipedia2vec", "glove840", "fasttext")


class WordVectors:
    """cache/word_vectors.npz (scripts/data/build_word_vectors.py) as one
    (n_words, n_spaces, 300) float16 array plus an index. A word missing from
    a space has a zero vector there, so every similarity with it is 0."""

    def __init__(self, path: Path = WORD_VECTORS):
        z = np.load(path)
        self.words = [str(w) for w in z["words"]]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.vecs = np.stack([z[s] for s in SPACES], axis=1)
        self.ok = np.stack([z[f"{s}_ok"] for s in SPACES], axis=1)

    def rows(self, words: list[str]) -> list[int] | None:
        try:
            return [self.index[w.lower()] for w in words]
        except KeyError:
            return None


class _Layer(nn.Module):
    """Pre-norm transformer layer whose attention logits get an additive
    per-head bias from the pairwise word-word cosines."""

    def __init__(self, d: int, heads: int, n_edge: int, dropout: float):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.edge = nn.Linear(n_edge, heads)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edges: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        B, N, d = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(B, N, 3, self.h, self.dk).unbind(2)
        att = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.dk)
        att = att + self.edge(edges).permute(0, 3, 1, 2)
        att = att.masked_fill(~avail[:, None, None, :], float("-inf"))
        a = self.drop(att.softmax(-1))
        h = h + self.drop(self.o(torch.einsum("bhij,bjhd->bihd", a, v).reshape(B, N, d)))
        return h + self.drop(self.ff(self.ln2(h)))


class ListenerNet(nn.Module):
    def __init__(self, vecs: np.ndarray, n_feat: int, d: int = 64, layers: int = 2, heads: int = 4,
                 sim_heads: int = 4, rank: int = 16, dropout: float = 0.1, use_vectors: bool = True,
                 residual: bool = False):
        super().__init__()
        self.register_buffer("vecs", torch.as_tensor(vecs), persistent=False)
        S, D = vecs.shape[1], vecs.shape[2]
        self.S, self.H, self.r = S, sim_heads, rank
        self.use_vectors, self.residual = use_vectors, residual
        self.config = dict(n_feat=n_feat, d=d, layers=layers, heads=heads, sim_heads=sim_heads, rank=rank,
                           dropout=dropout, use_vectors=use_vectors, residual=residual)
        n_sim = 2 * (S * sim_heads + S) if use_vectors else 0
        if use_vectors:
            self.A = nn.Parameter(torch.randn(S, D, sim_heads * rank) / math.sqrt(D))
            self.B = nn.Parameter(torch.randn(S, D, sim_heads * rank) / math.sqrt(D))
        self.inp = nn.Sequential(nn.Linear(n_feat + n_sim, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.layers = nn.ModuleList([_Layer(d, heads, S if use_vectors else 1, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        if residual:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, clue: torch.Tensor, cand: torch.Tensor, avail: torch.Tensor,
                base: torch.Tensor | None = None) -> torch.Tensor:
        """Scores (B, N) with -inf off `avail`.

        x (B, N, F) standardised features; clue (B,) and cand (B, N) rows of the
        word-vector table; avail (B, N) the words still on the board; base
        (B, N) the GBT logit when residual."""
        B, N, _ = x.shape
        parts = [x]
        if self.use_vectors:
            c = self.vecs[clue].float()                                   # (B, S, D)
            w = self.vecs[cand].float()                                   # (B, N, S, D)
            qc = torch.einsum("bsd,sdk->bsk", c, self.A).view(B, self.S, self.H, self.r)
            kw = torch.einsum("bnsd,sdk->bnsk", w, self.B).view(B, N, self.S, self.H, self.r)
            sim = torch.einsum("bshr,bnshr->bnsh", qc, kw).reshape(B, N, -1) / math.sqrt(self.r)
            cos = torch.einsum("bsd,bnsd->bns", c, w)
            s = torch.cat([sim, cos], -1)
            top = s.masked_fill(~avail[..., None], float("-inf")).amax(1, keepdim=True)
            parts += [s, s - top]
            edges = torch.einsum("bisd,bjsd->bijs", w, w)                 # (B, N, N, S)
        else:
            edges = x.new_zeros(B, N, N, 1)
        h = self.inp(torch.cat(parts, -1))
        for layer in self.layers:
            h = layer(h, edges, avail)
        score = self.out(self.ln(h)).squeeze(-1)
        if self.residual:
            score = score + base
        return score.masked_fill(~avail, float("-inf"))


def save_listener_net(path: Path, net: ListenerNet, mean: np.ndarray, sd: np.ndarray, meta: dict) -> None:
    torch.save({"config": net.config, "state": net.state_dict(), "mean": mean, "sd": sd, "meta": meta}, path)


def load_listener_net(path: Path, vectors: WordVectors, device: str = "cpu") -> tuple[ListenerNet, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = ListenerNet(vectors.vecs, **ck["config"]).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck

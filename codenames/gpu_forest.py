"""A LightGBM forest evaluated on the GPU, bit-identical to `Booster.predict(X, raw_score=True)`.

Why: a full-pool turn of the GBT spymaster scores ~10,700 clues x 25 words, about 267,000 rows, through
2,933 trees of up to 35 levels. On one CPU core LightGBM's predict takes ~73 s of the turn's ~93 s
(profiled 2026-11-17). Here one Triton kernel gives each row its own GPU thread.

Exactness. Three things could make the GPU disagree with LightGBM, and each is handled:
- Split comparisons. LightGBM tests `x <= threshold` in double precision. Instead of comparing floats on
  the GPU, each feature's thresholds are sorted once and every value is replaced on the CPU by its rank
  among them (`searchsorted`, float64): `x <= T[k]` exactly when `rank(x) <= k`. The GPU only compares
  integers.
- Missing and tiny values. LightGBM's predictor drops values with |x| <= kZeroThreshold (1e-35f) when it
  reads a row, so they reach the trees as 0.0. Its NumericalDecision then replaces NaN by 0 for a split
  whose missing type is not NaN, sends zeros to the default side of a split of missing type Zero, and
  NaN to the default side of one of missing type NaN. The same flags are computed on the CPU and packed
  into the integer code.
- The sum. LightGBM adds the trees' outputs one at a time, in tree order, starting from 0.0 in double.
  The GPU does the same: one float64 addition per tree, in order, so the result has the same rounding.
`tests/test_gpu_forest.py` checks `np.array_equal` against LightGBM.

CUDA is initialised lazily (`GpuBooster`), on the first predict, so the arena can fork its workers from a
parent that never touched the GPU.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Triton caches compiled kernels under ~/.triton by default; keep them in the project's gitignored cache/.
os.environ.setdefault("TRITON_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "cache" / "triton"))

K_ZERO = float(np.float32(1e-35))   # LightGBM's kZeroThreshold, a float constant compared as double
NAN_BIT, ZERO_BIT = 1 << 30, 1 << 29
RANK_MASK = ZERO_BIT - 1


class GpuForest:
    def __init__(self, booster, device: str = "cuda", chunk_rows: int = 1 << 20):
        import torch

        self.torch = torch
        self.device = device
        self.chunk_rows = chunk_rows
        self.n_features = booster.num_feature()
        trees = parse_model_text(booster.model_to_string())

        # One flat node table for the whole forest. Splits and leaves share it; a leaf points at itself, so
        # rows that reach a leaf early stay there for the remaining steps.
        feat, thr, mtype, defl, left, right, leaf, roots, depths = [], [], [], [], [], [], [], [], []
        for t in trees:
            n_split, n_leaf = len(t["split_feature"]), len(t["leaf_value"])
            base = len(feat)                    # splits at base + i, leaves at base + n_split + j
            node = lambda c: base + c if c >= 0 else base + n_split + (~c)   # noqa: E731
            for i in range(n_split):
                dt = t["decision_type"][i]
                if dt & 1:
                    raise ValueError("categorical splits are not supported")
                feat.append(t["split_feature"][i])
                thr.append(t["threshold"][i])
                mtype.append((dt >> 2) & 3)        # 0 None, 1 Zero, 2 NaN
                defl.append((dt >> 1) & 1)
                left.append(node(t["left_child"][i]))
                right.append(node(t["right_child"][i]))
                leaf.append(0.0)
            for j in range(n_leaf):
                i = len(feat)
                feat.append(0); thr.append(np.nan); mtype.append(0); defl.append(0)
                left.append(i); right.append(i); leaf.append(t["leaf_value"][j])
            roots.append(base)                  # the root split, or a one-leaf tree's leaf
            depths.append(_depth(t))

        feat, thr = np.array(feat, dtype=np.int64), np.array(thr, dtype=np.float64)
        is_split = ~np.isnan(thr)
        # Per feature, its sorted distinct thresholds; a split stores its threshold's index among them.
        self.thresholds = [np.unique(thr[is_split & (feat == f)]) for f in range(self.n_features)]
        rank = np.zeros(len(feat), dtype=np.int64)
        for f, ts in enumerate(self.thresholds):
            m = is_split & (feat == f)
            rank[m] = np.searchsorted(ts, thr[m])
        if max((len(ts) for ts in self.thresholds), default=0) >= RANK_MASK or self.n_features >= 256:
            raise ValueError("model too large for the packed node table")
        if len(feat) >= 2**31:
            raise ValueError("too many nodes")

        # Each node packed into one int64 (rank | feature << 32 | missing type << 40 | default left << 42),
        # and its two children into another (left | right << 32), so a level costs two lookups, not six.
        self.info_t = rank | (feat << 32) | (np.array(mtype, np.int64) << 40) | (np.array(defl, np.int64) << 42)
        self.child_t = np.array(left, np.int64) | (np.array(right, np.int64) << 32)

        dev = lambda a, dt: torch.as_tensor(np.asarray(a), dtype=dt, device=device)   # noqa: E731
        self.info = dev(self.info_t, torch.int64)
        self.child = dev(self.child_t, torch.int64)
        self.leaf = dev(leaf, torch.float64)
        self.roots = dev(roots, torch.int64)
        self.depths = dev(depths, torch.int64)
        self.max_depth = int(max(depths, default=0))

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Each value -> its threshold rank, with LightGBM's NaN and zero flags packed in (CPU, float64)."""
        X = np.asarray(X, dtype=np.float64)
        isnan = np.isnan(X)
        # LightGBM's predictor reads a dense row sparsely: it keeps only values with |x| > kZeroThreshold
        # (or NaN), so anything smaller arrives at the trees as exactly 0.0. A split can sit at -1e-35.
        x0 = np.where(isnan | (np.abs(X) <= K_ZERO), 0.0, X)   # NaN -> 0 too, for non-NaN-missing splits
        iszero = x0 == 0.0
        code = np.empty(X.shape, dtype=np.int32)
        for f, ts in enumerate(self.thresholds):
            code[:, f] = np.searchsorted(ts, x0[:, f], side="left")
        code |= np.where(isnan, NAN_BIT, 0).astype(np.int32)
        code |= np.where(iszero, ZERO_BIT, 0).astype(np.int32)
        return code

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch = self.torch
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"expected (n, {self.n_features}) features, got {X.shape}")
        out = np.empty(len(X), dtype=np.float64)
        for s in range(0, len(X), self.chunk_rows):
            code = torch.as_tensor(self.encode(X[s:s + self.chunk_rows]), device=self.device).contiguous()
            acc = torch.empty(len(code), dtype=torch.float64, device=self.device)
            grid = (triton_cdiv(len(code), BLOCK_ROWS),)
            _forest_kernel[grid](code, self.info, self.child, self.leaf, self.roots, self.depths, acc,
                                 len(code), len(self.roots), self.n_features,
                                 RANK_MASK=RANK_MASK, NAN_BIT=NAN_BIT, ZERO_BIT=ZERO_BIT, BLOCK=BLOCK_ROWS)
            out[s:s + len(code)] = acc.cpu().numpy()
        return out


BLOCK_ROWS = 128


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _make_kernel():
    """One GPU thread per row. It walks the trees one after another in the model's order and adds each
    tree's leaf to a float64 total, exactly as LightGBM's PredictRaw does, so the sum rounds the same way.
    Every lane of a block is at the same tree, so the depth loop has one bound per block."""
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(code_ptr, info_ptr, child_ptr, leaf_ptr, root_ptr, depth_ptr, out_ptr,
               n_rows, n_trees, n_feat,
               RANK_MASK: tl.constexpr, NAN_BIT: tl.constexpr, ZERO_BIT: tl.constexpr, BLOCK: tl.constexpr):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = rows < n_rows
        row_base = rows.to(tl.int64) * n_feat
        acc = tl.zeros([BLOCK], dtype=tl.float64)
        for t in range(0, n_trees):
            node = tl.load(root_ptr + t) + tl.zeros([BLOCK], dtype=tl.int64)
            depth = tl.load(depth_ptr + t)
            for _ in range(0, depth):
                info = tl.load(info_ptr + node)
                c = tl.load(code_ptr + row_base + ((info >> 32) & 0xFF), mask=mask, other=0)
                go_left = (c & RANK_MASK).to(tl.int64) <= (info & 0xFFFFFFFF)
                mtype = (info >> 40) & 3
                default_left = ((info >> 42) & 1) == 1
                go_left = tl.where((mtype == 2) & ((c & NAN_BIT) != 0), default_left, go_left)
                go_left = tl.where((mtype == 1) & ((c & ZERO_BIT) != 0), default_left, go_left)
                ch = tl.load(child_ptr + node)
                node = tl.where(go_left, ch & 0xFFFFFFFF, ch >> 32)
            acc += tl.load(leaf_ptr + node)
        tl.store(out_ptr + rows, acc, mask=mask)

    return kernel


class _LazyKernel:
    """Compiles the Triton kernel on first launch, in the process that launches it."""
    _k = None

    def __getitem__(self, grid):
        if _LazyKernel._k is None:
            _LazyKernel._k = _make_kernel()
        return _LazyKernel._k[grid]


_forest_kernel = _LazyKernel()


def parse_model_text(text: str) -> list[dict]:
    """The trees of a LightGBM model string. The text format, not `dump_model()`: the JSON dump rounds some
    thresholds (172 of the shipped listener's 369,558) and turns huge ones into inf, which moves rows that
    sit exactly on a split. The text format writes doubles with 17 significant digits."""
    if "num_tree_per_iteration=1\n" not in text.split("Tree=", 1)[0]:
        raise ValueError("only single-output models are supported")
    trees, cur = [], None
    ints = {"split_feature", "decision_type", "left_child", "right_child"}
    for line in text.splitlines():
        if line.startswith("Tree="):
            cur = {}
            trees.append(cur)
        elif line == "end of trees":
            break
        elif cur is not None and "=" in line:
            key, val = line.split("=", 1)
            if key in ints:
                cur[key] = [int(v) for v in val.split()]
            elif key in ("threshold", "leaf_value"):
                cur[key] = [float(v) for v in val.split()]
            elif key == "linear":
                if val.strip() not in ("0", ""):
                    raise ValueError("linear trees are not supported")
    for t in trees:
        for key in ("split_feature", "decision_type", "left_child", "right_child", "threshold"):
            t.setdefault(key, [])
    return trees


def _depth(t: dict) -> int:
    def d(c: int) -> int:
        return 0 if c < 0 else 1 + max(d(t["left_child"][c]), d(t["right_child"][c]))
    return d(0) if t["split_feature"] else 0


class GpuBooster:
    """Stands in for a `lightgbm.Booster` where only `predict(X, raw_score=True)` is called. Builds the GPU
    forest on first use, so nothing touches CUDA before the arena forks."""

    def __init__(self, booster, **kwargs):
        self.booster = booster
        self._kwargs = kwargs
        self._forest: GpuForest | None = None

    def predict(self, X, raw_score: bool = False, **kwargs):
        if not raw_score or kwargs:
            return self.booster.predict(X, raw_score=raw_score, **kwargs)
        if self._forest is None:
            self._forest = GpuForest(self.booster, **self._kwargs)
        return self._forest.predict(X)

    def __getattr__(self, name):
        return getattr(self.booster, name)

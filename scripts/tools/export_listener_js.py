"""Export the distilled listener as compact JSON a browser can evaluate.

LightGBM's own dump is deeply nested and verbose; this flattens each tree into
parallel arrays. A child slot holds a non-negative internal-node index, or the
bitwise complement of a leaf index, which is the usual trick for keeping the
whole tree in flat typed arrays.

**Missing values decide correctness here.** LightGBM's numerical decision is:

    if isnan(v) and missing_type != NaN:  v = 0
    if missing_type == NaN and isnan(v):  take default_left
    else:                                 v <= threshold ? left : right

This model uses only `<=` splits with missing_type in {None, NaN}, so two flag
bits carry everything: bit 0 is default_left, bit 1 is "missing_type is NaN".
Getting this wrong is silent -- 41% of clues have no SWOW row and 71% no entity
row, so NaN is the common case, not an edge case.

Usage:
    python scripts/tools/export_listener_js.py --out app/model.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from codenames.listener_features import FEATURE_NAMES


def flatten(node, feat, thr, left, right, flags, leaves):
    """Returns this node's child-slot encoding."""
    if "leaf_value" in node:
        leaves.append(float(node["leaf_value"]))
        return ~(len(leaves) - 1)
    idx = len(feat)
    feat.append(int(node["split_feature"]))
    thr.append(float(node["threshold"]))
    flags.append((1 if node["default_left"] else 0) | (2 if node["missing_type"] == "NaN" else 0))
    left.append(0); right.append(0)
    left[idx] = flatten(node["left_child"], feat, thr, left, right, flags, leaves)
    right[idx] = flatten(node["right_child"], feat, thr, left, right, flags, leaves)
    return idx


def main() -> None:
    import lightgbm as lgb

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=PROJECT_ROOT / "cache" / "listener_gbt.txt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dump = lgb.Booster(model_file=str(args.model)).dump_model()
    trees = []
    for ti in dump["tree_info"]:
        feat, thr, left, right, flags, leaves = [], [], [], [], [], []
        root = flatten(ti["tree_structure"], feat, thr, left, right, flags, leaves)
        trees.append({"r": root, "f": feat, "t": thr,
                      "l": left, "g": right, "m": flags,
                      "v": leaves})

    payload = {"features": list(FEATURE_NAMES), "trees": trees}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    n_nodes = sum(len(t["f"]) for t in trees)
    print(f"{len(trees)} trees, {n_nodes:,} internal nodes, "
          f"{sum(len(t['v']) for t in trees):,} leaves "
          f"-> {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

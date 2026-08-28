"""Post-hoc feature-vector ablations (SCOPE.md §9).

Pure array transforms on an already-generated `(N, feature_dim)` batch of
feature vectors, using `FeatureLayout` to know which columns are which.
No `Board`/`SimilarityTensor` involved, and no regeneration needed: both
ablations here can be derived directly from a base dataset already produced
by `scripts/generate_training_data.py`, since dropping a space or averaging
across spaces only rearranges/combines columns that already exist in the
full feature vector. (The sort ablation is different -- sorting is lossy,
so it needs `codenames.features.build_features_unsorted` and a fresh
generation pass instead; see `scripts/run_ablation_study.py`.)
"""

from __future__ import annotations

import numpy as np

from codenames.features import FeatureLayout


def drop_space(features: np.ndarray, layout: FeatureLayout, space: str) -> np.ndarray:
    """SCOPE §9's per-space ablation: remove one space's 25-column block
    entirely (not zero it out -- the model shouldn't have access to the
    dimension at all). Shape (N, D) -> (N, D - 25)."""
    keep_spaces = [s for s in layout.spaces if s != space]
    parts = [features[:, layout.space_slice(s)] for s in keep_spaces]
    parts.append(features[:, layout.mask_slice()])
    parts.append(features[:, layout.scalar_slice()])
    return np.concatenate(parts, axis=1)


def average_concatenation(features: np.ndarray, layout: FeatureLayout) -> np.ndarray:
    """SCOPE §9's concatenation ablation: replace the n_spaces separate
    25-column blocks with their straightforward positional mean (one
    25-column block). Deliberately naive -- averaging mixes values that
    (per §2 step 3's independent per-space sort) may come from different
    underlying words at the same position, no special-casing of that or of
    the -1 sentinel is applied, since demonstrating this is worse than
    proper concatenation is the point of the ablation (§2's own
    justification for concatenating rather than averaging). Shape
    (N, D) -> (N, 25 + 25 + 3)."""
    space_blocks = np.stack([features[:, layout.space_slice(s)] for s in layout.spaces], axis=0)
    averaged = space_blocks.mean(axis=0)
    return np.concatenate([averaged, features[:, layout.mask_slice()], features[:, layout.scalar_slice()]], axis=1)

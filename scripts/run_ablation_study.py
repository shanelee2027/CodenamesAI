"""Run the M9 evaluation/ablation study (SCOPE.md §9), moderate real scale.

Generates a base dataset plus what each ablation axis actually needs (see
module docstrings on codenames/features.py, codenames/ablation.py, and
generate_training_data.py's feature_builder/guesser_weights params for why
most axes don't need their own fresh generation), trains one model per
variant via scripts/train_scorer.py's reused training loop, and writes a
comparison report.

Variants trained:
- full: the base dataset, full sorted/concatenated features (Scorer).
- drop_<space>: base dataset with one embedding space's columns removed.
- averaged: base dataset with per-space blocks averaged instead of
  concatenated.
- unsorted: a fresh dataset with the same sampled boards/clues/guessers as
  the base (same seed), but unsorted features.
- pool_<uniform|glove_heavy|numberbatch_heavy|wikipedia2vec_heavy>: fresh
  datasets sharing the same board/clue sample sequence (same seed) but
  different guesser-selection weights.
- linear_baseline: the base dataset, trained with LinearScorer instead of
  Scorer (SCOPE §6 baseline 4).
- noise_<value> (opt-in via --noise-levels): fresh datasets, one per
  requested noise_std, each generated from a temporary copy of the
  guesser pool config with every guesser's noise_std overridden to that
  value (same spaces, same per-guesser seeds 1/2/3) -- since NoisyGuesser
  draws `rng.normal(0, noise_std)` from a seed-determined stream, the same
  seed at a different noise_std is the same underlying draws at a
  different scale, so levels are directly comparable, not just similarly
  distributed. Same generation seed as `base` too, so all noise levels
  (and `full`, effectively noise_std=whatever's in the default pool
  config) share the identical board/clue sample sequence as well.

The 6 dataset-generation calls above are independent (separate output
dirs, no shared mutable state) and CPU-bound, so they run across a
process pool (like codenames/arena.py's multiprocessing) instead of one
after another -- this is the dominant cost (~40 min sequentially at
moderate scale), while the 11 training runs after it are individually
fast enough that parallelizing them too wasn't worth the added complexity.

Usage:
    python scripts/run_ablation_study.py
    python scripts/run_ablation_study.py --noise-levels "0.0,0.03,0.06,0.1,0.15" --noise-only

`--noise-only` skips every variant above except the noise sweep -- for
refreshing just the web UI's learned:noise_* checkpoints
(scripts/web_inspector.py) without paying for the other 6 axes, which
aren't kept as permanent UI options.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from generate_training_data import generate  # noqa: E402
from train_scorer import train  # noqa: E402

from codenames.ablation import average_concatenation, drop_space
from codenames.features import FeatureLayout, build_features_unsorted
from codenames.guessers.registry import DEFAULT_POOL_CONFIG, training_pool
from codenames.scorer import LinearScorer, Scorer
from codenames.similarity import DEFAULT_CACHE_DIR, SimilarityTensor


def _generate_if_needed(output_dir: Path, n_examples: int, **kwargs) -> None:
    if output_dir.exists() and any(output_dir.glob("features_*.npy")):
        print(f"[skip] {output_dir} already generated")
        return
    print(f"[generate] {output_dir} ({n_examples} examples)")
    t0 = time.time()
    generate(n_examples=n_examples, shard_size=n_examples, output_dir=output_dir, **kwargs)
    print(f"  done: {output_dir} in {time.time() - t0:.0f}s")


def _generation_job(job: dict) -> None:
    """Top-level (picklable) wrapper so _generate_if_needed's independent
    calls can run under ProcessPoolExecutor -- each writes to its own
    output_dir, so there's no shared mutable state between jobs."""
    job = dict(job)
    output_dir = job.pop("output_dir")
    n_examples = job.pop("n_examples")
    _generate_if_needed(output_dir, n_examples, **job)


def _noise_level_tag(noise_std: float) -> str:
    return str(noise_std).replace(".", "_")


def _write_noise_pool_config(base_pool_config: Path, noise_std: float, output_path: Path) -> Path:
    """A copy of base_pool_config with every noisy guesser's noise_std
    overridden -- same spaces and per-guesser seeds, so only the noise
    magnitude differs between levels, not which guesser is which or what
    its underlying (pre-noise) random stream looks like."""
    config = json.loads(base_pool_config.read_text())
    for entry in config["guessers"]:
        if entry.get("type") == "noisy":
            entry["params"]["noise_std"] = noise_std
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config))
    return output_path


def _derive_if_needed(src_dir: Path, dst_dir: Path, transform) -> None:
    if dst_dir.exists() and any(dst_dir.glob("features_*.npy")):
        print(f"[skip] {dst_dir} already derived")
        return
    print(f"[derive] {dst_dir} from {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for features_path in sorted(src_dir.glob("features_*.npy")):
        idx = features_path.stem.split("_", 1)[1]
        features = np.load(features_path)
        np.save(dst_dir / f"features_{idx}.npy", transform(features).astype(np.float32))
        for name in ("outcome", "seed"):
            np.save(dst_dir / f"{name}_{idx}.npy", np.load(src_dir / f"{name}_{idx}.npy"))


def _best_epoch_metrics(checkpoint_dir: Path) -> dict:
    with (checkpoint_dir / "training_curves.csv").open() as f:
        rows = list(csv.DictReader(f))
    best = min(rows, key=lambda r: float(r["val_loss"]))
    return {"val_loss": float(best["val_loss"]), "val_accuracy": float(best["val_accuracy"]), "epoch": int(best["epoch"])}


def _train_variant(name: str, data_dir: Path, checkpoints_root: Path, train_kwargs: dict, model_factory=Scorer) -> dict:
    out_dir = checkpoints_root / name
    print(f"[train] {name}")
    t0 = time.time()
    train(data_dir=data_dir, output_dir=out_dir, model_factory=model_factory, **train_kwargs)
    metrics = _best_epoch_metrics(out_dir)
    metrics["train_seconds"] = time.time() - t0
    metrics["n_examples"] = sum(len(np.load(p, mmap_mode="r")) for p in data_dir.glob("outcome_*.npy"))
    print(f"  val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_accuracy']:.4f} ({metrics['train_seconds']:.0f}s)")
    return metrics


def _linear_feature_importance(checkpoint_dir: Path, layout: FeatureLayout, top_k: int = 20) -> list[tuple[str, float]]:
    checkpoint = torch.load(checkpoint_dir / "scorer_best.pt", map_location="cpu")
    model = LinearScorer(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state"])
    weight = model.net.weight.detach().numpy()  # (N_OUTCOME_CLASSES, input_dim)
    importance = np.linalg.norm(weight, axis=0)
    order = np.argsort(-importance)[:top_k]
    return [(layout.describe(int(i)), float(importance[i])) for i in order]


def _write_report(path: Path, results: dict[str, dict], importances: list[tuple[str, float]] | None) -> None:
    lines = ["# M9 ablation study report\n", "| variant | n_examples | val_loss | val_accuracy |", "|---|---|---|---|"]
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["val_loss"]):
        lines.append(f"| {name} | {m['n_examples']} | {m['val_loss']:.4f} | {m['val_accuracy']:.4f} |")
    if importances is not None:
        lines.append("\n## Linear baseline: top features by importance (L2 norm of weight across k-classes)\n")
        lines.append("| feature | importance |")
        lines.append("|---|---|")
        for label, value in importances:
            lines.append(f"| {label} | {value:.4f} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-n", type=int, default=200_000)
    parser.add_argument("--pool-sensitivity-n", type=int, default=50_000)
    parser.add_argument("--data-root", type=Path, default=Path("cache/m9"))
    parser.add_argument("--guesser-pool-config", type=Path, default=DEFAULT_POOL_CONFIG)
    parser.add_argument("--sims-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=None, help="generation-phase worker processes (default: os.cpu_count())")
    parser.add_argument(
        "--noise-levels",
        type=str,
        default="",
        help="comma-separated noise_std values to sweep (e.g. '0.0,0.03,0.06,0.1,0.15'); "
        "empty (default) skips the noise sweep entirely. Each level trains its own "
        "'noise_<value>' variant from a fresh dataset generated with that noise_std, "
        "otherwise identical to the default guesser pool config.",
    )
    parser.add_argument(
        "--noise-only",
        action="store_true",
        help="skip base/unsorted/pool-sensitivity/drop-space/averaged/linear_baseline entirely -- "
        "generate and train just the --noise-levels variants. For refreshing the web UI's "
        "learned:noise_* checkpoints without paying for a full 11-variant study.",
    )
    args = parser.parse_args()
    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]
    if args.noise_only and not noise_levels:
        parser.error("--noise-only requires --noise-levels")

    sims = SimilarityTensor.load(args.sims_cache_dir)
    layout = FeatureLayout(spaces=sims.spaces)
    guesser_names = list(training_pool(args.guesser_pool_config).keys())
    train_kwargs = dict(max_epochs=args.max_epochs, patience=args.patience, seed=args.seed)

    data_root = args.data_root
    checkpoints_root = data_root / "checkpoints"
    common = dict(guesser_pool_config=args.guesser_pool_config, sims_cache_dir=args.sims_cache_dir)

    base_dir = data_root / "base"
    unsorted_dir = data_root / "unsorted"

    # Pool-sensitivity sweep: same seed across configs -> same underlying
    # board/clue samples, only guesser weighting differs.
    pool_configs = {"uniform": {n: 1.0 for n in guesser_names}}
    for n in guesser_names:
        # e.g. "noisy_glove" -> "glove_heavy"
        axis = n.replace("noisy_", "")
        pool_configs[f"{axis}_heavy"] = {other: (3.0 if other == n else 1.0) for other in guesser_names}
    pool_dirs = {name: data_root / f"pool_{name}" for name in pool_configs}
    if args.noise_only:
        pool_configs = {}
        pool_dirs = {}

    # Noise-std sweep (opt-in): same generation seed as base, so it's the
    # same underlying board/clue samples -- only the guesser noise
    # magnitude differs between levels and against `full`.
    noise_pool_configs = {}
    noise_dirs = {}
    for noise_std in noise_levels:
        tag = _noise_level_tag(noise_std)
        pool_config_path = _write_noise_pool_config(
            args.guesser_pool_config, noise_std, data_root / "pool_configs" / f"noise_{tag}.json"
        )
        noise_pool_configs[tag] = pool_config_path
        noise_dirs[tag] = data_root / f"noise_{tag}"

    jobs = []
    if not args.noise_only:
        jobs.extend(
            [
                {"output_dir": base_dir, "n_examples": args.base_n, "seed": args.seed, **common},
                {
                    "output_dir": unsorted_dir, "n_examples": args.base_n, "seed": args.seed,
                    "feature_builder": build_features_unsorted, **common,
                },
            ]
        )
    jobs.extend(
        {"output_dir": pool_dirs[name], "n_examples": args.pool_sensitivity_n, "seed": args.seed + 1, "guesser_weights": weights, **common}
        for name, weights in pool_configs.items()
    )
    jobs.extend(
        {
            "output_dir": noise_dirs[tag], "n_examples": args.base_n, "seed": args.seed,
            "guesser_pool_config": noise_pool_configs[tag], "sims_cache_dir": args.sims_cache_dir,
        }
        for tag in noise_pool_configs
    )

    print(f"[generate] running {len(jobs)} generation jobs across up to {args.max_workers or 'os.cpu_count()'} workers")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        list(pool.map(_generation_job, jobs))
    print(f"[generate] all {len(jobs)} jobs done in {time.time() - t0:.0f}s")

    variants: dict[str, Path] = {}
    importances = None
    if not args.noise_only:
        # --- Derive: drop-space (x n_spaces) + averaged-concatenation, from base ---
        drop_space_dirs = {}
        for space in sims.spaces:
            d = data_root / f"drop_{space}"
            _derive_if_needed(base_dir, d, lambda f, s=space: drop_space(f, layout, s))
            drop_space_dirs[space] = d

        averaged_dir = data_root / "averaged"
        _derive_if_needed(base_dir, averaged_dir, lambda f: average_concatenation(f, layout))

        variants = {"full": base_dir, "unsorted": unsorted_dir, "averaged": averaged_dir}
        variants.update({f"drop_{space}": d for space, d in drop_space_dirs.items()})
        variants.update({f"pool_{name}": d for name, d in pool_dirs.items()})

    variants.update({f"noise_{tag}": d for tag, d in noise_dirs.items()})

    results = {name: _train_variant(name, d, checkpoints_root, train_kwargs) for name, d in variants.items()}

    if not args.noise_only:
        results["linear_baseline"] = _train_variant("linear_baseline", base_dir, checkpoints_root, train_kwargs, model_factory=LinearScorer)
        importances = _linear_feature_importance(checkpoints_root / "linear_baseline", layout)

    report_path = data_root / "report.md"
    _write_report(report_path, results, importances)
    print(f"\nreport written to {report_path}")

    print(f"\n{'variant':20s} {'n':>8s} {'val_loss':>10s} {'val_acc':>9s}")
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["val_loss"]):
        print(f"{name:20s} {m['n_examples']:8d} {m['val_loss']:10.4f} {m['val_accuracy']:9.4f}")


if __name__ == "__main__":
    main()

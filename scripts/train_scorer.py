"""Train the learned scorer (SCOPE.md §M8) on M7's generated training data.

Board-seed splitting (SCOPE §4): "the same board appears in many training
examples [if a board is reused across several sampled clues]. Row-wise
splits leak boards across train/val and inflate validation numbers." Split
here is a deterministic hash of each example's board seed
(`seed % 1000 < val_fraction * 1000`), not a random row-wise split -- every
example generated from a given board lands in the same partition no matter
which shard file or position it's in, and the split is stable across
re-runs/added shards without needing to load the whole dataset into memory
first.

Outputs to `--output-dir` (default cache/checkpoints/, gitignored):
- `scorer_best.pt`: best-val-loss checkpoint (model state_dict + input_dim).
- `training_curves.csv` / `training_curves.png`: per-epoch train/val loss
  and val accuracy.
- `reliability_diagrams.png`: one panel per k in 0..MAX_K, predicted
  probability of that class vs. its observed frequency, binned -- the
  calibration check SCOPE asks for.

Usage:
    python scripts/train_scorer.py --data-dir cache/training_data
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from codenames.scorer import N_OUTCOME_CLASSES, Scorer, decode_outcome_class


class ShardedTrainingData(Dataset):
    """Reads M7's sharded (features, outcome, seed) .npy triples, keeping
    only rows whose board seed passes `seed_predicate`. Feature shards
    stay memory-mapped (read-only); only the small outcome/seed arrays
    are loaded fully, since filtering needs to inspect every row's seed
    anyway. `outcome` is `codenames.scorer.outcome_class(k, cause)`, one
    of `N_OUTCOME_CLASSES` classes -- see scorer.py's module docstring."""

    def __init__(self, data_dir: Path, seed_predicate):
        feature_paths = sorted(data_dir.glob("features_*.npy"))
        if not feature_paths:
            raise ValueError(f"no shards found in {data_dir} (expected features_*.npy)")

        self._features: list[np.memmap] = []
        self._outcome: list[np.ndarray] = []
        self._row_map: list[tuple[int, int]] = []

        for path in feature_paths:
            idx = path.stem.split("_", 1)[1]
            seeds = np.load(data_dir / f"seed_{idx}.npy")
            keep = np.flatnonzero(seed_predicate(seeds))
            if len(keep) == 0:
                continue
            shard_i = len(self._features)
            self._features.append(np.load(path, mmap_mode="r"))
            self._outcome.append(np.load(data_dir / f"outcome_{idx}.npy"))
            self._row_map.extend((shard_i, int(local)) for local in keep)

    @property
    def feature_dim(self) -> int:
        return self._features[0].shape[1]

    def __len__(self) -> int:
        return len(self._row_map)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        shard_i, local = self._row_map[i]
        x = np.array(self._features[shard_i][local], dtype=np.float32, copy=True)
        y = int(self._outcome[shard_i][local])
        return torch.from_numpy(x), y


def _is_val_seed(seeds: np.ndarray, val_fraction: float) -> np.ndarray:
    return (seeds % 1000) < int(val_fraction * 1000)


def _run_epoch(model: Scorer, loader: DataLoader, device: torch.device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_count += len(y)

    return total_loss / total_count, total_correct / total_count


def _reliability_curve(pred_probs: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(pred_probs, bins[1:-1]), 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    freq = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_ids == b
        counts[b] = mask.sum()
        if counts[b] > 0:
            mean_pred[b] = pred_probs[mask].mean()
            freq[b] = actual[mask].mean()
    return mean_pred, freq, counts


def _plot_training_curves(history: list[dict], output_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.legend()
    ax1.set_title("Loss")

    ax2.plot(epochs, [h["val_accuracy"] for h in history])
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("val accuracy")
    ax2.set_title("Validation accuracy")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_reliability_diagrams(all_probs: np.ndarray, all_labels: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, N_OUTCOME_CLASSES, figsize=(3 * N_OUTCOME_CLASSES, 4), sharey=True)
    for cls in range(N_OUTCOME_CLASSES):
        mean_pred, freq, counts = _reliability_curve(all_probs[:, cls], (all_labels == cls).astype(np.float32))
        ax = axes[cls]
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
        valid = counts > 0
        ax.plot(mean_pred[valid], freq[valid], marker="o", label="observed")
        k, cause = decode_outcome_class(cls)
        ax.set_title(f"k={k}\n{cause.value if cause else 'clean'}", fontsize=9)
        ax.set_xlabel("predicted P")
        if cls == 0:
            ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train(
    data_dir: Path,
    output_dir: Path,
    val_fraction: float = 0.1,
    batch_size: int = 2048,
    max_epochs: int = 50,
    patience: int = 5,
    lr: float = 1e-3,
    seed: int = 0,
    model_factory: Callable[[int], nn.Module] = Scorer,
    num_workers: int = 4,
) -> Path:
    """`model_factory` exists for SCOPE §9's linear baseline
    (scripts/run_ablation_study.py passes `LinearScorer`) -- everything
    else here (splitting, early stopping, checkpointing, curves,
    reliability diagrams) is architecture-agnostic.

    `num_workers>0` prefetches batches in worker subprocesses (each
    ShardedTrainingData.__getitem__ call is a small memmap read + copy --
    otherwise that happens serially in the main process between GPU
    steps). `batch_size` defaults much higher than a typical vision/NLP
    model's would: this MLP is tiny, so the GPU is barely exercised per
    step at small batch sizes -- bigger batches do more useful work per
    kernel launch instead of just adding more (cheap) steps."""
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = ShardedTrainingData(data_dir, lambda s: ~_is_val_seed(s, val_fraction))
    val_data = ShardedTrainingData(data_dir, lambda s: _is_val_seed(s, val_fraction))
    if len(val_data) == 0:
        raise ValueError(
            f"no validation examples in {data_dir} with val_fraction={val_fraction} -- "
            "generate more data or check that board seeds vary"
        )

    loader_kwargs = dict(num_workers=num_workers, pin_memory=(device.type == "cuda"), persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = model_factory(train_data.feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_path = output_dir / "scorer_best.pt"
    history: list[dict] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, _ = _run_epoch(model, train_loader, device, optimizer)
        val_loss, val_accuracy = _run_epoch(model, val_loader, device, optimizer=None)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy})
        print(f"epoch {epoch:3d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "input_dim": train_data.feature_dim, "val_loss": val_loss}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

    with (output_dir / "training_curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"])
        writer.writeheader()
        writer.writerows(history)
    _plot_training_curves(history, output_dir / "training_curves.png")

    best = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best["model_state"])
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            probs = model.predict_proba(x.to(device)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())
    _plot_reliability_diagrams(np.concatenate(all_probs), np.concatenate(all_labels), output_dir / "reliability_diagrams.png")

    print(f"best val_loss={best_val_loss:.4f}, checkpoint at {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("cache/training_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/checkpoints"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader prefetch workers (0 disables)")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

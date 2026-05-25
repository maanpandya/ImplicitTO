"""Train the conditional implicit topology optimization model."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import TopologyDataset
from model import ConditionalImplicitNetwork


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")
    if args.amp and device.type != "cuda":
        print("--amp requested, but CUDA is not available; using full precision.")

    torch.manual_seed(args.seed)

    eval_dataset = TopologyDataset(
        dataset_path=args.dataset,
        points_per_sample=args.points_per_sample,
        jitter_coordinates=False,
    )
    train_source = TopologyDataset(
        dataset_path=args.dataset,
        points_per_sample=args.points_per_sample,
        jitter_coordinates=args.jitter_coordinates,
    )
    train_indices, val_indices = split_indices(len(eval_dataset), args.val_fraction, args.seed)
    train_dataset = Subset(train_source, train_indices)
    val_dataset = Subset(eval_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ConditionalImplicitNetwork(
        cond_dim=eval_dataset.conditions.shape[1],
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        num_frequencies=args.num_frequencies,
        fourier_sigma=args.fourier_sigma,
        activation=args.activation,
        encoding_type=args.encoding_type,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = make_loss(args.loss)

    checkpoint_dir = Path(args.checkpoint_dir)
    viz_dir = Path(args.viz_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    start_epoch = 0
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            Path(args.resume),
            model,
            optimizer,
            device,
        )

    final_epoch = args.epochs
    if args.additional_epochs is not None:
        final_epoch = start_epoch + args.additional_epochs
    if final_epoch <= start_epoch:
        raise ValueError(
            f"Final epoch ({final_epoch}) must be greater than resume epoch "
            f"({start_epoch})"
        )

    for epoch in range(start_epoch + 1, final_epoch + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            args.max_train_batches,
            args.amp,
        )
        val_loss = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            args.max_val_batches,
            args.amp,
        )
        print(
            f"epoch {epoch:04d}/{final_epoch:04d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if epoch % args.val_every == 0:
            save_validation_figure(
                model=model,
                dataset=eval_dataset,
                val_dataset=val_dataset,
                device=device,
                epoch=epoch,
                output_dir=viz_dir,
            )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if is_best or epoch % args.checkpoint_every == 0 or epoch == final_epoch:
            save_checkpoint(
                checkpoint_dir / ("best.pth" if is_best else f"epoch_{epoch:04d}.pth"),
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                args,
                eval_dataset,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/dataset.npz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    parser.add_argument(
        "--additional-epochs",
        type=int,
        default=None,
        help="Train this many more epochs after the resumed checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--points-per-sample", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", choices=("mse", "bce"), default="mse")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=6)
    parser.add_argument(
        "--encoding-type",
        choices=("positional", "gaussian"),
        default="positional",
        help="Coordinate encoding. Positional is bandwidth-limited for smoother super-resolution.",
    )
    parser.add_argument(
        "--num-frequencies",
        type=int,
        default=5,
        help="Maximum positional frequency L, or number of Gaussian Fourier features.",
    )
    parser.add_argument(
        "--fourier-sigma",
        type=float,
        default=1.0,
        help="Gaussian Fourier sigma; ignored by positional encoding.",
    )
    parser.add_argument("--activation", choices=("silu", "relu", "gelu"), default="silu")
    parser.add_argument(
        "--no-jitter-coordinates",
        dest="jitter_coordinates",
        action="store_false",
        help="Disable training-time voxel jitter.",
    )
    parser.set_defaults(jitter_coordinates=True)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--viz-dir", default="outputs/validation")
    parser.add_argument("--val-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision when training on an NVIDIA GPU.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.additional_epochs is not None and args.additional_epochs <= 0:
        raise ValueError("--additional-epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.points_per_sample <= 0:
        raise ValueError("--points-per-sample must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if args.val_every <= 0:
        raise ValueError("--val-every must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    return args


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_dataset(
    dataset: TopologyDataset, val_fraction: float, seed: int
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    train_indices, val_indices = split_indices(len(dataset), val_fraction, seed)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def split_indices(
    dataset_size: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    val_size = max(1, int(round(dataset_size * val_fraction)))
    train_size = dataset_size - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested validation split")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_size, generator=generator).tolist()
    return permutation[:train_size], permutation[train_size:]


def make_loss(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "bce":
        return nn.BCELoss()
    raise ValueError(f"Unsupported loss: {name}")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: int | None,
    amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    use_amp = amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for batch_idx, batch in enumerate(dataloader, start=1):
        coords = batch["coords"].to(device, non_blocking=True)
        cond = batch["cond"].to(device, non_blocking=True)
        gt_density = batch["gt_density"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        amp_context = torch.cuda.amp.autocast() if use_amp else nullcontext()
        with amp_context:
            pred_density = model(coords, cond)
            loss = loss_fn(pred_density, gt_density)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_batches += 1
        if max_batches is not None and batch_idx >= max_batches:
            break

    return total_loss / max(1, total_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: int | None,
    amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    use_amp = amp and device.type == "cuda"

    for batch_idx, batch in enumerate(dataloader, start=1):
        coords = batch["coords"].to(device, non_blocking=True)
        cond = batch["cond"].to(device, non_blocking=True)
        gt_density = batch["gt_density"].to(device, non_blocking=True)

        amp_context = torch.cuda.amp.autocast() if use_amp else nullcontext()
        with amp_context:
            pred_density = model(coords, cond)
            loss = loss_fn(pred_density, gt_density)

        total_loss += loss.item()
        total_batches += 1
        if max_batches is not None and batch_idx >= max_batches:
            break

    return total_loss / max(1, total_batches)


@torch.no_grad()
def save_validation_figure(
    *,
    model: nn.Module,
    dataset: TopologyDataset,
    val_dataset: torch.utils.data.Subset,
    device: torch.device,
    epoch: int,
    output_dir: Path,
) -> None:
    model.eval()
    sample_idx = int(val_dataset.indices[0])
    coords = dataset.coords.to(device).unsqueeze(0)
    cond = dataset.conditions[sample_idx].to(device).unsqueeze(0)
    gt_topology = dataset.topologies[sample_idx].cpu().numpy()

    pred_topology = model(coords, cond).squeeze(0).squeeze(-1)
    pred_topology = pred_topology.reshape(dataset.nelx, dataset.nely).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, image, title in (
        (axes[0], gt_topology, "Ground Truth SIMP"),
        (axes[1], pred_topology, "Network Prediction"),
    ):
        ax.imshow(
            image.T,
            cmap="gray_r",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    volfrac = float(dataset.conditions[sample_idx, 0])
    load_x = float(dataset.conditions[sample_idx, 1])
    load_y = float(dataset.conditions[sample_idx, 2])
    load_fx = float(dataset.conditions[sample_idx, 3])
    load_fy = float(dataset.conditions[sample_idx, 4])
    fig.suptitle(
        f"epoch={epoch} sample={sample_idx} vol={volfrac:.2f} "
        f"loc=({load_x:.2f}, {load_y:.2f}) F=({load_fx:.2f}, {load_fy:.2f})"
    )
    fig.tight_layout()

    output_path = output_dir / f"epoch_{epoch:04d}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved validation figure to {output_path}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    args: argparse.Namespace,
    dataset: TopologyDataset,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "args": vars(args),
        "nelx": dataset.nelx,
        "nely": dataset.nely,
        "condition_dim": dataset.conditions.shape[1],
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = int(checkpoint["epoch"])
    best_val_loss = float(checkpoint.get("val_loss", float("inf")))
    print(
        f"Resumed checkpoint {path} from epoch {start_epoch} "
        f"with val_loss={best_val_loss:.6f}"
    )
    return start_epoch, best_val_loss


if __name__ == "__main__":
    main()

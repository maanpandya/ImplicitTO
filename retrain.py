"""Generate the dataset and retrain the conditional implicit model."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    args = parse_args()

    if not args.skip_data_generation:
        run(
            [
                sys.executable,
                "generate_data.py",
                "--output",
                args.dataset,
                "--num-samples",
                str(args.num_samples),
                "--nelx",
                str(args.nelx),
                "--nely",
                str(args.nely),
                "--min-volfrac",
                str(args.min_volfrac),
                "--max-volfrac",
                str(args.max_volfrac),
                "--max-iter",
                str(args.simp_max_iter),
                "--change-tol",
                str(args.simp_change_tol),
                "--save-every",
                str(args.save_every),
                "--num-workers",
                str(args.data_num_workers),
                "--seed",
                str(args.seed),
            ]
        )

    train_command = [
        sys.executable,
        "train.py",
        "--dataset",
        args.dataset,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--points-per-sample",
        str(args.points_per_sample),
        "--hidden-dim",
        "256",
        "--num-hidden-layers",
        "6",
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--viz-dir",
        args.viz_dir,
        "--num-workers",
        str(args.train_num_workers),
        "--seed",
        str(args.seed),
    ]
    if args.amp:
        train_command.append("--amp")

    run(train_command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/dataset.npz")
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--nelx", type=int, default=60)
    parser.add_argument("--nely", type=int, default=30)
    parser.add_argument("--min-volfrac", type=float, default=0.2)
    parser.add_argument("--max-volfrac", type=float, default=0.6)
    parser.add_argument("--simp-max-iter", type=int, default=200)
    parser.add_argument("--simp-change-tol", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--data-num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Worker processes for parallel SIMP dataset generation.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--points-per-sample", type=int, default=1024)
    parser.add_argument(
        "--train-num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers used during training.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA mixed precision during training.",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--viz-dir", default="outputs/validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Train from an existing dataset.",
    )
    args = parser.parse_args()

    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    if args.data_num_workers <= 0:
        raise ValueError("--data-num-workers must be positive")
    if args.train_num_workers < 0:
        raise ValueError("--train-num-workers cannot be negative")

    return args


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

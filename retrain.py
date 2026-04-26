"""Generate the dataset and retrain the conditional implicit model."""

from __future__ import annotations

import argparse
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
                "--seed",
                str(args.seed),
            ]
        )

    run(
        [
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
            "--seed",
            str(args.seed),
        ]
    )


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
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--points-per-sample", type=int, default=1024)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--viz-dir", default="outputs/validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Train from an existing dataset.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

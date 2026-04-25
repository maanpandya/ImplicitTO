"""Generate an offline SIMP dataset for amortized topology optimization."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

from SIMP import solve_simp


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    conditions, topologies, compliances, load_dofs = _load_existing_dataset(
        output_path, args.num_samples, args.nelx, args.nely
    )
    start_idx = len(conditions)

    if start_idx:
        print(f"Resuming {output_path} from {start_idx}/{args.num_samples} samples")

    for sample_idx in range(start_idx, args.num_samples):
        load_dof, load_val, condition = _sample_right_edge_load(
            args.nelx, args.nely, rng, args.min_abs_load
        )
        topology, compliance = solve_simp(
            args.nelx,
            args.nely,
            args.volfrac,
            args.penal,
            args.rmin,
            args.ft,
            load_dof,
            load_val,
            max_iter=args.max_iter,
            change_tol=args.change_tol,
            verbose=args.verbose_simp,
        )

        conditions.append(condition)
        topologies.append(topology)
        compliances.append(compliance)
        load_dofs.append(load_dof)

        completed = sample_idx + 1
        if completed % args.save_every == 0 or completed == args.num_samples:
            _save_dataset(
                output_path,
                conditions,
                topologies,
                compliances,
                load_dofs,
                args,
            )

        print(
            f"[{completed}/{args.num_samples}] "
            f"dof={load_dof} load={load_val:.3f} compliance={compliance:.3f}"
        )

    print(f"Saved {len(conditions)} samples to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/dataset.npz")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--nelx", type=int, default=60)
    parser.add_argument("--nely", type=int, default=30)
    parser.add_argument("--volfrac", type=float, default=0.4)
    parser.add_argument("--penal", type=float, default=3.0)
    parser.add_argument("--rmin", type=float, default=2.0)
    parser.add_argument("--ft", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--change-tol", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-abs-load",
        type=float,
        default=0.1,
        help="Minimum absolute vertical load magnitude to avoid near-zero cases.",
    )
    parser.add_argument(
        "--verbose-simp",
        action="store_true",
        help="Print every SIMP iteration instead of one line per sample.",
    )
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    if not 0.0 <= args.min_abs_load < 1.0:
        raise ValueError("--min-abs-load must be in [0, 1)")

    return args


def _sample_right_edge_load(
    nelx: int, nely: int, rng: np.random.Generator, min_abs_load: float
) -> tuple[int, float, np.ndarray]:
    y_node = int(rng.integers(0, nely + 1))
    right_edge_node = nelx * (nely + 1) + y_node
    load_dof = 2 * right_edge_node + 1

    magnitude = float(rng.uniform(min_abs_load, 1.0))
    sign = float(rng.choice([-1.0, 1.0]))
    load_val = sign * magnitude

    condition = np.array([1.0, y_node / nely, load_val], dtype=np.float32)
    return load_dof, load_val, condition


def _load_existing_dataset(
    output_path: Path, num_samples: int, nelx: int, nely: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[int]]:
    if not output_path.exists():
        return [], [], [], []

    with np.load(output_path) as data:
        conditions = data["conditions"]
        topologies = data["topologies"]
        compliances = data["compliances"]
        load_dofs = data["load_dofs"]

    if topologies.shape[1:] != (nelx, nely):
        raise ValueError(
            f"Existing topology shape {topologies.shape[1:]} does not match "
            f"requested shape {(nelx, nely)}"
        )
    if len(conditions) > num_samples:
        raise ValueError(
            f"Existing dataset already has {len(conditions)} samples, "
            f"which is more than --num-samples={num_samples}"
        )

    return (
        [row.astype(np.float32) for row in conditions],
        [topology.astype(np.float32) for topology in topologies],
        [float(value) for value in compliances],
        [int(value) for value in load_dofs],
    )


def _save_dataset(
    output_path: Path,
    conditions: list[np.ndarray],
    topologies: list[np.ndarray],
    compliances: list[float],
    load_dofs: list[int],
    args: argparse.Namespace,
) -> None:
    metadata = {
        "nelx": args.nelx,
        "nely": args.nely,
        "volfrac": args.volfrac,
        "penal": args.penal,
        "rmin": args.rmin,
        "ft": args.ft,
        "max_iter": args.max_iter,
        "change_tol": args.change_tol,
        "seed": args.seed,
    }

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", dir=output_path.parent, delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)
        np.savez_compressed(
            temp_file,
            conditions=np.asarray(conditions, dtype=np.float32),
            topologies=np.asarray(topologies, dtype=np.float32),
            compliances=np.asarray(compliances, dtype=np.float32),
            load_dofs=np.asarray(load_dofs, dtype=np.int64),
            **metadata,
        )

    os.replace(temp_path, output_path)


if __name__ == "__main__":
    main()

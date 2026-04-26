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
    conditions, topologies, compliances, load_dofs, load_values = _load_existing_dataset(
        output_path, args.num_samples, args.nelx, args.nely
    )
    start_idx = len(conditions)

    if start_idx:
        print(f"Resuming {output_path} from {start_idx}/{args.num_samples} samples")

    for sample_idx in range(start_idx, args.num_samples):
        load_dof, load_val, condition = _sample_design_condition(
            args.nelx,
            args.nely,
            rng,
            args.min_volfrac,
            args.max_volfrac,
            args.min_load_norm,
        )
        topology, compliance = solve_simp(
            args.nelx,
            args.nely,
            float(condition[0]),
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
        load_values.append(load_val)

        completed = sample_idx + 1
        if completed % args.save_every == 0 or completed == args.num_samples:
            _save_dataset(
                output_path,
                conditions,
                topologies,
                compliances,
                load_dofs,
                load_values,
                args,
            )

        print(
            f"[{completed}/{args.num_samples}] "
            f"volfrac={condition[0]:.3f} load=({condition[1]:.2f}, {condition[2]:.2f}) "
            f"F=({load_val[0]:.3f}, {load_val[1]:.3f}) compliance={compliance:.3f}"
        )

    print(f"Saved {len(conditions)} samples to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/dataset.npz")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--nelx", type=int, default=60)
    parser.add_argument("--nely", type=int, default=30)
    parser.add_argument("--min-volfrac", type=float, default=0.2)
    parser.add_argument("--max-volfrac", type=float, default=0.6)
    parser.add_argument("--penal", type=float, default=3.0)
    parser.add_argument("--rmin", type=float, default=2.0)
    parser.add_argument("--ft", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--change-tol", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-load-norm",
        type=float,
        default=0.0,
        help="Optional minimum vector norm for sampled loads.",
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
    if args.nelx < 2 or args.nely < 2:
        raise ValueError("--nelx and --nely must both be at least 2")
    if not 0.0 < args.min_volfrac <= args.max_volfrac <= 1.0:
        raise ValueError("--min-volfrac and --max-volfrac must satisfy 0 < min <= max <= 1")
    if not 0.0 <= args.min_load_norm <= np.sqrt(2.0):
        raise ValueError("--min-load-norm must be in [0, sqrt(2)]")

    return args


def _sample_design_condition(
    nelx: int,
    nely: int,
    rng: np.random.Generator,
    min_volfrac: float,
    max_volfrac: float,
    min_load_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    load_x = int(rng.integers(nelx // 2, nelx))
    load_y = int(rng.integers(0, nely))
    load_node = load_x * (nely + 1) + load_y
    load_dof = np.array([2 * load_node, 2 * load_node + 1], dtype=np.int64)

    while True:
        load_val = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        if np.linalg.norm(load_val) >= min_load_norm:
            break

    volfrac = float(rng.uniform(min_volfrac, max_volfrac))
    normalized_load_x = load_x / max(1, nelx - 1)
    normalized_load_y = load_y / max(1, nely - 1)
    condition = np.array(
        [volfrac, normalized_load_x, normalized_load_y, load_val[0], load_val[1]],
        dtype=np.float32,
    )
    return load_dof, load_val, condition


def _load_existing_dataset(
    output_path: Path, num_samples: int, nelx: int, nely: int
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[float],
    list[np.ndarray],
    list[np.ndarray],
]:
    if not output_path.exists():
        return [], [], [], [], []

    with np.load(output_path) as data:
        conditions = data["conditions"]
        topologies = data["topologies"]
        compliances = data["compliances"]
        load_dofs = data["load_dofs"]
        load_values = data["load_values"]

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
    if conditions.shape[1:] != (5,):
        raise ValueError(
            f"Existing conditions shape {conditions.shape[1:]} is incompatible with "
            "the expected condition shape (5,)"
        )
    if load_dofs.shape[1:] != (2,) or load_values.shape[1:] != (2,):
        raise ValueError("Existing load_dofs and load_values must have shape (N, 2)")

    return (
        [row.astype(np.float32) for row in conditions],
        [topology.astype(np.float32) for topology in topologies],
        [float(value) for value in compliances],
        [row.astype(np.int64) for row in load_dofs],
        [row.astype(np.float32) for row in load_values],
    )


def _save_dataset(
    output_path: Path,
    conditions: list[np.ndarray],
    topologies: list[np.ndarray],
    compliances: list[float],
    load_dofs: list[np.ndarray],
    load_values: list[np.ndarray],
    args: argparse.Namespace,
) -> None:
    metadata = {
        "nelx": args.nelx,
        "nely": args.nely,
        "min_volfrac": args.min_volfrac,
        "max_volfrac": args.max_volfrac,
        "penal": args.penal,
        "rmin": args.rmin,
        "ft": args.ft,
        "max_iter": args.max_iter,
        "change_tol": args.change_tol,
        "seed": args.seed,
        "condition_schema": "volfrac,normalized_load_x,normalized_load_y,Fx,Fy",
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
            load_values=np.asarray(load_values, dtype=np.float32),
            **metadata,
        )

    os.replace(temp_path, output_path)


if __name__ == "__main__":
    main()

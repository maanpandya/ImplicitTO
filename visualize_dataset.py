"""Visualize optimized SIMP designs stored in a dataset .npz file."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)

    with np.load(dataset_path) as data:
        conditions = data["conditions"]
        topologies = data["topologies"]
        compliances = data["compliances"] if "compliances" in data else None
        load_dofs = data["load_dofs"] if "load_dofs" in data else None

    if len(topologies) == 0:
        raise ValueError(f"{dataset_path} does not contain any topologies")

    indices = _select_indices(len(topologies), args)
    _plot_samples(
        conditions=conditions,
        topologies=topologies,
        compliances=compliances,
        load_dofs=load_dofs,
        indices=indices,
        dataset_path=dataset_path,
        output_path=Path(args.output) if args.output else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/dataset.npz")
    parser.add_argument("--num", type=int, default=12, help="Number of designs to show.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument(
        "--indices",
        default=None,
        help="Comma-separated sample indices to show, e.g. '0,4,17'.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional PNG path. If omitted, an interactive window is shown.",
    )
    args = parser.parse_args()

    if args.num <= 0:
        raise ValueError("--num must be positive")

    return args


def _select_indices(num_samples: int, args: argparse.Namespace) -> np.ndarray:
    if args.indices:
        indices = np.array([int(idx.strip()) for idx in args.indices.split(",")])
        if np.any(indices < 0) or np.any(indices >= num_samples):
            raise ValueError(f"--indices must be within [0, {num_samples})")
        return indices

    count = min(args.num, num_samples)
    rng = np.random.default_rng(args.seed)
    return np.sort(rng.choice(num_samples, size=count, replace=False))


def _plot_samples(
    *,
    conditions: np.ndarray,
    topologies: np.ndarray,
    compliances: np.ndarray | None,
    load_dofs: np.ndarray | None,
    indices: np.ndarray,
    dataset_path: Path,
    output_path: Path | None,
) -> None:
    columns = min(4, len(indices))
    rows = math.ceil(len(indices) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 2.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, sample_idx in zip(axes, indices):
        topology = topologies[sample_idx]
        condition = conditions[sample_idx]
        nelx, nely = topology.shape

        ax.imshow(
            topology.T,
            cmap="gray_r",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        _draw_load_arrow(ax, condition, nelx, nely)

        title = _sample_title(sample_idx, condition, compliances, load_dofs)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(indices) :]:
        ax.axis("off")

    fig.suptitle(f"SIMP Dataset: {dataset_path}", fontsize=12)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def _draw_load_arrow(
    ax: plt.Axes, condition: np.ndarray, nelx: int, nely: int
) -> None:
    _, load_y, load_magnitude = condition
    y_coord = float(np.clip(load_y, 0.0, 1.0) * (nely - 1))
    direction = -1.0 if load_magnitude < 0 else 1.0
    arrow_length = max(1.0, 0.22 * nely) * direction

    ax.annotate(
        "",
        xy=(nelx - 1, y_coord + arrow_length),
        xytext=(nelx - 1, y_coord),
        arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 2.0},
    )
    ax.scatter([nelx - 1], [y_coord], s=16, c="tab:red")


def _sample_title(
    sample_idx: int,
    condition: np.ndarray,
    compliances: np.ndarray | None,
    load_dofs: np.ndarray | None,
) -> str:
    _, load_y, load_magnitude = condition
    lines = [f"idx={sample_idx}  y={load_y:.2f}  Fy={load_magnitude:.2f}"]

    details = []
    if compliances is not None:
        details.append(f"J={float(compliances[sample_idx]):.1f}")
    if load_dofs is not None:
        details.append(f"dof={int(load_dofs[sample_idx])}")
    if details:
        lines.append("  ".join(details))

    return "\n".join(lines)


if __name__ == "__main__":
    main()

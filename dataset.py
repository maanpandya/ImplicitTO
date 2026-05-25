"""PyTorch dataset utilities for neural implicit topology fields."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TopologyDataset(Dataset):
    """Serve random coordinate-density samples from optimized topologies."""

    def __init__(
        self,
        dataset_path: str | Path = "data/dataset.npz",
        points_per_sample: int = 1024,
        jitter_coordinates: bool = False,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.points_per_sample = points_per_sample
        self.jitter_coordinates = jitter_coordinates

        if points_per_sample <= 0:
            raise ValueError("points_per_sample must be positive")

        with np.load(self.dataset_path) as data:
            self.conditions = torch.from_numpy(data["conditions"].astype(np.float32))
            self.topologies = torch.from_numpy(data["topologies"].astype(np.float32))

        if self.topologies.ndim != 3:
            raise ValueError(
                "Expected topologies with shape (N, nelx, nely), "
                f"got {tuple(self.topologies.shape)}"
            )
        if self.conditions.ndim != 2:
            raise ValueError(
                "Expected conditions with shape (N, C), "
                f"got {tuple(self.conditions.shape)}"
            )
        if len(self.conditions) != len(self.topologies):
            raise ValueError(
                "conditions and topologies must have the same first dimension"
            )

        _, self.nelx, self.nely = self.topologies.shape
        self.coords = _make_normalized_coords(self.nelx, self.nely)
        self.num_grid_points = self.nelx * self.nely
        self.voxel_size = torch.tensor(
            [2.0 / self.nelx, 2.0 / self.nely],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.topologies)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.points_per_sample <= self.num_grid_points:
            point_indices = torch.randperm(self.num_grid_points)[
                : self.points_per_sample
            ]
        else:
            point_indices = torch.randint(
                self.num_grid_points,
                (self.points_per_sample,),
            )

        density = self.topologies[idx].reshape(-1)[point_indices].unsqueeze(-1)

        coords = self.coords[point_indices]
        if self.jitter_coordinates:
            jitter = (torch.rand_like(coords) - 0.5) * self.voxel_size
            coords = (coords + jitter).clamp(-1.0, 1.0)

        return {
            "coords": coords,
            "cond": self.conditions[idx],
            "gt_density": density,
        }


class FourierFeatures(nn.Module):
    """Gaussian random Fourier features for 2D coordinates."""

    def __init__(
        self,
        num_frequencies: int = 64,
        input_dim: int = 2,
        sigma: float = 10.0,
    ) -> None:
        super().__init__()

        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        b_matrix = torch.randn(input_dim, num_frequencies) * sigma
        self.register_buffer("B", b_matrix)

    @property
    def output_dim(self) -> int:
        return 2 * self.B.shape[1]

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[-1] != self.B.shape[0]:
            raise ValueError(
                f"Expected coords last dimension {self.B.shape[0]}, "
                f"got {coords.shape[-1]}"
            )

        projected = 2.0 * torch.pi * coords @ self.B
        return torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)


def _make_normalized_coords(nelx: int, nely: int) -> torch.Tensor:
    x = torch.linspace(-1.0, 1.0, nelx)
    y = torch.linspace(-1.0, 1.0, nely)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/dataset.npz")
    parser.add_argument("--points-per-sample", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frequencies", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument(
        "--jitter-coordinates",
        action="store_true",
        help="Apply voxel-scale coordinate jitter for training-time sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = TopologyDataset(
        dataset_path=args.dataset,
        points_per_sample=args.points_per_sample,
        jitter_coordinates=args.jitter_coordinates,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(dataloader))

    fourier = FourierFeatures(
        num_frequencies=args.num_frequencies,
        sigma=args.sigma,
    )
    encoded_coords = fourier(batch["coords"])

    print(f"coords: {tuple(batch['coords'].shape)}")
    print(f"cond: {tuple(batch['cond'].shape)}")
    print(f"gt_density: {tuple(batch['gt_density'].shape)}")
    print(f"fourier(coords): {tuple(encoded_coords.shape)}")


if __name__ == "__main__":
    main()

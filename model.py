"""Conditional neural implicit field for topology optimization."""

from __future__ import annotations

import argparse

import torch
from torch import nn

from dataset import FourierFeatures


class FiLMLayer(nn.Module):
    """Feature-wise linear modulation conditioned on design parameters."""

    def __init__(self, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()

        if cond_dim <= 0:
            raise ValueError("cond_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.hidden_dim = hidden_dim
        self.modulation = nn.Linear(cond_dim, 2 * hidden_dim)
        self._initialize_identity_modulation()

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim == 1:
            cond = cond.unsqueeze(0)

        gamma, beta = self.modulation(cond).chunk(2, dim=-1)
        while gamma.ndim < hidden.ndim:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return gamma * hidden + beta

    def _initialize_identity_modulation(self) -> None:
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        with torch.no_grad():
            self.modulation.bias[: self.hidden_dim].fill_(1.0)


class ConditionalImplicitNetwork(nn.Module):
    """MLP mapping coordinates and design conditions to material density."""

    def __init__(
        self,
        cond_dim: int = 5,
        hidden_dim: int = 256,
        num_hidden_layers: int = 6,
        num_frequencies: int = 5,
        fourier_sigma: float = 1.0,
        activation: str = "silu",
        encoding_type: str = "positional",
    ) -> None:
        super().__init__()

        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")

        normalized_encoding = encoding_type.lower()
        if normalized_encoding == "positional":
            self.fourier = PositionalEncoding(input_dim=2, max_frequency=num_frequencies)
        elif normalized_encoding == "gaussian":
            self.fourier = FourierFeatures(
                input_dim=2,
                num_frequencies=num_frequencies,
                sigma=fourier_sigma,
            )
        else:
            raise ValueError(f"Unsupported encoding_type: {encoding_type}")
        self.hidden_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()

        input_dim = self.fourier.output_dim
        for layer_idx in range(num_hidden_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            self.hidden_layers.append(nn.Linear(in_dim, hidden_dim))
            self.film_layers.append(FiLMLayer(cond_dim, hidden_dim))

        self.activation = _make_activation(activation)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, coords: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        hidden = self.fourier(coords)

        for linear, film in zip(self.hidden_layers, self.film_layers):
            hidden = linear(hidden)
            hidden = film(hidden, cond)
            hidden = self.activation(hidden)

        density = self.output_layer(hidden)
        return self.output_activation(density)


class PositionalEncoding(nn.Module):
    """Deterministic NeRF-style positional encoding for bounded coordinates."""

    def __init__(self, input_dim: int = 2, max_frequency: int = 5) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if max_frequency < 0:
            raise ValueError("max_frequency must be non-negative")

        self.input_dim = input_dim
        self.max_frequency = max_frequency
        frequency_bands = 2.0 ** torch.arange(max_frequency + 1, dtype=torch.float32)
        self.register_buffer("frequency_bands", frequency_bands)

    @property
    def output_dim(self) -> int:
        return 2 * self.input_dim * len(self.frequency_bands)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected coords last dimension {self.input_dim}, "
                f"got {coords.shape[-1]}"
            )

        encoded = torch.pi * coords.unsqueeze(-1) * self.frequency_bands
        encoded = torch.cat([torch.sin(encoded), torch.cos(encoded)], dim=-1)
        return encoded.flatten(start_dim=-2)


def _make_activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--points-per-sample", type=int, default=1024)
    parser.add_argument("--cond-dim", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=6)
    parser.add_argument("--num-frequencies", type=int, default=5)
    parser.add_argument("--fourier-sigma", type=float, default=1.0)
    parser.add_argument("--encoding-type", choices=("positional", "gaussian"), default="positional")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = ConditionalImplicitNetwork(
        cond_dim=args.cond_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        num_frequencies=args.num_frequencies,
        fourier_sigma=args.fourier_sigma,
        encoding_type=args.encoding_type,
    )
    coords = torch.rand(args.batch_size, args.points_per_sample, 2) * 2.0 - 1.0
    cond = torch.randn(args.batch_size, args.cond_dim)

    with torch.no_grad():
        density = model(coords, cond)

    print(f"coords: {tuple(coords.shape)}")
    print(f"cond: {tuple(cond.shape)}")
    print(f"density: {tuple(density.shape)}")
    print(f"density range: [{density.min().item():.4f}, {density.max().item():.4f}]")


if __name__ == "__main__":
    main()

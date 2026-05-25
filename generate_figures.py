"""Generate publication-ready figures for the amortized topology paper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from dataset import TopologyDataset
from eval import generate_topology, load_model, timed_generate_topology
from SIMP import compute_compliance, solve_simp
from train import get_device, split_dataset


DEFAULT_FIGURE_DIR = Path("figures")


@dataclass(frozen=True)
class FigureDataset:
    """Dense dataset arrays needed for paper figure generation."""

    conditions: np.ndarray
    topologies: np.ndarray
    compliances: np.ndarray | None
    load_dofs: np.ndarray | None
    load_values: np.ndarray | None
    nelx: int
    nely: int

    def subset(self, indices: np.ndarray) -> "FigureDataset":
        return FigureDataset(
            conditions=self.conditions[indices],
            topologies=self.topologies[indices],
            compliances=None if self.compliances is None else self.compliances[indices],
            load_dofs=None if self.load_dofs is None else self.load_dofs[indices],
            load_values=None if self.load_values is None else self.load_values[indices],
            nelx=self.nelx,
            nely=self.nely,
        )

    def __len__(self) -> int:
        return len(self.conditions)


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, metadata = load_model(Path(args.checkpoint), device)
    dataset = load_figure_dataset(Path(args.dataset))
    test_dataset = make_test_dataset(dataset, Path(args.dataset), args.val_fraction, args.seed)

    nelx = int(metadata["nelx"])
    nely = int(metadata["nely"])
    if (dataset.nelx, dataset.nely) != (nelx, nely):
        raise ValueError(
            f"Dataset grid {(dataset.nelx, dataset.nely)} does not match "
            f"checkpoint grid {(nelx, nely)}"
        )

    simp_solver = make_simp_solver(
        penal=args.penal,
        rmin=args.rmin,
        ft=args.ft,
        max_iter=args.simp_max_iter,
        change_tol=args.simp_change_tol,
    )

    plot_qualitative_matrix(
        model,
        test_dataset,
        num_samples=args.qualitative_samples,
        output_dir=output_dir,
        device=device,
    )
    plot_speed_vs_compliance(
        model,
        simp_solver,
        test_dataset.conditions,
        nelx=nelx,
        nely=nely,
        num_cases=args.speed_cases,
        output_dir=output_dir,
        device=device,
        inference_repeats=args.inference_repeats,
        seed=args.seed,
        penal=args.penal,
    )
    base_condition = pick_representative_condition(test_dataset.conditions)
    plot_continuous_morphing(
        model,
        base_condition,
        nelx=nelx,
        nely=nely,
        output_dir=output_dir,
        device=device,
    )
    plot_super_resolution(
        model,
        base_condition,
        nelx=nelx,
        nely=nely,
        output_dir=output_dir,
        device=device,
        simp_solver=simp_solver,
        scale=args.super_res_scale,
    )
    plot_error_histogram(
        model,
        test_dataset,
        num_cases=args.histogram_cases,
        output_dir=output_dir,
        device=device,
        seed=args.seed,
        penal=args.penal,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/dataset.npz")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--output-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--qualitative-samples", type=int, default=4)
    parser.add_argument("--speed-cases", type=int, default=50)
    parser.add_argument("--histogram-cases", type=int, default=200)
    parser.add_argument("--inference-repeats", type=int, default=20)
    parser.add_argument("--super-res-scale", type=int, default=4)
    parser.add_argument("--penal", type=float, default=3.0)
    parser.add_argument("--rmin", type=float, default=2.0)
    parser.add_argument("--ft", type=int, choices=(0, 1), default=1)
    parser.add_argument("--simp-max-iter", type=int, default=200)
    parser.add_argument("--simp-change-tol", type=float, default=0.01)
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    for name in (
        "qualitative_samples",
        "speed_cases",
        "histogram_cases",
        "inference_repeats",
        "super_res_scale",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    return args


def configure_matplotlib() -> None:
    """Use paper-friendly typography without requiring a LaTeX install."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "text.usetex": False,
        }
    )


def load_figure_dataset(dataset_path: Path) -> FigureDataset:
    with np.load(dataset_path) as data:
        topologies = data["topologies"].astype(np.float32)
        conditions = data["conditions"].astype(np.float32)
        compliances = data["compliances"].astype(np.float32) if "compliances" in data else None
        load_dofs = data["load_dofs"].astype(np.int64) if "load_dofs" in data else None
        load_values = data["load_values"].astype(np.float32) if "load_values" in data else None

    if topologies.ndim != 3:
        raise ValueError(f"Expected topologies with shape (N, nelx, nely), got {topologies.shape}")
    if conditions.ndim != 2 or len(conditions) != len(topologies):
        raise ValueError("conditions must have shape (N, C) and match topologies")
    _, nelx, nely = topologies.shape
    return FigureDataset(conditions, topologies, compliances, load_dofs, load_values, nelx, nely)


def make_test_dataset(
    dataset: FigureDataset,
    dataset_path: Path,
    val_fraction: float,
    seed: int,
) -> FigureDataset:
    topology_dataset = TopologyDataset(
        dataset_path=dataset_path,
        points_per_sample=1,
    )
    if len(topology_dataset) != len(dataset):
        raise ValueError("Loaded dataset length changed while constructing the validation split")
    _, val_subset = split_dataset(topology_dataset, val_fraction, seed)
    return dataset.subset(np.asarray(val_subset.indices, dtype=np.int64))


def make_simp_solver(
    *,
    penal: float,
    rmin: float,
    ft: int,
    max_iter: int,
    change_tol: float,
) -> Callable[[np.ndarray, int, int], tuple[np.ndarray, float, float]]:
    def _solver(condition: np.ndarray, nelx: int, nely: int) -> tuple[np.ndarray, float, float]:
        load_dof, load_val = condition_to_load(condition, nelx, nely)
        start = time.perf_counter()
        density, compliance = solve_simp(
            nelx,
            nely,
            float(condition[0]),
            penal,
            rmin,
            ft,
            load_dof,
            load_val,
            max_iter=max_iter,
            change_tol=change_tol,
        )
        return density, compliance, time.perf_counter() - start

    return _solver


@torch.no_grad()
def predict_density(
    model: torch.nn.Module,
    condition: np.ndarray,
    nelx: int,
    nely: int,
    device: torch.device,
) -> np.ndarray:
    return generate_topology(model, condition, nelx, nely, device)


def plot_qualitative_matrix(
    model: torch.nn.Module,
    test_dataset: FigureDataset,
    num_samples: int = 4,
    *,
    output_dir: Path = DEFAULT_FIGURE_DIR,
    device: torch.device | None = None,
) -> None:
    """Plot SIMP, continuous prediction, and signed density error."""

    device = device or next(model.parameters()).device
    indices = select_diverse_indices(test_dataset.conditions, min(num_samples, len(test_dataset)))
    rows = len(indices)
    fig, axes = plt.subplots(rows, 3, figsize=(7.2, 1.75 * rows), squeeze=False)

    for row, sample_idx in enumerate(indices):
        condition = test_dataset.conditions[sample_idx]
        gt = test_dataset.topologies[sample_idx]
        pred = predict_density(model, condition, test_dataset.nelx, test_dataset.nely, device)
        error = pred - gt

        images = (gt, pred, error)
        titles = ("SIMP ground truth", "Neural prediction", "Signed error")
        for col, (image, title) in enumerate(zip(images, titles)):
            ax = axes[row, col]
            if col < 2:
                ax.imshow(image.T, cmap="gray_r", origin="lower", vmin=0.0, vmax=1.0)
                draw_load_arrow(ax, condition, test_dataset.nelx, test_dataset.nely)
            else:
                ax.imshow(
                    image.T,
                    cmap="coolwarm",
                    origin="lower",
                    norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
                )
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(condition_label(condition), labelpad=2)
            clean_axis(ax)

    fig.tight_layout(pad=0.35)
    save_pdf(fig, output_dir / "fig_qualitative.pdf")


def plot_speed_vs_compliance(
    model: torch.nn.Module,
    simp_solver_func: Callable[[np.ndarray, int, int], tuple[np.ndarray, float, float]],
    test_conditions: np.ndarray,
    *,
    nelx: int,
    nely: int,
    num_cases: int = 50,
    output_dir: Path = DEFAULT_FIGURE_DIR,
    device: torch.device | None = None,
    inference_repeats: int = 20,
    seed: int = 0,
    penal: float = 3.0,
) -> None:
    """Compare SIMP and neural prediction runtimes against compliance."""

    device = device or next(model.parameters()).device
    rng = np.random.default_rng(seed)
    case_indices = rng.choice(len(test_conditions), size=min(num_cases, len(test_conditions)), replace=False)

    simp_times: list[float] = []
    simp_compliances: list[float] = []
    nn_times: list[float] = []
    nn_compliances: list[float] = []

    for case_idx in case_indices:
        condition = test_conditions[case_idx]
        _, simp_compliance, simp_seconds = simp_solver_func(condition, nelx, nely)
        pred_density, nn_seconds = timed_generate_topology(
            model,
            condition,
            nelx,
            nely,
            device,
            repeats=inference_repeats,
        )
        load_dof, load_val = condition_to_load(condition, nelx, nely)
        pred_compliance = compute_compliance(pred_density, load_dof, load_val, penal=penal)

        simp_times.append(simp_seconds)
        simp_compliances.append(simp_compliance)
        nn_times.append(nn_seconds)
        nn_compliances.append(pred_compliance)

    fig, ax = plt.subplots(figsize=(4.7, 3.25))
    ax.scatter(simp_times, simp_compliances, s=32, marker="o", label="SIMP", color="#4c78a8")
    ax.scatter(nn_times, nn_compliances, s=32, marker="^", label="Neural field", color="#f58518")
    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Compliance")
    ax.grid(True, which="both", alpha=0.28, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_pdf(fig, output_dir / "fig_speed_pareto.pdf")


def plot_continuous_morphing(
    model: torch.nn.Module,
    base_condition: np.ndarray,
    *,
    nelx: int,
    nely: int,
    output_dir: Path = DEFAULT_FIGURE_DIR,
    device: torch.device | None = None,
) -> None:
    """Visualize continuous changes under volume and load-angle sweeps."""

    device = device or next(model.parameters()).device
    fig, axes = plt.subplots(2, 5, figsize=(8.3, 3.15))

    volfracs = np.linspace(0.2, 0.6, 5)
    for col, volfrac in enumerate(volfracs):
        condition = base_condition.copy()
        condition[0] = volfrac
        density = predict_density(model, condition, nelx, nely, device)
        axes[0, col].imshow(density.T, cmap="gray_r", origin="lower", vmin=0.0, vmax=1.0)
        draw_load_arrow(axes[0, col], condition, nelx, nely)
        axes[0, col].set_title(rf"$v={volfrac:.2f}$")
        clean_axis(axes[0, col])

    angles = np.linspace(0.0, np.pi / 2.0, 5)
    force_norm = max(float(np.linalg.norm(base_condition[3:5])), 1e-6)
    for col, angle in enumerate(angles):
        condition = base_condition.copy()
        condition[3] = force_norm * np.cos(angle)
        condition[4] = force_norm * np.sin(angle)
        density = predict_density(model, condition, nelx, nely, device)
        axes[1, col].imshow(density.T, cmap="gray_r", origin="lower", vmin=0.0, vmax=1.0)
        draw_load_arrow(axes[1, col], condition, nelx, nely)
        axes[1, col].set_title(rf"$\theta={np.degrees(angle):.0f}^\circ$")
        clean_axis(axes[1, col])

    axes[0, 0].set_ylabel("Volume sweep")
    axes[1, 0].set_ylabel("Angle sweep")
    fig.tight_layout(pad=0.25)
    save_pdf(fig, output_dir / "fig_morphing.pdf")


def plot_super_resolution(
    model: torch.nn.Module,
    sample_condition: np.ndarray,
    *,
    nelx: int,
    nely: int,
    output_dir: Path = DEFAULT_FIGURE_DIR,
    device: torch.device | None = None,
    simp_solver: Callable[[np.ndarray, int, int], tuple[np.ndarray, float, float]] | None = None,
    scale: int = 4,
) -> None:
    """Show the same condition as coarse SIMP pixels and dense neural queries."""

    device = device or next(model.parameters()).device
    if simp_solver is None:
        simp_solver = make_simp_solver(penal=3.0, rmin=2.0, ft=1, max_iter=200, change_tol=0.01)

    simp_density, _, _ = simp_solver(sample_condition, nelx, nely)
    hi_nelx = nelx * scale
    hi_nely = nely * scale

    # Match dataset._make_normalized_coords exactly, only with more samples
    # along the same normalized coordinate interval.
    x_coords = torch.linspace(-1.0, 1.0, hi_nelx, device=device)
    y_coords = torch.linspace(-1.0, 1.0, hi_nely, device=device)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).float()
    num_points = hi_nelx * hi_nely
    condition = torch.as_tensor(sample_condition, dtype=torch.float32, device=device)
    conditions = condition.unsqueeze(0).repeat(num_points, 1)

    model.eval()
    with torch.no_grad():
        neural_density_tensor = model(coords, conditions).squeeze(-1)
    neural_density = neural_density_tensor.reshape(hi_nelx, hi_nely).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    axes[0].imshow(simp_density.T, cmap="gray_r", origin="lower", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"SIMP ({nelx}x{nely})")
    axes[1].imshow(
        neural_density.T,
        cmap="gray_r",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    axes[1].set_title(f"Neural implicit ({hi_nelx}x{hi_nely})")
    for ax in axes:
        clean_axis(ax)

    add_zoom_inset(axes[0], simp_density, interpolation="nearest")
    add_zoom_inset(axes[1], neural_density, interpolation="bilinear")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.04, wspace=0.08)
    save_pdf(fig, output_dir / "fig_super_res.pdf")


def plot_error_histogram(
    model: torch.nn.Module,
    test_dataset: FigureDataset,
    *,
    num_cases: int = 200,
    output_dir: Path = DEFAULT_FIGURE_DIR,
    device: torch.device | None = None,
    seed: int = 0,
    penal: float = 3.0,
) -> None:
    """Plot compliance error percentages for held-out test samples."""

    device = device or next(model.parameters()).device
    rng = np.random.default_rng(seed)
    count = min(num_cases, len(test_dataset))
    case_indices = rng.choice(len(test_dataset), size=count, replace=False)
    errors = []

    for case_idx in case_indices:
        condition = test_dataset.conditions[case_idx]
        pred_density = predict_density(model, condition, test_dataset.nelx, test_dataset.nely, device)
        load_dof, load_val = sample_load(test_dataset, case_idx)
        pred_compliance = compute_compliance(pred_density, load_dof, load_val, penal=penal)

        if test_dataset.compliances is not None:
            simp_compliance = float(test_dataset.compliances[case_idx])
        else:
            simp_compliance = compute_compliance(
                test_dataset.topologies[case_idx],
                load_dof,
                load_val,
                penal=penal,
            )
        relative_error = (pred_compliance - simp_compliance) / max(abs(simp_compliance), 1e-12)
        errors.append(100.0 * abs(relative_error))

    error_values = np.asarray(errors, dtype=float)
    mean_error = float(error_values.mean())
    std_error = float(error_values.std())
    x_min, x_max = smart_histogram_limits(error_values)
    x_min = max(0.0, x_min)
    visible_errors = error_values[(error_values >= x_min) & (error_values <= x_max)]
    bin_edges = smart_histogram_bins(visible_errors, x_min, x_max)
    clipped_count = int(len(error_values) - len(visible_errors))

    fig, ax = plt.subplots(figsize=(4.7, 3.1))
    ax.hist(visible_errors, bins=bin_edges, color="#4c78a8", edgecolor="white", linewidth=0.6)
    std_left = max(x_min, mean_error - std_error)
    std_right = min(x_max, mean_error + std_error)
    if std_left < std_right:
        sigma_label = rf"$\pm 1\sigma$ = {std_error:.1f}%"
        ax.axvline(
            std_left,
            color="#f58518",
            linestyle="-",
            linewidth=1.4,
            label=sigma_label,
        )
        ax.axvline(std_right, color="#f58518", linestyle="-", linewidth=1.4)
    if x_min <= mean_error <= x_max:
        ax.axvline(
            mean_error,
            color="#b2182b",
            linestyle="--",
            linewidth=1.4,
            label=f"mean absolute = {mean_error:.1f}%",
        )
    ax.legend(frameon=False)
    if clipped_count:
        ax.text(
            0.98,
            0.94,
            f"{clipped_count} outlier{'s' if clipped_count != 1 else ''} outside view",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="0.35",
        )
    ax.set_xlabel("Absolute Compliance Error (%)")
    ax.set_ylabel("Frequency")
    ax.set_xlim(x_min, x_max)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    save_pdf(fig, output_dir / "fig_histogram.pdf")


def condition_to_load(condition: np.ndarray, nelx: int, nely: int) -> tuple[np.ndarray, np.ndarray]:
    load_x = int(round(float(np.clip(condition[1], 0.0, 1.0)) * (nelx - 1)))
    load_y = int(round(float(np.clip(condition[2], 0.0, 1.0)) * (nely - 1)))
    load_node = load_x * (nely + 1) + load_y
    load_dof = np.array([2 * load_node, 2 * load_node + 1], dtype=np.int64)
    load_val = np.asarray(condition[3:5], dtype=np.float32)
    return load_dof, load_val


def sample_load(dataset: FigureDataset, sample_idx: int) -> tuple[np.ndarray, np.ndarray]:
    if dataset.load_dofs is not None and dataset.load_values is not None:
        return dataset.load_dofs[sample_idx], dataset.load_values[sample_idx]
    return condition_to_load(dataset.conditions[sample_idx], dataset.nelx, dataset.nely)


def pick_representative_condition(conditions: np.ndarray) -> np.ndarray:
    center = conditions.mean(axis=0)
    distances = np.linalg.norm(conditions - center, axis=1)
    return conditions[int(np.argmin(distances))].copy()


def smart_histogram_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0

    lower, upper = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower = float(finite.min())
        upper = float(finite.max())
    if lower == upper:
        pad = max(1.0, abs(lower) * 0.1)
        return lower - pad, upper + pad

    iqr = np.subtract(*np.percentile(finite, [75.0, 25.0]))
    pad = max(0.08 * (upper - lower), 0.25 * max(float(iqr), 1.0))
    return float(lower - pad), float(upper + pad)


def smart_histogram_bins(values: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    if values.size < 2 or x_min == x_max:
        return np.linspace(x_min, x_max, 12)

    q25, q75 = np.percentile(values, [25.0, 75.0])
    iqr = float(q75 - q25)
    width = 2.0 * iqr / np.cbrt(values.size) if iqr > 0.0 else 0.0
    if width <= 0.0:
        width = max((x_max - x_min) / 30.0, 1.0)
    num_bins = int(np.clip(np.ceil((x_max - x_min) / width), 25, 70))
    return np.linspace(x_min, x_max, num_bins + 1)


def select_diverse_indices(conditions: np.ndarray, count: int) -> np.ndarray:
    if count >= len(conditions):
        return np.arange(len(conditions))

    normalized = conditions.astype(float).copy()
    span = np.ptp(normalized, axis=0)
    normalized = (normalized - normalized.mean(axis=0)) / np.maximum(span, 1e-12)
    selected = [int(np.argmin(np.linalg.norm(normalized, axis=1)))]
    while len(selected) < count:
        distances = np.min(
            np.linalg.norm(normalized[:, None, :] - normalized[np.asarray(selected)][None, :, :], axis=2),
            axis=1,
        )
        distances[selected] = -np.inf
        selected.append(int(np.argmax(distances)))
    return np.asarray(selected, dtype=np.int64)


def draw_load_arrow(ax: plt.Axes, condition: np.ndarray, nelx: int, nely: int) -> None:
    x_coord = float(np.clip(condition[1], 0.0, 1.0) * (nelx - 1))
    y_coord = float(np.clip(condition[2], 0.0, 1.0) * (nely - 1))
    force = np.asarray(condition[3:5], dtype=float)
    force_norm = max(float(np.linalg.norm(force)), 1e-12)
    arrow = force / force_norm * max(1.0, 0.18 * min(nelx, nely))
    ax.annotate(
        "",
        xy=(x_coord + arrow[0], y_coord + arrow[1]),
        xytext=(x_coord, y_coord),
        arrowprops={"arrowstyle": "-|>", "color": "#d62728", "lw": 1.6, "mutation_scale": 8},
    )
    ax.scatter([x_coord], [y_coord], s=14, c="#d62728", edgecolors="white", linewidths=0.3)


def add_zoom_inset(ax: plt.Axes, image: np.ndarray, *, interpolation: str) -> None:
    nelx, nely = image.shape
    x0 = int(0.58 * nelx)
    x1 = min(nelx, x0 + max(8, nelx // 5))
    y0 = int(0.30 * nely)
    y1 = min(nely, y0 + max(6, nely // 4))

    inset = inset_axes(ax, width="42%", height="42%", loc="lower left", borderpad=0.9)
    inset.imshow(
        image[x0:x1, y0:y1].T,
        cmap="gray_r",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        interpolation=interpolation,
    )
    clean_axis(inset)
    ax.indicate_inset_zoom(inset, edgecolor="0.25", linewidth=0.8)
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="0.25", lw=0.6)


def condition_label(condition: np.ndarray) -> str:
    return (
        f"v={condition[0]:.2f}\n"
        f"F=({condition[3]:+.1f},{condition[4]:+.1f})"
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("0.35")


def save_pdf(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, bbox_inches="tight")
    except Exception:
        if not plt.rcParams["text.usetex"]:
            raise
        plt.rcParams["text.usetex"] = False
        fig.savefig(output_path, bbox_inches="tight")
        print("Matplotlib TeX rendering failed; fell back to serif fonts.")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

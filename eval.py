"""Evaluate trained neural implicit topology models against SIMP."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import _make_normalized_coords
from model import ConditionalImplicitNetwork
from SIMP import compute_compliance, solve_simp
from train import get_device


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    model, metadata = load_model(Path(args.checkpoint), device)
    nelx = int(metadata["nelx"])
    nely = int(metadata["nely"])
    model_args = metadata["model_args"]

    rng = np.random.default_rng(args.seed)
    num_plot_slots = args.num_plot_cases * args.num_plot_versions
    total_cases = max(args.num_cases, num_plot_slots)
    cases = [
        sample_unseen_case(
            nelx,
            nely,
            rng,
            args.min_volfrac,
            args.max_volfrac,
            args.min_load_norm,
        )
        for _ in range(total_cases)
    ]

    results = []
    plot_case_groups = [[] for _ in range(args.num_plot_versions)]
    for case_idx, case in enumerate(cases, start=1):
        condition, load_dof, load_val = case

        pred_density, inference_seconds = timed_generate_topology(
            model,
            condition,
            nelx,
            nely,
            device,
            repeats=args.inference_repeats,
        )

        simp_start = time.perf_counter()
        simp_density, simp_compliance = solve_simp(
            nelx,
            nely,
            float(condition[0]),
            args.penal,
            args.rmin,
            args.ft,
            load_dof,
            load_val,
            max_iter=args.simp_max_iter,
            change_tol=args.simp_change_tol,
        )
        simp_seconds = time.perf_counter() - simp_start

        pred_compliance = compute_compliance(
            pred_density,
            load_dof,
            load_val,
            penal=args.penal,
        )
        compliance_error = (
            (pred_compliance - simp_compliance) / max(abs(simp_compliance), 1e-12)
        )

        result = {
            "case": case_idx,
            "target_volfrac": float(condition[0]),
            "load_x": float(condition[1]),
            "load_y": float(condition[2]),
            "load_fx": float(load_val[0]),
            "load_fy": float(load_val[1]),
            "nn_time_ms": inference_seconds * 1_000.0,
            "simp_time_s": simp_seconds,
            "speedup": simp_seconds / max(inference_seconds, 1e-12),
            "pred_volfrac": volume_fraction(pred_density),
            "simp_compliance": simp_compliance,
            "pred_compliance": pred_compliance,
            "compliance_error_pct": 100.0 * compliance_error,
        }
        results.append(result)

        version_idx = (case_idx - 1) // max(args.num_plot_cases, 1)
        if args.num_plot_cases > 0 and version_idx < args.num_plot_versions:
            plot_case_groups[version_idx].append(
                (case_idx, simp_density, pred_density, result)
            )

    print_report(results, model_args)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for version_idx, plot_cases in enumerate(plot_case_groups, start=1):
            if not plot_cases:
                continue
            output_path = (
                output_dir / "eval_comparisons.png"
                if version_idx == 1
                else output_dir / f"eval_comparisons_{version_idx:02d}.png"
            )
            save_comparison_plot(
                plot_cases,
                output_path,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-volfrac", type=float, default=0.2)
    parser.add_argument("--max-volfrac", type=float, default=0.6)
    parser.add_argument("--penal", type=float, default=3.0)
    parser.add_argument("--rmin", type=float, default=2.0)
    parser.add_argument("--ft", type=int, choices=(0, 1), default=1)
    parser.add_argument("--simp-max-iter", type=int, default=200)
    parser.add_argument("--simp-change-tol", type=float, default=0.01)
    parser.add_argument("--min-load-norm", type=float, default=0.0)
    parser.add_argument("--inference-repeats", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--num-plot-cases", type=int, default=4)
    parser.add_argument("--num-plot-versions", type=int, default=5)
    args = parser.parse_args()

    if args.num_cases <= 0:
        raise ValueError("--num-cases must be positive")
    if args.min_load_norm < 0.0:
        raise ValueError("--min-load-norm cannot be negative")
    if args.inference_repeats <= 0:
        raise ValueError("--inference-repeats must be positive")
    if args.num_plot_cases < 0:
        raise ValueError("--num-plot-cases cannot be negative")
    if args.num_plot_versions <= 0:
        raise ValueError("--num-plot-versions must be positive")
    if not 0.0 < args.min_volfrac <= args.max_volfrac <= 1.0:
        raise ValueError("--min-volfrac and --max-volfrac must satisfy 0 < min <= max <= 1")
    if args.min_load_norm > np.sqrt(2.0):
        raise ValueError("--min-load-norm must be at most sqrt(2)")

    return args


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[ConditionalImplicitNetwork, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    state_dict = checkpoint["model_state_dict"]
    encoding_type = train_args.get("encoding_type")
    if encoding_type is None:
        encoding_type = "gaussian" if "fourier.B" in state_dict else "positional"

    model_args = {
        "cond_dim": int(checkpoint.get("condition_dim", 5)),
        "hidden_dim": int(train_args.get("hidden_dim", 256)),
        "num_hidden_layers": int(train_args.get("num_hidden_layers", 6)),
        "num_frequencies": int(train_args.get("num_frequencies", 64 if encoding_type == "gaussian" else 5)),
        "fourier_sigma": float(train_args.get("fourier_sigma", 10.0 if encoding_type == "gaussian" else 1.0)),
        "activation": train_args.get("activation", "silu"),
        "encoding_type": encoding_type,
    }
    model = ConditionalImplicitNetwork(**model_args).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    metadata = {
        "nelx": int(checkpoint["nelx"]),
        "nely": int(checkpoint["nely"]),
        "model_args": model_args,
    }
    print(f"Loaded checkpoint {checkpoint_path} from epoch {checkpoint.get('epoch')}")
    return model, metadata


def sample_unseen_case(
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
    condition = np.array(
        [
            volfrac,
            load_x / max(1, nelx - 1),
            load_y / max(1, nely - 1),
            load_val[0],
            load_val[1],
        ],
        dtype=np.float32,
    )
    return condition, load_dof, load_val


@torch.no_grad()
def generate_topology(
    model: ConditionalImplicitNetwork,
    condition: np.ndarray | torch.Tensor,
    nelx: int,
    nely: int,
    device: torch.device | None = None,
) -> np.ndarray:
    """Generate a full density grid for one load condition."""

    model_device = device or next(model.parameters()).device
    coords = _make_normalized_coords(nelx, nely).to(model_device).unsqueeze(0)
    cond = torch.as_tensor(condition, dtype=torch.float32, device=model_device).unsqueeze(0)
    density = model(coords, cond).squeeze(0).squeeze(-1)
    return density.reshape(nelx, nely).detach().cpu().numpy()


def timed_generate_topology(
    model: ConditionalImplicitNetwork,
    condition: np.ndarray,
    nelx: int,
    nely: int,
    device: torch.device,
    repeats: int,
) -> tuple[np.ndarray, float]:
    synchronize_device(device)
    density = generate_topology(model, condition, nelx, nely, device)
    synchronize_device(device)

    start = time.perf_counter()
    for _ in range(repeats):
        density = generate_topology(model, condition, nelx, nely, device)
    synchronize_device(device)
    elapsed = time.perf_counter() - start
    return density, elapsed / repeats


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def volume_fraction(density: np.ndarray) -> float:
    return float(np.mean(density))


def print_report(results: list[dict[str, float]], model_args: dict) -> None:
    print("\nModel configuration:")
    print(", ".join(f"{key}={value}" for key, value in model_args.items()))
    print()

    headers = [
        "case",
        "target",
        "x",
        "y",
        "Fx",
        "Fy",
        "NN ms",
        "SIMP s",
        "speedup",
        "vol",
        "SIMP J",
        "NN J",
        "J err %",
    ]
    print(
        f"{headers[0]:>4} {headers[1]:>6} {headers[2]:>5} {headers[3]:>5} "
        f"{headers[4]:>7} {headers[5]:>7} {headers[6]:>9} {headers[7]:>8} "
        f"{headers[8]:>9} {headers[9]:>7} "
        f"{headers[10]:>10} {headers[11]:>10} {headers[12]:>9}"
    )
    for row in results:
        print(
            f"{int(row['case']):4d} {row['target_volfrac']:6.3f} "
            f"{row['load_x']:5.2f} {row['load_y']:5.2f} "
            f"{row['load_fx']:7.3f} {row['load_fy']:7.3f} "
            f"{row['nn_time_ms']:9.3f} {row['simp_time_s']:8.3f} "
            f"{row['speedup']:9.1f} {row['pred_volfrac']:7.3f} "
            f"{row['simp_compliance']:10.2f} {row['pred_compliance']:10.2f} "
            f"{row['compliance_error_pct']:9.2f}"
        )

    print("\nAverages:")
    for key, label in (
        ("nn_time_ms", "NN inference (ms)"),
        ("simp_time_s", "SIMP runtime (s)"),
        ("speedup", "speedup"),
        ("target_volfrac", "target volume fraction"),
        ("pred_volfrac", "predicted volume fraction"),
        ("compliance_error_pct", "compliance error (%)"),
    ):
        values = np.array([row[key] for row in results], dtype=float)
        print(f"{label}: mean={values.mean():.4f}, std={values.std():.4f}")


def save_comparison_plot(
    plot_cases: list[tuple[int, np.ndarray, np.ndarray, dict[str, float]]],
    output_path: Path,
) -> None:
    if not plot_cases:
        return

    rows = len(plot_cases)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.titlesize": 13,
        }
    )
    fig = plt.figure(figsize=(11.8, 2.35 * rows + 0.45))
    grid = fig.add_gridspec(
        rows,
        5,
        width_ratios=[1.0, 1.0, 1.0, 0.055, 0.95],
        wspace=0.08,
        hspace=0.22,
    )
    axes = np.empty((rows, 5), dtype=object)
    cbar_ax = fig.add_subplot(grid[:, 3])
    for row_idx in range(rows):
        for col_idx in (0, 1, 2, 4):
            axes[row_idx, col_idx] = fig.add_subplot(grid[row_idx, col_idx])
        axes[row_idx, 3] = cbar_ax

    density_columns = (0, 1)
    error_artist = None

    for row_idx, (case_idx, simp_density, pred_density, result) in enumerate(plot_cases):
        error_density = np.abs(pred_density - simp_density)
        images = (simp_density, pred_density, error_density)

        for col_idx, image in enumerate(images):
            ax = axes[row_idx][col_idx]
            cmap = "gray_r" if col_idx in density_columns else "magma"
            vmax = 1.0 if col_idx in density_columns else max(0.25, float(error_density.max()))
            artist = ax.imshow(
                image.T,
                cmap=cmap,
                origin="lower",
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
                aspect="auto",
                extent=(0.0, 1.0, 0.0, 1.0),
            )
            ax.set_xticks([0.0, 0.5, 1.0] if row_idx == rows - 1 else [])
            ax.set_yticks([0.0, 0.5, 1.0] if col_idx == 0 else [])
            ax.tick_params(length=2, width=0.6, labelsize=7)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("0.25")
            if col_idx in density_columns:
                _add_load_arrow(ax, result)
            else:
                error_artist = artist
                ax.text(
                    0.03,
                    0.06,
                    f"mean={error_density.mean():.3f}\nmax={error_density.max():.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    color="white",
                    fontsize=7,
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 2, "edgecolor": "none"},
                )

        axes[row_idx][0].set_ylabel(f"Case {case_idx}", fontweight="bold")
        _write_metric_panel(axes[row_idx][4], result)

    if error_artist is not None:
        cbar = fig.colorbar(
            error_artist,
            cax=cbar_ax,
        )
        cbar.set_label("|density error|")
        cbar.ax.tick_params(labelsize=7, length=2)

    fig.text(
        0.5,
        0.012,
        "Columns show SIMP reference, neural density prediction, and absolute density error. "
        "Red arrows indicate applied load location and direction.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.25",
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.965,
        top=0.965,
        bottom=0.08,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")


def _add_load_arrow(ax: plt.Axes, result: dict[str, float]) -> None:
    load_x = float(result["load_x"])
    load_y = float(result["load_y"])
    force = np.array([result["load_fx"], result["load_fy"]], dtype=float)
    norm = max(float(np.linalg.norm(force)), 1e-12)
    direction = force / norm
    arrow_scale = 0.16
    end_x = np.clip(load_x + arrow_scale * direction[0], 0.03, 0.97)
    end_y = np.clip(load_y + arrow_scale * direction[1], 0.06, 0.94)

    ax.annotate(
        "",
        xy=(end_x, end_y),
        xytext=(load_x, load_y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#d62728",
            "lw": 1.8,
            "mutation_scale": 9,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=5,
    )
    ax.scatter(
        [load_x],
        [load_y],
        s=13,
        c="#d62728",
        edgecolors="white",
        linewidths=0.35,
        zorder=6,
    )


def _write_metric_panel(ax: plt.Axes, result: dict[str, float]) -> None:
    ax.axis("off")
    compliance_error = result["compliance_error_pct"]
    metrics = (
        ("Target volume", f"{result['target_volfrac']:.3f}"),
        ("NN volume", f"{result['pred_volfrac']:.3f}"),
        ("SIMP compliance", f"{result['simp_compliance']:.2f}"),
        ("NN compliance", f"{result['pred_compliance']:.2f}"),
        ("Compliance error", f"{compliance_error:+.1f}%"),
        ("NN inference", f"{result['nn_time_ms']:.2f} ms"),
        ("SIMP solve", f"{result['simp_time_s']:.2f} s"),
        ("Speedup", f"{result['speedup']:.0f}x"),
    )

    y = 0.94
    for label, value in metrics:
        color = "#b2182b" if label == "Compliance error" and abs(compliance_error) > 25.0 else "0.1"
        ax.text(0.0, y, label, ha="left", va="top", color="0.35", fontsize=8)
        ax.text(1.0, y, value, ha="right", va="top", color=color, fontsize=8, fontweight="bold")
        y -= 0.095

    ax.text(
        0.0,
        0.01,
        f"load=({result['load_fx']:+.2f}, {result['load_fy']:+.2f})\n"
        f"location=({result['load_x']:.2f}, {result['load_y']:.2f})",
        ha="left",
        va="bottom",
        color="0.35",
        fontsize=7,
    )


if __name__ == "__main__":
    main()

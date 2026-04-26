"""Interactive topology generation app for the conditional implicit model."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from dataset import _make_normalized_coords
from eval import load_model, volume_fraction
from model import ConditionalImplicitNetwork
from train import get_device


DEFAULT_CHECKPOINT = "checkpoints/best.pth"


def main() -> None:
    st.set_page_config(page_title="Real-Time Topology Generator", layout="wide")
    st.title("Real-Time Topology Generator")
    st.caption(
        "Adjust the design condition and the neural implicit field predicts the "
        "density field directly, without rerunning SIMP."
    )

    checkpoint_path = st.sidebar.text_input("Checkpoint", DEFAULT_CHECKPOINT)
    threshold_view = st.sidebar.checkbox("Show binarized density", value=False)
    threshold = st.sidebar.slider("Binarization threshold", 0.0, 1.0, 0.5, 0.01)

    try:
        model, metadata, device = load_cached_model(checkpoint_path)
    except FileNotFoundError:
        st.error(f"Checkpoint not found: `{checkpoint_path}`")
        st.stop()
    except Exception as exc:
        st.error(f"Could not load checkpoint: {exc}")
        st.stop()

    nelx = int(metadata["nelx"])
    nely = int(metadata["nely"])

    controls, visualization = st.columns([1, 2.2], gap="large")
    with controls:
        st.subheader("Condition")
        volfrac = st.slider("Volume Fraction", 0.2, 0.6, 0.4, 0.01)
        load_x = st.slider("Load X Position", nelx // 2, nelx - 1, nelx - 1, 1)
        load_y = st.slider("Load Y Position", 0, nely - 1, nely // 2, 1)
        load_fx = st.slider("Force X Magnitude", -1.0, 1.0, 0.0, 0.01)
        load_fy = st.slider("Force Y Magnitude", -1.0, 1.0, -1.0, 0.01)

    condition = make_condition(volfrac, load_x, load_y, load_fx, load_fy, nelx, nely)

    start = time.perf_counter()
    density = generate_topology(model, condition, nelx, nely, device)
    inference_ms = (time.perf_counter() - start) * 1_000.0

    display_density = (density > threshold).astype(np.float32) if threshold_view else density

    with visualization:
        st.subheader("Predicted Density")
        fig = make_density_figure(
            display_density,
            load_x=load_x,
            load_y=load_y,
            load_fx=load_fx,
            load_fy=load_fy,
        )
        st.pyplot(fig, clear_figure=True)

        metric_cols = st.columns(3)
        metric_cols[0].metric("Inference", f"{inference_ms:.2f} ms")
        metric_cols[1].metric("Target Volume", f"{volfrac:.3f}")
        metric_cols[2].metric("Predicted Volume", f"{volume_fraction(density):.3f}")

        st.code(
            "condition = "
            f"[{condition[0]:.3f}, {condition[1]:.3f}, {condition[2]:.3f}, "
            f"{condition[3]:.3f}, {condition[4]:.3f}]",
            language="python",
        )


@st.cache_resource(show_spinner="Loading model...")
def load_cached_model(
    checkpoint_path: str,
) -> tuple[ConditionalImplicitNetwork, dict[str, object], torch.device]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(path)

    device = get_device()
    model, metadata = load_model(path, device)
    model.eval()
    return model, metadata, device


def make_condition(
    volfrac: float,
    load_x: int,
    load_y: int,
    load_fx: float,
    load_fy: float,
    nelx: int,
    nely: int,
) -> np.ndarray:
    return np.array(
        [
            volfrac,
            load_x / max(1, nelx - 1),
            load_y / max(1, nely - 1),
            load_fx,
            load_fy,
        ],
        dtype=np.float32,
    )


@torch.no_grad()
def generate_topology(
    model: ConditionalImplicitNetwork,
    condition: np.ndarray,
    nelx: int,
    nely: int,
    device: torch.device,
) -> np.ndarray:
    coords = get_coords(nelx, nely, str(device)).to(device)
    cond = torch.as_tensor(condition, dtype=torch.float32, device=device).unsqueeze(0)
    density = model(coords, cond).squeeze(0).squeeze(-1)
    return density.reshape(nelx, nely).detach().cpu().numpy()


@st.cache_resource
def get_coords(nelx: int, nely: int, device_name: str) -> torch.Tensor:
    device = torch.device(device_name)
    return _make_normalized_coords(nelx, nely).to(device).unsqueeze(0)


def make_density_figure(
    density: np.ndarray,
    *,
    load_x: int,
    load_y: int,
    load_fx: float,
    load_fy: float,
) -> plt.Figure:
    nelx, nely = density.shape
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(
        density.T,
        cmap="gray_r",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    force = np.array([load_fx, load_fy], dtype=float)
    force_norm = float(np.linalg.norm(force))
    if force_norm > 1e-12:
        arrow_length = max(1.0, 0.2 * min(nelx, nely))
        arrow = force / force_norm * arrow_length
        ax.annotate(
            "",
            xy=(load_x + arrow[0], load_y + arrow[1]),
            xytext=(load_x, load_y),
            arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 2.5},
        )
    ax.scatter([load_x], [load_y], s=36, c="tab:red")

    ax.set_xlim(-0.5, nelx - 0.5)
    ax.set_ylim(-0.5, nely - 0.5)
    ax.set_xlabel("x element")
    ax.set_ylabel("y element")
    ax.set_title("Neural Implicit Density Prediction")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()

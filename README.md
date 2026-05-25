# Amortized 2D Topology Optimization via Conditional Neural Implicit Fields

Python
PyTorch
Status

This repository implements an amortized surrogate for 2D density-based topology optimization. Instead of solving a new SIMP problem from scratch for each load case, the trained neural implicit field maps a design condition and spatial coordinate directly to material density, producing a full topology in milliseconds.

> **Note:** This project is currently an unpublished AA222 final project. The repository is written in a paper-style format for reproducibility, but there is not yet a formal publication or BibTeX citation.

---

## Abstract

Classical topology optimization methods such as SIMP solve a new finite element optimization problem for every volume fraction, load location, and load direction. This instance-wise workflow is accurate but too slow for interactive design exploration. This project studies an amortized alternative: generate an offline dataset of SIMP-optimized cantilever beams, then train a conditional neural implicit field to represent the map from physical design conditions to optimized density fields.

The model is a PyTorch MLP that takes normalized coordinates `(x, y)` and a five-dimensional condition vector `(volume fraction, load x, load y, Fx, Fy)`. Positional or Gaussian Fourier features encode coordinates, and FiLM layers inject the condition vector into every hidden layer. Once trained, the network can generate continuous density fields, interpolate between load cases, and evaluate at higher spatial resolution than the training grid.

On a 20-case held-out evaluation using `checkpoints/best.pth`, the model produced topologies with a mean inference time of **3.50 ms**, compared with **1.62 s** for SIMP, corresponding to an average **845x speedup**. The predicted volume fraction closely matched the target volume fraction (`0.3886` vs. `0.3900` mean), with a mean compliance error of **32.14%** relative to SIMP.

---

## Overview

The workflow has three stages:

1. Generate a supervised dataset by running a refactored SIMP solver over random cantilever loading conditions.
2. Train a conditional neural implicit field on random coordinate-density samples from each optimized topology.
3. Evaluate the trained model against fresh SIMP solves and generate paper-ready figures.

The learned field represents density as a continuous function:

```text
rho = f_theta(x, y | volume fraction, load_x, load_y, Fx, Fy)
```

This formulation has two useful consequences. First, inference is a single neural network forward pass over the design grid, so it is much faster than iterative FEA-based optimization. Second, because the network is coordinate based, the same trained model can be queried at denser coordinate grids for smooth super-resolution visualizations.

**Condition vector:**


| Entry     | Description                     | Range / Meaning           |
| --------- | ------------------------------- | ------------------------- |
| `volfrac` | Target material volume fraction | `0.2` to `0.6` by default |
| `load_x`  | Normalized load x-location      | right half of beam domain |
| `load_y`  | Normalized load y-location      | full beam height          |
| `Fx`      | Horizontal load component       | sampled from `[-1, 1]`    |
| `Fy`      | Vertical load component         | sampled from `[-1, 1]`    |


The design domain defaults to a `60 x 30` element cantilever beam with the left wall fixed.

---

## Results

The latest evaluation run loaded `checkpoints/best.pth` from epoch 569 and compared neural predictions against fresh SIMP solves on 20 unseen cases.


| Metric                    | Mean        | Std. Dev.   |
| ------------------------- | ----------- | ----------- |
| Neural inference time     | `3.5035 ms` | `1.9715 ms` |
| SIMP runtime              | `1.6170 s`  | `0.9781 s`  |
| Speedup                   | `844.5673x` | `984.4837x` |
| Target volume fraction    | `0.3900`    | `0.0881`    |
| Predicted volume fraction | `0.3886`    | `0.0855`    |
| Compliance error          | `32.1419%`  | `14.4691%`  |


Generated evaluation comparisons are saved in `outputs/eval/`, and paper-style figures are saved in `figures/`:


| Figure                         | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| `figures/fig_qualitative.pdf`  | SIMP references vs. neural predictions            |
| `figures/fig_speed_pareto.pdf` | Inference speed and compliance comparison         |
| `figures/fig_morphing.pdf`     | Continuous interpolation across design conditions |
| `figures/fig_super_res.pdf`    | Higher-resolution queries of the implicit field   |
| `figures/fig_histogram.pdf`    | Distribution of density/compliance errors         |


---

## Repository Structure

```text
.
├── app.py                    # Streamlit app for interactive topology generation
├── dataset.py                # PyTorch dataset and coordinate encoding utilities
├── eval.py                   # Evaluation against fresh SIMP baselines
├── generate_data.py          # Offline SIMP dataset generation
├── generate_figures.py       # Publication-style figure generation
├── model.py                  # Conditional neural implicit field with FiLM
├── retrain.py                # End-to-end dataset generation + training wrapper
├── train.py                  # Training loop, checkpointing, validation figures
├── visualize_dataset.py      # Utility for inspecting generated SIMP datasets
│
├── SIMP/
│   ├── simp_solver.py        # Headless SIMP solver and compliance evaluator
│   └── topopt_cholmod.py     # Original/reference topology optimization script
│
├── data/
│   └── dataset.npz           # SIMP-generated supervised dataset
│
├── checkpoints/
│   └── best.pth              # Best trained model checkpoint
│
├── outputs/
│   ├── eval/                 # Evaluation comparison PNGs
│   └── validation/           # Training-time validation snapshots
│
└── figures/                  # Final PDF figures
```

---

## Method

### SIMP Dataset Generation

`generate_data.py` samples random volume fractions, load locations, and load vectors. For each condition, it runs the refactored SIMP solver in `SIMP/simp_solver.py` and stores:

- `conditions`: shape `(N, 5)`, using the schema `volfrac, normalized_load_x, normalized_load_y, Fx, Fy`
- `topologies`: optimized density fields with shape `(N, nelx, nely)`
- `compliances`: final SIMP compliance values
- `load_dofs` and `load_values`: finite element load metadata

Dataset generation supports multiprocessing and resumes safely from an existing `.npz` file.

### Conditional Neural Implicit Field

`model.py` defines `ConditionalImplicitNetwork`, an MLP that predicts density in `[0, 1]` for arbitrary 2D coordinates. The default trained configuration is:

```text
cond_dim=5
hidden_dim=256
num_hidden_layers=6
num_frequencies=5
fourier_sigma=1.0
activation=silu
encoding_type=positional
```

Coordinates are encoded with NeRF-style positional encoding by default. Gaussian Fourier features are also supported. Each hidden layer is modulated by a FiLM layer, allowing the same coordinate network to adapt to the requested load and volume condition.

### Training Objective

`train.py` trains on random coordinate samples from each topology rather than full dense grids every step. This keeps each batch compact while still covering the full design domain over time. The default loss is MSE between predicted and SIMP density values, with optional BCE support.

During training, the script saves:

- `checkpoints/best.pth` whenever validation loss improves
- periodic epoch checkpoints such as `checkpoints/epoch_0500.pth`
- validation image comparisons in `outputs/validation/`

### Evaluation

`eval.py` samples new unseen load cases, generates a neural topology, solves the corresponding SIMP problem, and reports inference time, SIMP runtime, speedup, predicted volume fraction, SIMP compliance, neural compliance, and compliance error. Comparison plots are written to `outputs/eval/`.

---

## Getting Started

### Dependencies

This project uses NumPy, SciPy, Matplotlib, PyTorch, and Streamlit. A minimal install is:

```bash
pip install numpy scipy matplotlib torch streamlit
```

CUDA is optional but recommended for training and fast evaluation. The scripts automatically choose CUDA, Apple MPS, or CPU through `train.get_device()`.

### Generate the Dataset

```bash
python generate_data.py --output data/dataset.npz --num-samples 3000 --nelx 60 --nely 30
```

Useful options:

```bash
python generate_data.py \
  --num-samples 3000 \
  --num-workers 8 \
  --min-volfrac 0.2 \
  --max-volfrac 0.6 \
  --max-iter 200 \
  --change-tol 0.01
```

### Train the Model

```bash
python train.py --dataset data/dataset.npz --epochs 600 --batch-size 16 --points-per-sample 1024 --amp
```

To resume from a checkpoint:

```bash
python train.py --resume checkpoints/best.pth --additional-epochs 100 --amp
```

To run data generation and training from one command:

```bash
python retrain.py --num-samples 3000 --epochs 600 --amp
```

### Evaluate Against SIMP

```bash
python eval.py --checkpoint checkpoints/best.pth --num-cases 20
```

This writes comparison plots to `outputs/eval/`.

### Generate Figures

```bash
python generate_figures.py --checkpoint checkpoints/best.pth --dataset data/dataset.npz
```

The generated PDFs are written to `figures/`.

### Launch the Interactive App

```bash
streamlit run app.py
```

The app loads `checkpoints/best.pth` by default and exposes sliders for volume fraction, load location, and load vector. It displays the predicted density field, inference time, target volume, and predicted volume.

---

## Output Files


| Path                                 | Description                                 |
| ------------------------------------ | ------------------------------------------- |
| `data/dataset.npz`                   | Supervised SIMP dataset used for training   |
| `checkpoints/best.pth`               | Best trained model checkpoint               |
| `checkpoints/epoch_*.pth`            | Periodic training checkpoints               |
| `outputs/validation/epoch_*.png`     | Training-time SIMP vs. prediction snapshots |
| `outputs/eval/eval_comparisons*.png` | Evaluation comparisons on unseen cases      |
| `figures/*.pdf`                      | Final paper-style figures                   |


---

## Limitations

The surrogate is much faster than SIMP, but it does not replace physics-based optimization when high-accuracy compliance is required. The current model preserves volume fraction well and captures the qualitative load-dependent topology structure, but compliance error remains non-negligible on some cases. The model should therefore be interpreted as an amortized design-space explorer or warm-start generator rather than a certified optimizer.

The current dataset and model are also specific to the chosen 2D cantilever setup, SIMP settings, load parameterization, and `60 x 30` training grid. Extending to different boundary conditions or domains would require regenerating data and retraining.

---

## Acknowledgments

The SIMP implementation is based on the classic educational topology optimization codes by Andreassen et al. and Aage Python variants, refactored here into a headless solver for dataset generation and evaluation. This project was developed as an AA222 final project on data-driven optimization and surrogate modeling.
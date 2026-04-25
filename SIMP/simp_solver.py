"""Headless SIMP topology optimization solver.

This module refactors the classic 2D minimum-compliance topology optimization
script into a reusable function for offline dataset generation.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def solve_simp(
    nelx: int,
    nely: int,
    volfrac: float,
    penal: float,
    rmin: float,
    ft: int,
    load_dof: int,
    load_val: float,
    *,
    max_iter: int = 200,
    change_tol: float = 0.01,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """Solve a 2D SIMP minimum-compliance problem.

    The design domain is a rectangular grid of ``nelx`` by ``nely`` elements.
    Nodes on the left wall are fixed in both displacement directions; the load
    is applied at ``load_dof`` with magnitude ``load_val``.

    Returns:
        A tuple ``(x_phys, compliance)`` where ``x_phys`` has shape
        ``(nelx, nely)`` and contains the final physical density field.
    """

    if ft not in (0, 1):
        raise ValueError("ft must be 0 (sensitivity filter) or 1 (density filter)")

    ndof = 2 * (nelx + 1) * (nely + 1)
    if load_dof < 0 or load_dof >= ndof:
        raise ValueError(f"load_dof must be in [0, {ndof})")

    emin = 1e-9
    emax = 1.0

    x = volfrac * np.ones(nely * nelx, dtype=float)
    xold = x.copy()
    x_phys = x.copy()
    g = 0.0

    ke = _element_stiffness()
    edof_mat = _build_element_dofs(nelx, nely)
    iK = np.kron(edof_mat, np.ones((8, 1))).ravel()
    jK = np.kron(edof_mat, np.ones((1, 8))).ravel()

    h, hs = _build_filter(nelx, nely, rmin)

    dofs = np.arange(ndof)
    left_nodes = np.arange(nely + 1)
    fixed = np.union1d(2 * left_nodes, 2 * left_nodes + 1)
    if np.isin(load_dof, fixed):
        raise ValueError("load_dof cannot be on the fixed left wall")
    free = np.setdiff1d(dofs, fixed)

    force = np.zeros(ndof, dtype=float)
    force[load_dof] = load_val
    displacement = np.zeros(ndof, dtype=float)

    dv = np.ones(nely * nelx, dtype=float)
    dc = np.ones(nely * nelx, dtype=float)
    ce = np.ones(nely * nelx, dtype=float)

    compliance = np.inf
    change = np.inf
    loop = 0

    while change > change_tol and loop < max_iter:
        loop += 1

        stiffness_values = (
            ke.ravel()[:, np.newaxis] * (emin + x_phys**penal * (emax - emin))
        ).ravel(order="F")
        stiffness = coo_matrix((stiffness_values, (iK, jK)), shape=(ndof, ndof)).tocsc()

        displacement[:] = 0.0
        displacement[free] = spsolve(stiffness[free, :][:, free], force[free])

        ue = displacement[edof_mat].reshape(nelx * nely, 8)
        ce[:] = np.einsum("ij,jk,ik->i", ue, ke, ue)
        compliance = float(((emin + x_phys**penal * (emax - emin)) * ce).sum())
        dc[:] = -penal * x_phys ** (penal - 1) * (emax - emin) * ce
        dv[:] = 1.0

        if ft == 0:
            dc[:] = np.asarray(h * (x * dc)[:, np.newaxis] / hs[:, np.newaxis]).ravel()
            dc[:] /= np.maximum(0.001, x)
        else:
            dc[:] = np.asarray(h * (dc[:, np.newaxis] / hs[:, np.newaxis])).ravel()
            dv[:] = np.asarray(h * (dv[:, np.newaxis] / hs[:, np.newaxis])).ravel()

        xold[:] = x
        x[:], g = _optimality_criteria(nelx, nely, x, volfrac, dc, dv, g)

        if ft == 0:
            x_phys[:] = x
        else:
            x_phys[:] = np.asarray(h * x[:, np.newaxis] / hs[:, np.newaxis]).ravel()

        change = float(np.linalg.norm(x - xold, ord=np.inf))

        if verbose:
            current_volume = (g + volfrac * nelx * nely) / (nelx * nely)
            print(
                f"it.: {loop} , obj.: {compliance:.3f} "
                f"Vol.: {current_volume:.3f}, ch.: {change:.3f}"
            )

    return x_phys.reshape((nelx, nely)).astype(np.float32), compliance


def compute_compliance(
    density: np.ndarray,
    load_dof: int,
    load_val: float,
    *,
    penal: float = 3.0,
) -> float:
    """Evaluate true FEA compliance for a fixed density field."""

    if density.ndim != 2:
        raise ValueError(f"density must have shape (nelx, nely), got {density.shape}")

    nelx, nely = density.shape
    density_flat = np.asarray(density, dtype=float).reshape(nelx * nely)
    ndof = 2 * (nelx + 1) * (nely + 1)
    if load_dof < 0 or load_dof >= ndof:
        raise ValueError(f"load_dof must be in [0, {ndof})")

    emin = 1e-9
    emax = 1.0
    ke = _element_stiffness()
    edof_mat = _build_element_dofs(nelx, nely)
    iK = np.kron(edof_mat, np.ones((8, 1))).ravel()
    jK = np.kron(edof_mat, np.ones((1, 8))).ravel()

    dofs = np.arange(ndof)
    left_nodes = np.arange(nely + 1)
    fixed = np.union1d(2 * left_nodes, 2 * left_nodes + 1)
    if np.isin(load_dof, fixed):
        raise ValueError("load_dof cannot be on the fixed left wall")
    free = np.setdiff1d(dofs, fixed)

    force = np.zeros(ndof, dtype=float)
    force[load_dof] = load_val
    displacement = np.zeros(ndof, dtype=float)

    stiffness_values = (
        ke.ravel()[:, np.newaxis] * (emin + density_flat**penal * (emax - emin))
    ).ravel(order="F")
    stiffness = coo_matrix((stiffness_values, (iK, jK)), shape=(ndof, ndof)).tocsc()
    displacement[free] = spsolve(stiffness[free, :][:, free], force[free])

    ue = displacement[edof_mat].reshape(nelx * nely, 8)
    ce = np.einsum("ij,jk,ik->i", ue, ke, ue)
    return float(((emin + density_flat**penal * (emax - emin)) * ce).sum())


def _build_element_dofs(nelx: int, nely: int) -> np.ndarray:
    elx_idx = np.repeat(np.arange(nelx), nely)
    ely_idx = np.tile(np.arange(nely), nelx)
    n1 = (nely + 1) * elx_idx + ely_idx
    n2 = (nely + 1) * (elx_idx + 1) + ely_idx
    return np.stack(
        [
            2 * n1 + 2,
            2 * n1 + 3,
            2 * n2 + 2,
            2 * n2 + 3,
            2 * n2,
            2 * n2 + 1,
            2 * n1,
            2 * n1 + 1,
        ],
        axis=1,
    )


def _build_filter(nelx: int, nely: int, rmin: float) -> tuple[coo_matrix, np.ndarray]:
    ceil_rmin = int(np.ceil(rmin))
    dk = np.arange(-(ceil_rmin - 1), ceil_rmin)
    dl = np.arange(-(ceil_rmin - 1), ceil_rmin)
    dk_grid, dl_grid = np.meshgrid(dk, dl, indexing="ij")
    weights = rmin - np.sqrt(dk_grid**2.0 + dl_grid**2.0)

    ix, jy = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    ix4 = ix[:, :, np.newaxis, np.newaxis]
    jy4 = jy[:, :, np.newaxis, np.newaxis]
    k_abs = ix4 + dk_grid[np.newaxis, np.newaxis, :, :]
    l_abs = jy4 + dl_grid[np.newaxis, np.newaxis, :, :]
    weights4 = np.broadcast_to(weights[np.newaxis, np.newaxis, :, :], k_abs.shape)

    mask = (
        (weights4 > 0.0)
        & (k_abs >= 0)
        & (k_abs < nelx)
        & (l_abs >= 0)
        & (l_abs < nely)
    )
    iH = np.broadcast_to(ix4 * nely + jy4, k_abs.shape)[mask]
    jH = (k_abs * nely + l_abs)[mask]
    sH = weights4[mask]

    h = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    hs = np.asarray(h.sum(axis=1)).ravel()
    return h, hs


def _optimality_criteria(
    nelx: int,
    nely: int,
    x: np.ndarray,
    volfrac: float,
    dc: np.ndarray,
    dv: np.ndarray,
    g: float,
) -> tuple[np.ndarray, float]:
    l1 = 0.0
    l2 = 1e9
    move = 0.2
    xnew = np.zeros(nelx * nely, dtype=float)
    gt = g

    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        update = x * np.sqrt(np.maximum(0.0, -dc / dv / lmid))
        xnew[:] = np.maximum(
            0.0,
            np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, update))),
        )
        gt = g + np.sum(dv * (xnew - x))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid

    return xnew, float(gt)


def _element_stiffness() -> np.ndarray:
    youngs_modulus = 1.0
    poisson_ratio = 0.3
    k = np.array(
        [
            1 / 2 - poisson_ratio / 6,
            1 / 8 + poisson_ratio / 8,
            -1 / 4 - poisson_ratio / 12,
            -1 / 8 + 3 * poisson_ratio / 8,
            -1 / 4 + poisson_ratio / 12,
            -1 / 8 - poisson_ratio / 8,
            poisson_ratio / 6,
            1 / 8 - 3 * poisson_ratio / 8,
        ]
    )
    return youngs_modulus / (1 - poisson_ratio**2) * np.array(
        [
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        ]
    )

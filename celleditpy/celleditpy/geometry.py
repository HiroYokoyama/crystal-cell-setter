#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometry.py  –  Pure crystallographic / geometric computation functions.

No Qt, no pyvista.  All functions are stateless and operate on numpy arrays
or ASE Atoms objects.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from ase.data import vdw_radii


# ---------------------------------------------------------------------------
# Periodic-boundary helpers
# ---------------------------------------------------------------------------

def min_image_cart_offset(cell: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Return the minimum-image Cartesian offset from *p1* to *p2*.

    Parameters
    ----------
    cell : (3, 3) array   – unit-cell matrix (rows are cell vectors)
    p1, p2 : (3,) arrays  – Cartesian positions
    """
    inv_cell = np.linalg.inv(cell)
    delta_s = (p2 - p1).dot(inv_cell)
    delta_s -= np.round(delta_s)
    return delta_s.dot(cell)


# ---------------------------------------------------------------------------
# Van der Waals radii
# ---------------------------------------------------------------------------

def get_vdw_radii_array(atoms) -> np.ndarray:
    """Return per-atom VdW radius (Å).  Defaults to 1.5 Å for unknown atoms."""
    vdw_max = len(vdw_radii)
    nums = atoms.get_atomic_numbers()
    return np.array([
        vdw_radii[n] if (n < vdw_max and vdw_radii[n] > 0) else 1.5
        for n in nums
    ])


# ---------------------------------------------------------------------------
# Auto-fit cell size
# ---------------------------------------------------------------------------

def compute_autofit_cell_params(atoms) -> dict:
    """Compute the minimum cell sizes that contain the molecule.

    The current cell angles are preserved; only the lengths (a, b, c) change.

    Returns
    -------
    dict  –  keys: 'a', 'b', 'c', 'alpha', 'beta', 'gamma'

    Raises
    ------
    ValueError  –  if a non-finite cell size is computed.
    """
    cell = atoms.get_cell()
    params = atoms.cell.cellpar()
    positions = atoms.get_positions()
    radii = get_vdw_radii_array(atoms)

    if np.all(params[:3] == 0) or not np.any(params):
        # No cell set – build orthogonal cell aligned to XYZ axes
        directions = [np.eye(3)[i] for i in range(3)]
        angles = (90.0, 90.0, 90.0)
    else:
        directions = [cell[i] / np.linalg.norm(cell[i]) for i in range(3)]
        angles = (params[3], params[4], params[5])

    new_sizes = []
    for direction in directions:
        coords = positions.dot(direction)
        min_r = radii[np.argmin(coords)]
        max_r = radii[np.argmax(coords)]
        size = float(coords.max() - coords.min() + min_r + max_r)
        if not np.isfinite(size):
            raise ValueError("Non-finite cell size computed during auto-fit.")
        new_sizes.append(float(np.clip(size, 1.0, 1000.0)))

    return {
        'a': new_sizes[0], 'b': new_sizes[1], 'c': new_sizes[2],
        'alpha': angles[0], 'beta': angles[1], 'gamma': angles[2],
    }


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rotation_angle_search(
    positions: np.ndarray,
    rotation_axis: np.ndarray,
    rotation_center: np.ndarray,
    cell: np.ndarray,
    check_axes: list,
    step_deg: int = 5,
) -> float:
    """Find the rotation angle (radians) that minimises cell overflow.

    Tries angles from 0° to 360° in *step_deg* increments and returns the
    angle that gives the smallest total overflow outside the cell.

    Parameters
    ----------
    positions       : (N, 3) array of atomic positions
    rotation_axis   : (3,) unit vector – rotation axis
    rotation_center : (3,) Cartesian rotation origin
    cell            : (3, 3) cell matrix
    check_axes      : list of axis indices (0/1/2) to evaluate overflow on
    step_deg        : angular resolution of the search (degrees)

    Returns
    -------
    float  –  best angle in radians (0 if no improvement found)
    """
    best_angle = 0.0
    min_overflow = float('inf')

    for deg in range(0, 360, step_deg):
        angle = np.radians(deg)
        pts = positions.copy() - rotation_center
        pts = Rotation.from_rotvec(angle * rotation_axis).apply(pts)
        pts += rotation_center

        overflow = 0.0
        for idx in check_axes:
            vec = cell[idx]
            size = np.linalg.norm(vec)
            if size < 1e-10:
                continue
            coords = pts.dot(vec / size)
            overflow += max(0.0, -coords.min()) + max(0.0, coords.max() - size)

        if overflow < min_overflow:
            min_overflow = overflow
            best_angle = angle

    return best_angle


def compute_principal_axis(positions: np.ndarray) -> np.ndarray:
    """Return the principal axis of a set of positions (PCA, largest variance).

    Parameters
    ----------
    positions : (N, 3) array

    Returns
    -------
    (3,) unit vector
    """
    centered = positions - positions.mean(axis=0)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)   # eigh is correct for symmetric matrices
    return vecs[:, np.argmax(vals)]


# ---------------------------------------------------------------------------
# MOL-file bond reader
# ---------------------------------------------------------------------------

def read_mol_bonds(filepath: str) -> tuple:
    """Parse bond connectivity and bond orders from a V2000 or V3000 MDL MOL file.

    Returns
    -------
    pairs  : list of (int, int)  – zero-based atom-index pairs
    orders : list of int         – bond order for each pair (1=single, 2=double, 3=triple,
                                   4=aromatic treated as 1.5 → stored as 4)

    Both lists have the same length.  Returns ([], []) if the file cannot be parsed.
    """
    pairs, orders = [], []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
            lines = fh.readlines()

        # Detect V3000 vs V2000
        v3000 = any('V3000' in line for line in lines[:10])

        if v3000:
            in_bond = False
            for line in lines:
                stripped = line.strip()
                if 'BEGIN BOND' in stripped:
                    in_bond = True
                    continue
                if 'END BOND' in stripped:
                    break
                if in_bond and stripped.startswith('M  V30'):
                    parts = stripped.split()
                    # M  V30  <bond_idx>  <bond_type>  <atom1>  <atom2>
                    if len(parts) >= 6:
                        try:
                            bond_type = int(parts[3])
                            a1 = int(parts[4]) - 1
                            a2 = int(parts[5]) - 1
                            pairs.append((a1, a2))
                            orders.append(bond_type)
                        except ValueError:
                            pass
        else:
            if len(lines) < 4:
                return [], []
            counts = lines[3]
            try:
                n_atoms = int(counts[0:3])
                n_bonds = int(counts[3:6])
            except ValueError:
                return [], []
            bond_start = 4 + n_atoms
            for i in range(n_bonds):
                line = lines[bond_start + i] if bond_start + i < len(lines) else ''
                try:
                    a1 = int(line[0:3]) - 1
                    a2 = int(line[3:6]) - 1
                    bond_type = int(line[6:9])
                    pairs.append((a1, a2))
                    orders.append(bond_type)
                except (ValueError, IndexError):
                    pass
    except Exception as exc:
        print(f"[geometry] MOL bond parse failed: {exc}")
    return pairs, orders


def apply_rotation_to_atoms(atoms, angle_rad: float, axis: np.ndarray, center: np.ndarray):
    """Rotate *atoms* in-place around *axis* by *angle_rad* about *center*."""
    if abs(angle_rad) < 1e-9:
        return
    rot = Rotation.from_rotvec(angle_rad * axis)
    atoms.positions -= center
    atoms.positions = rot.apply(atoms.positions)
    atoms.positions += center

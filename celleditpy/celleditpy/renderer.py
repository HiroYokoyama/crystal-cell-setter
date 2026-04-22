#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
renderer.py  –  PyVista scene-drawing helpers for CelleditPy.

All functions accept a PyVista plotter as their first argument and are
otherwise stateless so they can be unit-tested or reused independently.
"""

import numpy as np
import pyvista as pv
from ase.data import vdw_radii, atomic_numbers
from ase.neighborlist import NeighborList, natural_cutoffs

from .constants import CPK_COLORS

# ---------------------------------------------------------------------------
# Shared mesh properties
# ---------------------------------------------------------------------------

MESH_PROPS = dict(smooth_shading=True, specular=0.2, specular_power=20, lighting=True)


def _cpk_rgb(symbol: str) -> tuple:
    """Return (r, g, b) floats for a chemical symbol using CPK colours."""
    c = CPK_COLORS.get(symbol, CPK_COLORS['DEFAULT'])
    return (c.redF(), c.greenF(), c.blueF())


# ---------------------------------------------------------------------------
# Individual drawing functions
# ---------------------------------------------------------------------------

def draw_origin_label(plotter) -> None:
    """Add a white 'O' label at the crystallographic origin."""
    try:
        plotter.add_point_labels(
            [np.array([0.0, 0.0, 0.0])], ["O"],
            point_size=0, font_size=16, text_color='white',
            always_visible=True, shape=None, show_points=False,
        )
    except Exception as exc:
        print(f"[renderer] origin label failed: {exc}")


def draw_atoms(plotter, atoms_to_draw) -> None:
    """Render atoms as CPK-coloured spheres scaled by 30 % of VdW radius."""
    symbols = atoms_to_draw.get_chemical_symbols()
    positions = atoms_to_draw.get_positions()
    if len(positions) == 0:
        return

    vdw_max = len(vdw_radii)
    colors, radii = [], []
    for sym in symbols:
        colors.append(_cpk_rgb(sym))
        num = atomic_numbers.get(sym, 0)
        r = vdw_radii[num] * 0.3 if (num < vdw_max and vdw_radii[num] > 0) else 0.3
        radii.append(r)

    src = pv.PolyData(positions)
    src['colors'] = np.array(colors)
    src['radii'] = np.array(radii)
    geom = pv.Sphere(radius=1.0, theta_resolution=16, phi_resolution=16)
    glyphs = src.glyph(scale='radii', geom=geom, orient=False)
    plotter.add_mesh(glyphs, scalars='colors', rgb=True, **MESH_PROPS)


_BOND_RADIUS   = 0.10   # Å – tube radius for single bonds
_DOUBLE_OFFSET = 0.14   # Å – half-separation between double-bond tubes
_TRIPLE_OFFSET = 0.16   # Å – half-separation between outer triple-bond tubes


def _perpendicular_to(vec: np.ndarray) -> np.ndarray:
    """Return an arbitrary unit vector perpendicular to *vec*."""
    ref = np.array([0.0, 0.0, 1.0])
    perp = np.cross(vec, ref)
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(vec, np.array([0.0, 1.0, 0.0]))
    return perp / np.linalg.norm(perp)


def _add_tube(plotter, p1: np.ndarray, p2: np.ndarray, radius: float = _BOND_RADIUS) -> None:
    mesh = pv.PolyData(np.array([p1, p2]))
    mesh.lines = np.array([2, 0, 1])
    plotter.add_mesh(mesh.tube(radius=radius, n_sides=12), color='grey', **MESH_PROPS)


def draw_bonds(plotter, atoms_to_draw) -> None:
    """Render bonds as grey tubes with proper single / double / triple geometry.

    Bond source priority
    --------------------
    1. ``atoms_to_draw.info['_bond_pairs']`` + ``'_bond_orders'``
       – set when loading a MOL file; encodes exact connectivity and bond order.
    2. ASE NeighborList distance heuristic – fallback for CIF and other formats
       (all bonds drawn as single).
    """
    positions = atoms_to_draw.get_positions()
    if len(positions) < 2:
        return

    try:
        info = getattr(atoms_to_draw, 'info', {}) or {}
        explicit_pairs  = info.get('_bond_pairs')
        explicit_orders = info.get('_bond_orders')
        n = len(positions)

        if explicit_pairs:
            # Build deduplicated (i, j, order) list from stored data
            seen = {}
            for k, (i, j) in enumerate(explicit_pairs):
                if not (0 <= i < n and 0 <= j < n and i != j):
                    continue
                key = (min(i, j), max(i, j))
                order = explicit_orders[k] if explicit_orders and k < len(explicit_orders) else 1
                if key not in seen:
                    seen[key] = order
            bond_list = [(i, j, o) for (i, j), o in seen.items()]
        else:
            # Distance-based fallback – all bonds are order 1.
            # mult=1.15 adds 15 % slack so strained/elongated bonds (e.g.
            # cyclobutadiene C–C single at 1.57 Å) are not missed.
            cutoffs = natural_cutoffs(atoms_to_draw, mult=1.15)
            nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
            nl.update(atoms_to_draw)
            coo = nl.get_connectivity_matrix().tocoo()
            if coo.nnz == 0:
                return
            mask = coo.row < coo.col
            bond_list = [
                (int(r), int(c), 1)
                for r, c in zip(coo.row[mask], coo.col[mask])
            ]

        if not bond_list:
            return

        for i, j, order in bond_list:
            p1, p2 = positions[i], positions[j]
            bond_dir = p2 - p1
            length = np.linalg.norm(bond_dir)
            if length < 1e-6:
                continue
            bond_dir /= length

            if order == 1 or order > 3:
                # Single bond (or aromatic drawn single)
                _add_tube(plotter, p1, p2)

            elif order == 2:
                # Double bond – two parallel tubes offset perpendicular to bond
                perp = _perpendicular_to(bond_dir)
                offset = perp * _DOUBLE_OFFSET
                _add_tube(plotter, p1 + offset, p2 + offset)
                _add_tube(plotter, p1 - offset, p2 - offset)

            elif order == 3:
                # Triple bond – one central tube + two offset tubes
                perp = _perpendicular_to(bond_dir)
                offset = perp * _TRIPLE_OFFSET
                _add_tube(plotter, p1, p2)
                _add_tube(plotter, p1 + offset, p2 + offset, radius=_BOND_RADIUS * 0.8)
                _add_tube(plotter, p1 - offset, p2 - offset, radius=_BOND_RADIUS * 0.8)

    except Exception as exc:
        print(f"[renderer] bond drawing failed: {exc}")


def draw_atom_labels(plotter, positions) -> None:
    """Overlay blue atom index labels at each atomic position."""
    for i, pos in enumerate(positions):
        plotter.add_point_labels(
            [pos], [str(i)],
            point_size=0, font_size=12, text_color='#0066CC',
            always_visible=True, shape=None,
        )


def draw_cell(plotter, atoms) -> None:
    """Draw the unit cell as coloured axes (a=red, b=green, c=blue) plus white edges."""
    if not atoms.pbc.any():
        return
    cell = atoms.get_cell()
    o = np.zeros(3)
    c0, c1, c2 = cell[0], cell[1], cell[2]
    corners = [
        o, c0, c1, c2,
        c0 + c1, c0 + c2, c1 + c2, c0 + c1 + c2,
    ]

    # Primary axes with colour + label
    for end_idx, color, label in [(1, 'red', 'a'), (2, 'green', 'b'), (3, 'blue', 'c')]:
        plotter.add_lines(np.array([corners[0], corners[end_idx]]), color=color, width=5)
        plotter.add_point_labels(
            [corners[end_idx]], [label],
            point_size=0, font_size=20, text_color=color,
            bold=True, always_visible=True, shape=None,
        )

    # Remaining cell edges
    for s, e in [(1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]:
        plotter.add_lines(np.array([corners[s], corners[e]]), color='white', width=3)


def make_selection_glyphs(atoms, indices: list):
    """Return a PyVista glyph mesh for selection markers (yellow spheres).

    Returns *None* when *indices* is empty.
    """
    if not indices:
        return None
    idx_arr = np.array(indices, dtype=int)
    positions = atoms.positions[idx_arr]
    nums = atoms.get_atomic_numbers()
    vdw_max = len(vdw_radii)
    radii = []
    for i in idx_arr:
        n = nums[i]
        r = vdw_radii[n] * 0.39 if (n < vdw_max and vdw_radii[n] > 0) else 0.4 * 0.39
        radii.append(r)
    src = pv.PolyData(positions)
    src['radii'] = np.array(radii)
    geom = pv.Sphere(radius=1.0, theta_resolution=16, phi_resolution=16)
    return src.glyph(scale='radii', geom=geom, orient=False)

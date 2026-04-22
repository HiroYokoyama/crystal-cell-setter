#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for celleditpy.geometry  (no GUI required).
"""

import sys
import os
import tempfile

# Ensure the local package is imported, not the installed one.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg_dir = os.path.join(_repo_root, "celleditpy")
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.build import molecule  # noqa: E402

from celleditpy.geometry import (  # noqa: E402
    min_image_cart_offset,
    get_vdw_radii_array,
    compute_autofit_cell_params,
    rotation_angle_search,
    compute_principal_axis,
    apply_rotation_to_atoms,
    read_mol_bonds,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cubic_cell():
    """10 Å cubic unit-cell matrix."""
    return np.eye(3) * 10.0


@pytest.fixture
def water():
    """ASE water molecule with a 10 Å cubic cell."""
    mol = molecule('H2O')
    mol.set_cell(np.eye(3) * 10.0)
    mol.set_pbc(True)
    return mol


@pytest.fixture
def linear_chain():
    """5-atom H chain along x, equally spaced."""
    pos = np.array([[i, 0.0, 0.0] for i in range(5)], dtype=float)
    return Atoms('H5', positions=pos, cell=np.eye(3) * 20.0, pbc=True)


# Minimal V2000 MOL file: two carbons with 1 C-C single bond.
# Must have exactly 3 header lines before the counts line (lines[3]).
_V2000_MOL = (
    "ethane\n"
    "  test\n"
    "\n"
    "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
    "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "    1.5400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  1  2  1  0  0  0  0\n"
    "M  END\n"
)

# Minimal V2000 MOL file with a double bond (C=C)
_V2000_DOUBLE = (
    "ethylene\n"
    "  test\n"
    "\n"
    "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
    "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "    1.3400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
    "  1  2  2  0  0  0  0\n"
    "M  END\n"
)

# Minimal V3000 MOL file: two carbons with a triple bond
_V3000_MOL = (
    "\n"
    "  propyne\n"
    "\n"
    "\n"
    "  0  0  0  0  0  0            999 V3000\n"
    "M  V30 BEGIN CTAB\n"
    "M  V30 COUNTS 2 1 0 0 0\n"
    "M  V30 BEGIN ATOM\n"
    "M  V30 1 C 0.0 0.0 0.0 0\n"
    "M  V30 2 C 1.2 0.0 0.0 0\n"
    "M  V30 END ATOM\n"
    "M  V30 BEGIN BOND\n"
    "M  V30 1 3 1 2\n"
    "M  V30 END BOND\n"
    "M  V30 END CTAB\n"
    "M  END\n"
)


# ---------------------------------------------------------------------------
# min_image_cart_offset
# ---------------------------------------------------------------------------

class TestMinImageCartOffset:
    def test_same_cell(self, cubic_cell):
        p1 = np.array([1.0, 1.0, 1.0])
        p2 = np.array([3.0, 3.0, 3.0])
        offset = min_image_cart_offset(cubic_cell, p1, p2)
        np.testing.assert_allclose(offset, [2.0, 2.0, 2.0], atol=1e-10)

    def test_periodic_shortcut(self, cubic_cell):
        """Atom near x=0 and atom near x=9 should be only 1 Å apart via PBC."""
        p1 = np.array([0.5, 0.0, 0.0])
        p2 = np.array([9.5, 0.0, 0.0])
        offset = min_image_cart_offset(cubic_cell, p1, p2)
        np.testing.assert_allclose(np.linalg.norm(offset), 1.0, atol=1e-10)

    def test_zero_offset(self, cubic_cell):
        p = np.array([5.0, 5.0, 5.0])
        offset = min_image_cart_offset(cubic_cell, p, p)
        np.testing.assert_allclose(offset, np.zeros(3), atol=1e-10)

    def test_non_cubic(self):
        """Triclinic cell: offset should still lie in [-L/2, L/2]."""
        cell = np.array([[5.0, 0.0, 0.0],
                         [2.5, 4.33, 0.0],
                         [0.0, 0.0, 6.0]])
        p1 = np.zeros(3)
        p2 = np.array([4.9, 0.0, 0.0])
        offset = min_image_cart_offset(cell, p1, p2)
        assert np.linalg.norm(offset) < 3.0

    def test_antisymmetry(self, cubic_cell):
        """offset(p1→p2) == -offset(p2→p1) under minimum image."""
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([8.0, 7.0, 6.0])
        fwd = min_image_cart_offset(cubic_cell, p1, p2)
        bwd = min_image_cart_offset(cubic_cell, p2, p1)
        np.testing.assert_allclose(fwd, -bwd, atol=1e-10)

    def test_result_shape(self, cubic_cell):
        p1, p2 = np.zeros(3), np.ones(3)
        offset = min_image_cart_offset(cubic_cell, p1, p2)
        assert offset.shape == (3,)


# ---------------------------------------------------------------------------
# get_vdw_radii_array
# ---------------------------------------------------------------------------

class TestGetVdwRadii:
    def test_water_radii_positive(self, water):
        radii = get_vdw_radii_array(water)
        assert (radii > 0).all()
        assert len(radii) == len(water)

    def test_default_fallback(self):
        """Element number 0 (unknown) should fall back to 1.5 Å."""
        atoms = Atoms('H', positions=[[0, 0, 0]])
        atoms.numbers = np.array([0])   # force unknown element
        radii = get_vdw_radii_array(atoms)
        assert radii[0] == pytest.approx(1.5)

    def test_hydrogen_radius(self):
        h = Atoms('H', positions=[[0, 0, 0]])
        radii = get_vdw_radii_array(h)
        assert radii[0] > 0.5

    def test_length_matches_atoms(self, linear_chain):
        radii = get_vdw_radii_array(linear_chain)
        assert len(radii) == len(linear_chain)

    def test_carbon_radius(self):
        c = Atoms('C', positions=[[0, 0, 0]])
        radii = get_vdw_radii_array(c)
        assert 1.0 < radii[0] < 2.5  # carbon VdW ~1.7 Å


# ---------------------------------------------------------------------------
# compute_autofit_cell_params
# ---------------------------------------------------------------------------

class TestComputeAutofitCellParams:
    def test_returns_expected_keys(self, water):
        params = compute_autofit_cell_params(water)
        assert set(params) == {'a', 'b', 'c', 'alpha', 'beta', 'gamma'}

    def test_sizes_positive(self, water):
        params = compute_autofit_cell_params(water)
        assert params['a'] > 0
        assert params['b'] > 0
        assert params['c'] > 0

    def test_angles_preserved(self, water):
        """If the cell already has orthogonal angles, they must be kept."""
        params = compute_autofit_cell_params(water)
        assert params['alpha'] == pytest.approx(90.0, abs=0.1)
        assert params['beta'] == pytest.approx(90.0, abs=0.1)
        assert params['gamma'] == pytest.approx(90.0, abs=0.1)

    def test_molecule_fits_inside(self, water):
        """Computed cell must be large enough to contain all atoms."""
        from ase.data import vdw_radii as _vdw
        params = compute_autofit_cell_params(water)
        positions = water.get_positions()
        for axis, key in enumerate(['a', 'b', 'c']):
            coords = positions[:, axis]
            nums = water.get_atomic_numbers()
            radii = np.array([_vdw[n] if _vdw[n] > 0 else 1.5 for n in nums])
            span = coords.max() - coords.min() + radii[np.argmax(coords)] + radii[np.argmin(coords)]
            assert params[key] >= span - 1e-6

    def test_no_cell_set(self):
        """Molecule with no cell should still produce a valid result."""
        atoms = molecule('CH4')
        params = compute_autofit_cell_params(atoms)
        assert all(v > 0 for v in [params['a'], params['b'], params['c']])

    def test_single_atom(self):
        """Single atom should produce a cell sized ≥ 2×VdW radius."""
        from ase.data import vdw_radii as _vdw, atomic_numbers
        atoms = Atoms('C', positions=[[0, 0, 0]])
        params = compute_autofit_cell_params(atoms)
        n = atomic_numbers['C']
        r = _vdw[n]
        assert params['a'] >= 2 * r - 1e-6

    def test_sizes_clamped(self):
        """Sizes must be at least 1.0 Å (lower clamp in implementation)."""
        atoms = Atoms('H', positions=[[0, 0, 0]])
        params = compute_autofit_cell_params(atoms)
        assert params['a'] >= 1.0

    def test_non_orthogonal_angles_preserved(self):
        """Non-90° angles in the input cell must come back unchanged."""
        from ase.geometry import cellpar_to_cell
        cellpar = [10.0, 10.0, 10.0, 80.0, 100.0, 120.0]
        cell = cellpar_to_cell(cellpar)
        atoms = molecule('H2O')
        atoms.set_cell(cell)
        atoms.set_pbc(True)
        params = compute_autofit_cell_params(atoms)
        assert params['alpha'] == pytest.approx(80.0, abs=0.01)
        assert params['beta'] == pytest.approx(100.0, abs=0.01)
        assert params['gamma'] == pytest.approx(120.0, abs=0.01)


# ---------------------------------------------------------------------------
# rotation_angle_search
# ---------------------------------------------------------------------------

class TestRotationAngleSearch:
    def test_zero_for_contained_molecule(self):
        """Molecule already inside cell → best angle should give zero overflow."""
        positions = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=float)
        cell = np.eye(3) * 10.0
        axis = np.array([0.0, 0.0, 1.0])
        center = positions.mean(axis=0)
        best = rotation_angle_search(positions, axis, center, cell, [0, 1])
        # No overflow at best angle
        from scipy.spatial.transform import Rotation
        pts = positions.copy() - center
        pts = Rotation.from_rotvec(best * axis).apply(pts)
        pts += center
        for idx in [0, 1]:
            vec = cell[idx]
            size = np.linalg.norm(vec)
            coords = pts.dot(vec / size)
            assert coords.min() >= -1e-6
            assert coords.max() <= size + 1e-6

    def test_returns_float(self):
        positions = np.random.default_rng(42).random((5, 3)) * 5.0
        cell = np.eye(3) * 10.0
        axis = np.array([0.0, 1.0, 0.0])
        best = rotation_angle_search(positions, axis, np.zeros(3), cell, [0, 2])
        assert isinstance(best, float)

    def test_step_deg_resolution(self):
        """Coarser step should not diverge wildly from finer step."""
        positions = np.array([[0.0, 5.5, 1.0], [0.0, 4.5, 1.0]])
        cell = np.eye(3) * 5.0
        axis = np.array([1.0, 0.0, 0.0])
        center = positions.mean(axis=0)
        best5 = rotation_angle_search(positions, axis, center, cell, [1, 2], step_deg=5)
        best10 = rotation_angle_search(positions, axis, center, cell, [1, 2], step_deg=10)
        diff = abs(best5 - best10)
        assert diff < np.radians(15)

    def test_empty_check_axes(self):
        """Empty check_axes: all angles have zero overflow, returns 0 radians."""
        positions = np.array([[1.0, 2.0, 3.0]])
        cell = np.eye(3) * 10.0
        best = rotation_angle_search(positions, np.array([0, 0, 1.0]), np.zeros(3), cell, [])
        assert best == pytest.approx(0.0)

    def test_result_in_valid_range(self):
        """Result must be in [0, 2π)."""
        rng = np.random.default_rng(99)
        positions = rng.random((6, 3)) * 3.0 + 5.0
        cell = np.eye(3) * 12.0
        axis = np.array([0.0, 1.0, 0.0])
        best = rotation_angle_search(positions, axis, positions.mean(axis=0), cell, [0, 1, 2])
        assert 0.0 <= best < 2 * np.pi + 1e-9


# ---------------------------------------------------------------------------
# compute_principal_axis
# ---------------------------------------------------------------------------

class TestComputePrincipalAxis:
    def test_along_x(self):
        """Linear chain along x → principal axis must be x."""
        positions = np.array([[i * 1.0, 0.0, 0.0] for i in range(10)])
        axis = compute_principal_axis(positions)
        assert abs(np.dot(axis, [1, 0, 0])) > 0.99

    def test_unit_length(self):
        pos = np.random.default_rng(0).random((8, 3))
        axis = compute_principal_axis(pos)
        assert np.linalg.norm(axis) == pytest.approx(1.0, abs=1e-6)

    def test_returns_real(self):
        pos = np.random.default_rng(7).random((6, 3))
        axis = compute_principal_axis(pos)
        assert axis.dtype in (np.float64, np.float32)

    def test_along_y(self):
        positions = np.array([[0.0, i * 1.0, 0.0] for i in range(8)])
        axis = compute_principal_axis(positions)
        assert abs(np.dot(axis, [0, 1, 0])) > 0.99

    def test_shape(self):
        pos = np.random.default_rng(1).random((5, 3))
        axis = compute_principal_axis(pos)
        assert axis.shape == (3,)


# ---------------------------------------------------------------------------
# apply_rotation_to_atoms
# ---------------------------------------------------------------------------

class TestApplyRotationToAtoms:
    def test_360_deg_identity(self, water):
        original = water.get_positions().copy()
        axis = np.array([0.0, 0.0, 1.0])
        center = original.mean(axis=0)
        apply_rotation_to_atoms(water, 2 * np.pi, axis, center)
        np.testing.assert_allclose(water.get_positions(), original, atol=1e-10)

    def test_zero_angle_no_change(self, water):
        original = water.get_positions().copy()
        apply_rotation_to_atoms(water, 0.0, np.array([1, 0, 0]), np.zeros(3))
        np.testing.assert_allclose(water.get_positions(), original, atol=1e-10)

    def test_90_deg_x_axis(self):
        """90° around z should map (1,0,0) → (0,1,0) approximately."""
        atoms = Atoms('H', positions=[[1.0, 0.0, 0.0]])
        apply_rotation_to_atoms(atoms, np.pi / 2, np.array([0.0, 0.0, 1.0]), np.zeros(3))
        np.testing.assert_allclose(atoms.positions[0], [0.0, 1.0, 0.0], atol=1e-10)

    def test_center_invariant(self):
        """The rotation center itself must not move."""
        center = np.array([3.0, 4.0, 5.0])
        atoms = Atoms('HH', positions=[center.tolist(), [0.0, 0.0, 0.0]])
        apply_rotation_to_atoms(atoms, np.pi / 3, np.array([1.0, 0.0, 0.0]), center)
        np.testing.assert_allclose(atoms.positions[0], center, atol=1e-10)

    def test_preserves_bond_length(self):
        """Rotation must not change interatomic distance."""
        atoms = Atoms('HH', positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        d_before = np.linalg.norm(atoms.positions[1] - atoms.positions[0])
        apply_rotation_to_atoms(atoms, np.pi / 4, np.array([0, 1, 0]), np.zeros(3))
        d_after = np.linalg.norm(atoms.positions[1] - atoms.positions[0])
        assert d_after == pytest.approx(d_before, abs=1e-10)

    def test_180_deg_flip(self):
        """180° around z: (1,0,0) → (-1,0,0)."""
        atoms = Atoms('H', positions=[[1.0, 0.0, 0.0]])
        apply_rotation_to_atoms(atoms, np.pi, np.array([0.0, 0.0, 1.0]), np.zeros(3))
        np.testing.assert_allclose(atoms.positions[0], [-1.0, 0.0, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# read_mol_bonds
# ---------------------------------------------------------------------------

class TestReadMolBonds:
    def _write_mol(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix='.mol')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path

    def test_v2000_single_bond_count(self):
        path = self._write_mol(_V2000_MOL)
        pairs, orders = read_mol_bonds(path)
        os.unlink(path)
        assert len(pairs) == 1
        assert len(orders) == 1

    def test_v2000_single_bond_indices(self):
        path = self._write_mol(_V2000_MOL)
        pairs, orders = read_mol_bonds(path)
        os.unlink(path)
        assert pairs[0] == (0, 1)

    def test_v2000_single_bond_order(self):
        path = self._write_mol(_V2000_MOL)
        pairs, orders = read_mol_bonds(path)
        os.unlink(path)
        assert orders[0] == 1

    def test_v2000_double_bond_order(self):
        path = self._write_mol(_V2000_DOUBLE)
        pairs, orders = read_mol_bonds(path)
        os.unlink(path)
        assert orders[0] == 2

    def test_v3000_triple_bond_order(self):
        path = self._write_mol(_V3000_MOL)
        pairs, orders = read_mol_bonds(path)
        os.unlink(path)
        assert len(pairs) == 1
        assert orders[0] == 3

    def test_v3000_zero_based_indices(self):
        path = self._write_mol(_V3000_MOL)
        pairs, orders = read_mol_bonds(path)
        os.unlink(path)
        assert pairs[0] == (0, 1)  # V3000 uses 1-based → converted to 0-based

    def test_nonexistent_file_returns_empty(self):
        pairs, orders = read_mol_bonds('/nonexistent/path/file.mol')
        assert pairs == []
        assert orders == []

    def test_empty_file_returns_empty(self):
        fd, path = tempfile.mkstemp(suffix='.mol')
        os.close(fd)
        pairs, orders = read_mol_bonds(path)
        os.unlink(path)
        assert pairs == []
        assert orders == []

    def test_pairs_and_orders_same_length(self):
        path = self._write_mol(_V2000_MOL)
        pairs, orders = read_mol_bonds(path)
        os.unlink(path)
        assert len(pairs) == len(orders)

# ---------------------------------------------------------------------------
# detect_bonds helper (mirrors app.detect_bonds logic, no Qt)
# ---------------------------------------------------------------------------


def _detect_bonds(atoms):
    """Pure-Python version of app.detect_bonds for unit testing."""
    from ase.neighborlist import natural_cutoffs, neighbor_list as _nl
    cutoffs = natural_cutoffs(atoms, mult=1.15)
    ii, jj = _nl('ij', atoms, cutoffs)
    seen, pairs, orders = set(), [], []
    for a, b in zip(ii, jj):
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            pairs.append(key)
            orders.append(1)
    return pairs, orders


class TestDetectBonds:
    def test_water_has_two_bonds(self):
        atoms = molecule('H2O')
        atoms.set_cell(np.eye(3) * 10.0)
        atoms.set_pbc(True)
        pairs, orders = _detect_bonds(atoms)
        assert len(pairs) == 2

    def test_all_single_order(self):
        atoms = molecule('CH4')
        atoms.set_cell(np.eye(3) * 10.0)
        atoms.set_pbc(True)
        _, orders = _detect_bonds(atoms)
        assert all(o == 1 for o in orders)

    def test_pairs_unique(self):
        atoms = molecule('H2O')
        atoms.set_cell(np.eye(3) * 10.0)
        atoms.set_pbc(True)
        pairs, _ = _detect_bonds(atoms)
        assert len(pairs) == len(set(pairs))

    def test_pairs_normalised(self):
        """All pairs must be (min, max)."""
        atoms = molecule('CH4')
        atoms.set_cell(np.eye(3) * 10.0)
        atoms.set_pbc(True)
        pairs, _ = _detect_bonds(atoms)
        for a, b in pairs:
            assert a <= b

    def test_methane_has_four_bonds(self):
        atoms = molecule('CH4')
        atoms.set_cell(np.eye(3) * 10.0)
        atoms.set_pbc(True)
        pairs, _ = _detect_bonds(atoms)
        assert len(pairs) == 4


# ---------------------------------------------------------------------------
# complete_molecule helper (mirrors app.complete_molecule logic, no Qt)
# ---------------------------------------------------------------------------

def _complete_molecule(atoms):
    """Pure-Python version of app.complete_molecule for unit testing."""
    cell = atoms.get_cell()
    n = len(atoms)
    from ase.neighborlist import natural_cutoffs, neighbor_list as _nl
    cutoffs = natural_cutoffs(atoms, mult=1.15)
    ii, jj = _nl('ij', atoms, cutoffs)
    adj = {i: [] for i in range(n)}
    for a, b in zip(ii, jj):
        adj[a].append(b)

    positions = atoms.positions.copy()
    visited = np.zeros(n, dtype=bool)
    for start in range(n):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        while queue:
            i = queue.pop(0)
            for j in adj[i]:
                if visited[j]:
                    continue
                diff = positions[j] - positions[i]
                frac = np.linalg.solve(np.array(cell).T, diff)
                frac -= np.round(frac)
                positions[j] = positions[i] + frac @ np.array(cell)
                visited[j] = True
                queue.append(j)
    atoms.positions = positions


class TestCompleteMolecule:
    def _split_water(self):
        """Water molecule with O at origin and H shifted across cell boundary."""
        atoms = molecule('H2O')
        atoms.set_cell(np.eye(3) * 5.0)
        atoms.set_pbc(True)
        # Shift one H atom across the boundary
        atoms.positions[1] += np.array([5.0, 0.0, 0.0])
        return atoms

    def test_bond_length_preserved(self):
        atoms = self._split_water()
        d_before = np.linalg.norm(atoms.positions[0] - atoms.positions[1])
        _complete_molecule(atoms)
        d_after = np.linalg.norm(atoms.positions[0] - atoms.positions[1])
        # After completion, bond length should be the original O-H (~0.96 Å)
        # Before it was artificially ~5 Å; completion brings it back
        assert d_after < d_before

    def test_all_atoms_visited(self):
        atoms = molecule('H2O')
        atoms.set_cell(np.eye(3) * 10.0)
        atoms.set_pbc(True)
        original = atoms.positions.copy()
        _complete_molecule(atoms)
        # When no atoms are split, positions should be unchanged
        np.testing.assert_allclose(atoms.positions, original, atol=1e-10)

    def test_anchor_atom_unchanged(self):
        """First atom (index 0) must never move."""
        atoms = self._split_water()
        pos0_before = atoms.positions[0].copy()
        _complete_molecule(atoms)
        np.testing.assert_allclose(atoms.positions[0], pos0_before, atol=1e-10)

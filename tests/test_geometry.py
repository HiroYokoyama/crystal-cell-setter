#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for celleditpy.geometry  (no GUI required).
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from celleditpy.geometry import (
    min_image_cart_offset,
    get_vdw_radii_array,
    compute_autofit_cell_params,
    rotation_angle_search,
    compute_principal_axis,
    apply_rotation_to_atoms,
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
        # The minimum-image distance must be ≤ half the shortest cell length
        assert np.linalg.norm(offset) < 3.0


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
        assert params['beta']  == pytest.approx(90.0, abs=0.1)
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
        best5  = rotation_angle_search(positions, axis, center, cell, [1, 2], step_deg=5)
        best10 = rotation_angle_search(positions, axis, center, cell, [1, 2], step_deg=10)
        # Both should be within one step of each other
        diff = abs(best5 - best10)
        assert diff < np.radians(15)


# ---------------------------------------------------------------------------
# compute_principal_axis
# ---------------------------------------------------------------------------

class TestComputePrincipalAxis:
    def test_along_x(self):
        """Linear chain along x → principal axis must be x."""
        positions = np.array([[i * 1.0, 0.0, 0.0] for i in range(10)])
        axis = compute_principal_axis(positions)
        # Allow sign flip
        assert abs(np.dot(axis, [1, 0, 0])) > 0.99

    def test_unit_length(self):
        pos = np.random.default_rng(0).random((8, 3))
        axis = compute_principal_axis(pos)
        assert np.linalg.norm(axis) == pytest.approx(1.0, abs=1e-6)

    def test_returns_real(self):
        pos = np.random.default_rng(7).random((6, 3))
        axis = compute_principal_axis(pos)
        assert axis.dtype in (np.float64, np.float32)


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

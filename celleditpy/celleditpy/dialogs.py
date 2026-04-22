#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dialogs.py  –  Modeless dialog classes for CelleditPy.

Each dialog receives lightweight callbacks so it stays decoupled from the
main application window.
"""

import numpy as np
from scipy.spatial.transform import Rotation

import networkx as nx
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.data import vdw_radii
from ase.geometry import cellpar_to_cell

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit,
    QComboBox, QCheckBox, QDoubleSpinBox, QWidget,
    QButtonGroup, QRadioButton, QMessageBox,
)
from PyQt6.QtCore import Qt

from .geometry import (
    rotation_angle_search,
    compute_principal_axis,
    apply_rotation_to_atoms,
    min_image_cart_offset,
    get_vdw_radii_array,
)


# ---------------------------------------------------------------------------
# Delete dialog
# ---------------------------------------------------------------------------

class DeleteDialog(QDialog):
    """Modeless dialog for selecting and deleting atoms."""

    def __init__(self, atoms_ref, redraw_fn, mark_fn,
                 enable_pick_fn, disable_pick_fn, parent=None):
        super().__init__(parent)
        self._atoms = atoms_ref        # mutable list wrapper: [atoms]
        self._redraw = redraw_fn
        self._mark = mark_fn
        self._enable_pick = enable_pick_fn
        self._disable_pick = disable_pick_fn

        self.selected = set()

        self.setWindowTitle("Delete Atoms")
        self.setModal(False)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Click atoms in the 3D view to select them."))

        self._info_label = QLabel("Selected: 0 atoms")
        layout.addWidget(self._info_label)

        btn_layout = QHBoxLayout()
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self._execute)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(delete_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.finished.connect(self._on_closed)
        self._enable_pick()

    # -- picking callback (called by app's on_atom_picked) ------------------

    def picking_callback(self, idx: int):
        if idx in self.selected:
            self.selected.discard(idx)
        else:
            self.selected.add(idx)
        self._refresh_display()

    def _refresh_display(self):
        self._mark(list(self.selected))
        self._info_label.setText(f"Selected: {len(self.selected)} atoms")

    # -- actions ------------------------------------------------------------

    def _execute(self):
        if not self.selected:
            return
        atoms = self._atoms[0]
        indices = sorted(self.selected, reverse=True)
        try:
            mask = np.ones(len(atoms), dtype=bool)
            for i in indices:
                mask[i] = False
            self._atoms[0] = atoms[mask]
            self._redraw()
            QMessageBox.information(
                self, "Success",
                f"Deleted atom(s): {', '.join(map(str, sorted(self.selected)))}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to delete atoms:\n{exc}")
        self.close()

    def _on_closed(self, _result):
        self._mark([])
        self._disable_pick()


# ---------------------------------------------------------------------------
# Fit Molecule to Axis dialog
# ---------------------------------------------------------------------------

class FitDialog(QDialog):
    """Modeless dialog for fitting a molecule along a cell or Cartesian axis."""

    def __init__(self, atoms_ref, redraw_fn, mark_fn,
                 enable_pick_fn, disable_pick_fn, update_spinboxes_fn,
                 parent=None):
        super().__init__(parent)
        self._atoms = atoms_ref
        self._redraw = redraw_fn
        self._mark = mark_fn
        self._enable_pick = enable_pick_fn
        self._disable_pick = disable_pick_fn
        self._update_spinboxes = update_spinboxes_fn  # (params_dict) -> None

        self._pick_target = 'direction'

        self.setWindowTitle("Fit Molecule to Axis")
        self.resize(450, 420)
        self.setModal(False)
        self._build_ui()
        self.finished.connect(self._on_closed)
        # Start with direction picking active
        self._set_pick_target('direction')

    def _build_ui(self):
        layout = QVBoxLayout(self)
        atoms = self._atoms[0]
        layout.addWidget(QLabel(
            f"Total atoms: {len(atoms)}\n"
            "Pick atoms in the 3D view or enter indices manually."
        ))

        # Target axis selector
        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("Target axis:"))
        self._axis_combo = QComboBox()
        self._axis_combo.addItems(['a-axis', 'b-axis', 'c-axis', 'X-axis', 'Y-axis', 'Z-axis'])
        axis_row.addWidget(self._axis_combo)
        layout.addLayout(axis_row)

        # Direction atoms
        dir_header = QHBoxLayout()
        dir_header.addWidget(QLabel("Atoms defining axis direction:"))
        self._btn_pick_dir = QPushButton("Pick")
        self._btn_pick_dir.setCheckable(True)
        self._btn_pick_dir.clicked.connect(lambda: self._set_pick_target('direction'))
        dir_header.addWidget(self._btn_pick_dir)
        layout.addLayout(dir_header)
        self._dir_input = QTextEdit()
        self._dir_input.setMaximumHeight(55)
        self._dir_input.setPlaceholderText("e.g., 0,1,2,3")
        layout.addWidget(self._dir_input)

        # Terminal atoms (optional)
        term_header = QHBoxLayout()
        term_header.addWidget(QLabel("Terminal atom indices (optional):"))
        self._btn_pick_term = QPushButton("Pick")
        self._btn_pick_term.setCheckable(True)
        self._btn_pick_term.clicked.connect(lambda: self._set_pick_target('terminal'))
        term_header.addWidget(self._btn_pick_term)
        layout.addLayout(term_header)
        self._term_input = QTextEdit()
        self._term_input.setMaximumHeight(55)
        self._term_input.setPlaceholderText(f"e.g., {len(atoms)-1}  or leave empty")
        layout.addWidget(self._term_input)

        # Fractional position
        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("Position along axis (0.0 – 1.0):"))
        self._pos_input = QLineEdit("0.5")
        pos_row.addWidget(self._pos_input)
        layout.addLayout(pos_row)

        # VdW margin toggle
        self._vdw_check = QCheckBox("Add VdW radius margin at terminals")
        self._vdw_check.setChecked(True)
        layout.addWidget(self._vdw_check)

        # Buttons
        btn_row = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._execute)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # -- picking -------------------------------------------------------------

    def _set_pick_target(self, target: str):
        self._pick_target = target
        self._btn_pick_dir.setChecked(target == 'direction')
        self._btn_pick_term.setChecked(target == 'terminal')
        self._enable_pick()
        self._refresh_markers()

    def picking_callback(self, idx: int):
        target_widget = (
            self._dir_input if self._pick_target == 'direction' else self._term_input
        )
        current = target_widget.toPlainText().strip()
        indices = []
        if current:
            try:
                indices = [int(x.strip()) for x in current.split(',') if x.strip()]
            except ValueError:
                pass
        if idx in indices:
            indices.remove(idx)
        else:
            indices.append(idx)
        target_widget.setPlainText(", ".join(map(str, indices)))
        self._refresh_markers()

    def _refresh_markers(self):
        all_idx = set()
        for widget in (self._dir_input, self._term_input):
            try:
                txt = widget.toPlainText().strip()
                if txt:
                    all_idx.update(int(x.strip()) for x in txt.split(',') if x.strip())
            except ValueError:
                pass
        self._mark(list(all_idx))

    # -- execution -----------------------------------------------------------

    def _parse_indices(self, text: str) -> list:
        atoms = self._atoms[0]
        if not text:
            return []
        result = []
        for item in text.split(','):
            item = item.strip()
            if not item:
                continue
            idx = int(item)
            if not (0 <= idx < len(atoms)):
                raise ValueError(f"Atom index {idx} out of range (0–{len(atoms)-1})")
            result.append(idx)
        return result

    def _execute(self):
        try:
            atoms = self._atoms[0]
            axis_name = self._axis_combo.currentText()
            dir_text = self._dir_input.toPlainText().strip()
            term_text = self._term_input.toPlainText().strip()
            target_pos = float(self._pos_input.text())

            if not (0.0 <= target_pos <= 1.0):
                raise ValueError("Position must be between 0.0 and 1.0")

            dir_indices = self._parse_indices(dir_text)
            term_indices = self._parse_indices(term_text)

            if len(dir_indices) < 2:
                raise ValueError("At least 2 direction atoms are required.")

            positions = atoms.get_positions()

            # ---- 1. Compute principal axis of direction atoms ---------------
            dir_pos = positions[dir_indices]
            principal_axis = compute_principal_axis(dir_pos)

            # Orient towards terminal (or farthest direction atom)
            dir_centroid = dir_pos.mean(axis=0)
            if term_indices:
                to_end = positions[term_indices].mean(axis=0) - dir_centroid
            else:
                dists = np.linalg.norm(dir_pos - dir_centroid, axis=1)
                to_end = positions[dir_indices[int(np.argmax(dists))]] - dir_centroid
            if np.dot(principal_axis, to_end) < 0:
                principal_axis = -principal_axis

            # ---- 2. Determine target direction --------------------------------
            if axis_name in ('X-axis', 'Y-axis', 'Z-axis'):
                cart_map = {'X-axis': 0, 'Y-axis': 1, 'Z-axis': 2}
                target_dir = np.eye(3)[cart_map[axis_name]]
            else:
                cell = atoms.get_cell()
                cell_map = {'a-axis': 0, 'b-axis': 1, 'c-axis': 2}
                cv = cell[cell_map[axis_name]]
                cv_len = np.linalg.norm(cv)
                if cv_len < 1e-9:
                    raise ValueError(f"Cell vector for {axis_name} has zero length.")
                target_dir = cv / cv_len

            # ---- 3. Translate direction centroid to origin -----------------
            dir_pos = atoms.get_positions()[dir_indices]
            dir_centroid = dir_pos.mean(axis=0)
            atoms.positions -= dir_centroid

            # ---- 4. Rotate molecule to align principal axis with target ----
            mol_dir = principal_axis / np.linalg.norm(principal_axis)
            rot_axis = np.cross(mol_dir, target_dir)
            rot_axis_norm = np.linalg.norm(rot_axis)
            if rot_axis_norm > 1e-6:
                rot_axis /= rot_axis_norm
                rot_angle = np.arccos(np.clip(np.dot(mol_dir, target_dir), -1.0, 1.0))
                rot = Rotation.from_rotvec(rot_angle * rot_axis)
                atoms.positions = rot.apply(atoms.positions)

            # ---- 5. (Cartesian axes only) push perpendicular components to zero
            if axis_name in ('X-axis', 'Y-axis', 'Z-axis'):
                ax_idx = {'X-axis': 0, 'Y-axis': 1, 'Z-axis': 2}[axis_name]
                other = [i for i in range(3) if i != ax_idx]
                dir_now = atoms.get_positions()[dir_indices]
                mean_perp = dir_now.mean(axis=0).copy()
                mean_perp[ax_idx] = 0.0
                if np.linalg.norm(mean_perp) > 1e-10:
                    angle = np.arctan2(mean_perp[other[1]], mean_perp[other[0]])
                    rot2 = Rotation.from_rotvec(-angle * target_dir)
                    atoms.positions = rot2.apply(atoms.positions)
                # Force perpendicular mean to zero
                final_dir = atoms.get_positions()[dir_indices]
                mean_coords = final_dir.mean(axis=0)
                for o in other:
                    atoms.positions[:, o] -= mean_coords[o]

            # ---- 6. Optional VdW margin ------------------------------------
            if self._vdw_check.isChecked():
                vdw_max = len(vdw_radii)
                term_or_dir = term_indices if term_indices else dir_indices
                max_vdw = 0.0
                for i in term_or_dir:
                    n = atoms[i].number
                    r = vdw_radii[n] if (n < vdw_max and vdw_radii[n] > 0) else 1.5
                    max_vdw = max(max_vdw, r)

                if axis_name in ('X-axis', 'Y-axis', 'Z-axis'):
                    ax_idx = {'X-axis': 0, 'Y-axis': 1, 'Z-axis': 2}[axis_name]
                    ax_coords = atoms.get_positions()[:, ax_idx]
                    req = (ax_coords.max() - ax_coords.min()) + 2 * max_vdw
                    cell_params = list(atoms.cell.cellpar())
                    cell_params[ax_idx] = req
                    atoms.set_cell(cellpar_to_cell(cell_params), scale_atoms=False)
                    self._update_spinboxes({
                        'a': cell_params[0], 'b': cell_params[1], 'c': cell_params[2],
                        'alpha': cell_params[3], 'beta': cell_params[4], 'gamma': cell_params[5],
                    })

            # ---- 7. Rotate around target axis to minimise cell overflow ----
            cell = atoms.get_cell()
            positions_now = atoms.get_positions()

            if axis_name in ('X-axis', 'Y-axis', 'Z-axis'):
                ax_idx = {'X-axis': 0, 'Y-axis': 1, 'Z-axis': 2}[axis_name]
                other = [i for i in range(3) if i != ax_idx]
                dir_now = positions_now[dir_indices]
                rot_center = np.zeros(3)
                rot_center[ax_idx] = dir_now[:, ax_idx].mean()
                best = rotation_angle_search(
                    positions_now, target_dir, rot_center, cell, other
                )
                apply_rotation_to_atoms(atoms, best, target_dir, rot_center)

                # Position along axis
                positions_now = atoms.get_positions()
                cur_pos = positions_now[dir_indices][:, ax_idx].mean()
                cell_size = np.linalg.norm(cell[ax_idx])
                target_coord = target_pos * cell_size
                shift = np.zeros(3)
                shift[ax_idx] = target_coord - cur_pos
                atoms.positions += shift

                # Clamp to cell if needed
                ax_coords = atoms.get_positions()[:, ax_idx]
                if ax_coords.min() < 0 or ax_coords.max() > cell_size:
                    mol_len = ax_coords.max() - ax_coords.min()
                    center_pos = (cell_size - mol_len) / 2.0
                    shift2 = np.zeros(3)
                    shift2[ax_idx] = center_pos - ax_coords.min()
                    atoms.positions += shift2

            else:
                cell_map = {'a-axis': 0, 'b-axis': 1, 'c-axis': 2}
                cell_ax_idx = cell_map[axis_name]
                cell_vec = cell[cell_ax_idx]
                cell_dir = cell_vec / np.linalg.norm(cell_vec)

                dir_now = positions_now[dir_indices]
                dir_centroid = dir_now.mean(axis=0)
                rot_center = np.dot(dir_centroid, cell_dir) * cell_dir

                check = [i for i in range(3) if i != cell_ax_idx]
                best = rotation_angle_search(
                    positions_now, cell_dir, rot_center, cell, check
                )
                apply_rotation_to_atoms(atoms, best, cell_dir, rot_center)

                # Position along cell axis
                positions_now = atoms.get_positions()
                dir_now = positions_now[dir_indices]
                cur_coord = np.dot(dir_now.mean(axis=0), cell_dir)
                target_coord = target_pos * np.linalg.norm(cell_vec)
                atoms.positions += (target_coord - cur_coord) * cell_dir

                # Clamp
                final_coords = np.dot(atoms.get_positions(), cell_dir)
                ax_min, ax_max = final_coords.min(), final_coords.max()
                cell_ax_size = np.linalg.norm(cell_vec)
                if ax_min < 0 or ax_max > cell_ax_size:
                    mol_len = ax_max - ax_min
                    center_pos = (cell_ax_size - mol_len) / 2.0
                    atoms.positions += (center_pos - ax_min) * cell_dir

            self._redraw()
            margin_note = "\n(VdW margin added)" if self._vdw_check.isChecked() else ""
            QMessageBox.information(self, "Success",
                                    f"Molecule fitted to {axis_name}.{margin_note}")

        except ValueError as exc:
            QMessageBox.warning(self, "Input Error", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to fit molecule:\n{exc}")

    def _on_closed(self, _result):
        self._mark([])
        self._disable_pick()


# ---------------------------------------------------------------------------
# Group-picking dialog  (molecule-wise selection)
# ---------------------------------------------------------------------------

class GroupPickingDialog(QDialog):
    """First step of group operations: pick atoms → proceed to transforms."""

    def __init__(self, atoms_ref, redraw_fn, mark_fn,
                 enable_pick_fn, disable_pick_fn,
                 open_operation_fn, parent=None):
        super().__init__(parent)
        self._atoms = atoms_ref
        self._redraw = redraw_fn
        self._mark = mark_fn
        self._enable_pick = enable_pick_fn
        self._disable_pick = disable_pick_fn
        self._open_op = open_operation_fn

        self.selected = set()

        # Build connectivity graph once; rebuilt if atoms change
        self._graph = self._build_graph()

        self.setWindowTitle("Select Group")
        self.setModal(False)
        self.resize(300, 160)
        self._build_ui()
        self.finished.connect(self._on_closed)

        # Start picking immediately
        self._refresh_display()
        self.picking_callback = self._pick_cb
        self._enable_pick()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Click atoms in the 3D view.\n"
            "(Clicking selects/deselects the whole molecule.)"
        ))
        self._info_label = QLabel("Selected: 0 atoms")
        layout.addWidget(self._info_label)

        row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(self._select_all)
        btn_none = QPushButton("Deselect All")
        btn_none.clicked.connect(self._deselect_all)
        row.addWidget(btn_all)
        row.addWidget(btn_none)
        layout.addLayout(row)

        proceed_btn = QPushButton("Proceed to Operations")
        proceed_btn.clicked.connect(self._proceed)
        layout.addWidget(proceed_btn)

    def _build_graph(self) -> nx.Graph:
        atoms = self._atoms[0]
        cutoffs = natural_cutoffs(atoms, mult=1.2)
        nl_ij = neighbor_list('ij', atoms, cutoffs)
        G = nx.Graph()
        G.add_nodes_from(range(len(atoms)))
        G.add_edges_from(zip(nl_ij[0], nl_ij[1]))
        return G

    def _pick_cb(self, idx: int):
        component = set(nx.node_connected_component(self._graph, idx))
        if component.issubset(self.selected):
            self.selected -= component
        else:
            self.selected |= component
        self._refresh_display()

    def _refresh_display(self):
        indices = list(self.selected)
        self._mark(indices)
        self._info_label.setText(f"Selected: {len(indices)} atoms")

    def _select_all(self):
        atoms = self._atoms[0]
        self.selected = set(range(len(atoms)))
        self._refresh_display()

    def _deselect_all(self):
        self.selected.clear()
        self._refresh_display()

    def _proceed(self):
        if not self.selected:
            return
        indices = list(self.selected)
        self.close()
        self._open_op(indices)

    def _on_closed(self, _result):
        self._mark([])
        self._disable_pick()


# ---------------------------------------------------------------------------
# Group operation dialog  (translate / rotate selected group)
# ---------------------------------------------------------------------------

class GroupOperationDialog(QDialog):
    """Apply translation and rotation to a pre-selected group of atoms."""

    def __init__(self, atoms_ref, indices: list, redraw_fn, parent=None):
        super().__init__(parent)
        self._atoms = atoms_ref
        self._indices = indices
        self._redraw = redraw_fn

        self.setWindowTitle("Group Operations")
        self.setModal(False)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Selected atoms: {len(self._indices)}"))

        # ---- Translation ---------------------------------------------------
        layout.addWidget(QLabel("=== Translation ==="))
        t_group = QWidget()
        t_layout = QGridLayout(t_group)

        self._t_mode = QButtonGroup(t_group)
        self._t_abc = QRadioButton("ABC mode")
        self._t_xyz = QRadioButton("XYZ mode")
        self._t_abc.setChecked(True)
        self._t_mode.addButton(self._t_abc)
        self._t_mode.addButton(self._t_xyz)
        t_layout.addWidget(self._t_abc, 0, 0)
        t_layout.addWidget(self._t_xyz, 0, 1)

        self._t_labels, self._t_spins = [], []
        for i in range(3):
            lbl = QLabel()
            sb = QDoubleSpinBox()
            sb.setRange(-1000.0, 1000.0)
            sb.setSingleStep(0.1)
            sb.setDecimals(3)
            t_layout.addWidget(lbl, i + 1, 0)
            t_layout.addWidget(sb, i + 1, 1)
            self._t_labels.append(lbl)
            self._t_spins.append(sb)

        self._t_xyz.toggled.connect(self._refresh_t_labels)
        self._refresh_t_labels()

        apply_t = QPushButton("Apply Translation")
        apply_t.clicked.connect(self._apply_translation)
        t_layout.addWidget(apply_t, 4, 0, 1, 2)
        layout.addWidget(t_group)

        # ---- Rotation ------------------------------------------------------
        layout.addWidget(QLabel("=== Rotation ==="))
        r_group = QWidget()
        r_layout = QGridLayout(r_group)

        self._r_mode = QButtonGroup(r_group)
        self._r_abc = QRadioButton("ABC mode")
        self._r_xyz = QRadioButton("XYZ mode")
        self._r_manual = QRadioButton("Manual")
        self._r_abc.setChecked(True)
        for rb in (self._r_abc, self._r_xyz, self._r_manual):
            self._r_mode.addButton(rb)
        r_layout.addWidget(self._r_abc, 0, 0)
        r_layout.addWidget(self._r_xyz, 0, 1)
        r_layout.addWidget(self._r_manual, 0, 2)

        self._r_labels, self._r_spins = [], []
        for i in range(3):
            lbl = QLabel()
            sb = QDoubleSpinBox()
            sb.setRange(-360.0, 360.0)
            sb.setSingleStep(1.0)
            sb.setDecimals(2)
            r_layout.addWidget(lbl, i + 1, 0)
            r_layout.addWidget(sb, i + 1, 1, 1, 2)
            self._r_labels.append(lbl)
            self._r_spins.append(sb)

        # Manual mode extra widget
        self._manual_widget = QWidget()
        m_row = QHBoxLayout(self._manual_widget)
        m_row.setContentsMargins(0, 0, 0, 0)
        m_row.addWidget(QLabel("Axis (idx1, idx2):"))
        self._manual_axis_edit = QLineEdit()
        self._manual_axis_edit.setPlaceholderText("e.g., 0, 5")
        m_row.addWidget(self._manual_axis_edit)
        r_layout.addWidget(self._manual_widget, 4, 0, 1, 3)
        self._manual_widget.setVisible(False)

        for rb in (self._r_xyz, self._r_manual):
            rb.toggled.connect(self._refresh_r_labels)
        self._refresh_r_labels()

        apply_r = QPushButton("Apply Rotation")
        apply_r.clicked.connect(self._apply_rotation)
        r_layout.addWidget(apply_r, 5, 0, 1, 3)
        layout.addWidget(r_group)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    # -- label helpers -------------------------------------------------------

    def _refresh_t_labels(self):
        axes = ['X', 'Y', 'Z'] if self._t_xyz.isChecked() else ['a', 'b', 'c']
        for lbl, axis in zip(self._t_labels, axes):
            lbl.setText(f"{axis}:")

    def _refresh_r_labels(self):
        is_xyz = self._r_xyz.isChecked()
        is_manual = self._r_manual.isChecked()
        self._manual_widget.setVisible(is_manual)
        for sb in self._r_spins:
            sb.setVisible(True)
            sb.setEnabled(True)
        for lbl in self._r_labels:
            lbl.setVisible(True)
        if is_xyz:
            axes = ['X', 'Y', 'Z']
        elif is_manual:
            self._r_labels[0].setText("Angle:")
            for w in (self._r_labels[1], self._r_spins[1],
                      self._r_labels[2], self._r_spins[2]):
                w.setVisible(False)
            return
        else:
            axes = ['a', 'b', 'c']
        for lbl, axis in zip(self._r_labels, axes):
            lbl.setText(f"Around {axis}:")

    # -- operations ----------------------------------------------------------

    def _apply_translation(self):
        atoms = self._atoms[0]
        try:
            if self._t_xyz.isChecked():
                shift = np.array([sb.value() for sb in self._t_spins])
            else:
                cell = atoms.get_cell()
                shift = sum(self._t_spins[i].value() * cell[i] for i in range(3))
            atoms.positions[self._indices] += shift
            self._redraw()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Translation failed:\n{exc}")

    def _apply_rotation(self):
        atoms = self._atoms[0]
        try:
            centroid = atoms.positions[self._indices].mean(axis=0)

            if self._r_manual.isChecked():
                text = self._manual_axis_edit.text().strip()
                if not text:
                    QMessageBox.warning(self, "Warning", "Please specify two atom indices.")
                    return
                parts = [p.strip() for p in text.split(',')]
                if len(parts) != 2:
                    raise ValueError("Exactly two indices required.")
                idx1, idx2 = int(parts[0]), int(parts[1])
                if not (0 <= idx1 < len(atoms) and 0 <= idx2 < len(atoms)):
                    raise ValueError("Atom indices out of range.")
                cell = atoms.get_cell()
                vec = min_image_cart_offset(cell, atoms.positions[idx1], atoms.positions[idx2])
                mag = np.linalg.norm(vec)
                if mag < 1e-9:
                    raise ValueError("Selected atoms are at the same position; cannot define a rotation axis.")
                axis = vec / mag
                angle_deg = self._r_spins[0].value()
                rot = Rotation.from_rotvec(np.radians(angle_deg) * axis)
                for i in self._indices:
                    rel = atoms.positions[i] - centroid
                    atoms.positions[i] = centroid + rot.apply(rel)

            elif self._r_xyz.isChecked():
                angles = [sb.value() for sb in self._r_spins]
                rot = Rotation.from_euler('xyz', angles, degrees=True)
                for i in self._indices:
                    rel = atoms.positions[i] - centroid
                    atoms.positions[i] = centroid + rot.apply(rel)

            else:  # ABC mode
                cell = atoms.get_cell()
                for axis_idx, sb in enumerate(self._r_spins):
                    angle_deg = sb.value()
                    if abs(angle_deg) < 0.01:
                        continue
                    cv = cell[axis_idx]
                    cv_len = np.linalg.norm(cv)
                    if cv_len < 1e-9:
                        raise ValueError(f"Cell vector {'abc'[axis_idx]} has zero length.")
                    axis = cv / cv_len
                    rot = Rotation.from_rotvec(np.radians(angle_deg) * axis)
                    for i in self._indices:
                        rel = atoms.positions[i] - centroid
                        atoms.positions[i] = centroid + rot.apply(rel)

            self._redraw()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Rotation failed:\n{exc}")

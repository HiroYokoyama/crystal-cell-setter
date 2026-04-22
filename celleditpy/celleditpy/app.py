#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py  –  Main application window for CelleditPy.

CellSetterApp orchestrates the 3D viewer, control panels, and dialogs.
Heavy computation is delegated to geometry.py; drawing to renderer.py;
and dialogs to dialogs.py.
"""

import sys
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QGridLayout, QDoubleSpinBox,
    QMessageBox, QSizePolicy, QInputDialog, QComboBox,
    QTabWidget, QCheckBox, QButtonGroup, QRadioButton, QLineEdit,
)
from PyQt6.QtCore import Qt
from pyvistaqt import QtInteractor

import ase
import ase.io
from ase.geometry import cellpar_to_cell
from ase.build import make_supercell

try:
    from .constants import VERSION
    from .geometry import (
        min_image_cart_offset,
        get_vdw_radii_array,
        compute_autofit_cell_params,
        rotation_angle_search,
        apply_rotation_to_atoms,
    )
    from .renderer import (
        draw_origin_label, draw_atoms, draw_bonds,
        draw_atom_labels, draw_cell, make_selection_glyphs,
        MESH_PROPS,
    )
    from .dialogs import DeleteDialog, FitDialog, GroupPickingDialog, GroupOperationDialog
except ImportError:
    from constants import VERSION
    from geometry import (
        min_image_cart_offset,
        get_vdw_radii_array,
        compute_autofit_cell_params,
        rotation_angle_search,
        apply_rotation_to_atoms,
    )
    from renderer import (
        draw_origin_label, draw_atoms, draw_bonds,
        draw_atom_labels, draw_cell, make_selection_glyphs,
        MESH_PROPS,
    )
    from dialogs import DeleteDialog, FitDialog, GroupPickingDialog, GroupOperationDialog


# ---------------------------------------------------------------------------
# Custom PyVista Qt widget
# ---------------------------------------------------------------------------

class CustomQtInteractor(QtInteractor):
    """QtInteractor that suppresses double-click events."""

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

    def mouseDoubleClickEvent(self, event):
        event.accept()


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class CellSetterApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"CelleditPy ver. {VERSION}")
        self.setGeometry(100, 100, 1200, 800)

        # Internal state
        self.atoms = None
        self.camera_state = None
        self.show_atom_indices = True
        self.picking_callback = None      # set by dialogs
        self.selection_actor = None       # handle for yellow highlight mesh
        self._supercell_params = (False, 1, 1, 1)
        self._plotter_picking_enabled = False

        self._build_ui()

    # ==========================================================================
    # UI construction
    # ==========================================================================

    def _build_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ---- Left: control panel ----------------------------------------
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(350)
        control_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        control_layout.setSpacing(6)

        self.control_tabs = QTabWidget()
        control_layout.addWidget(self.control_tabs)

        # Tabs
        self.main_tab = QWidget()
        self.main_tab_layout = QVBoxLayout(self.main_tab)
        self.main_tab_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.main_tab_layout.setSpacing(6)
        self.control_tabs.addTab(self.main_tab, "Main")

        self.group_tab = QWidget()
        self.group_tab_layout = QVBoxLayout(self.group_tab)
        self.group_tab_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.group_tab_layout.setSpacing(6)
        self.control_tabs.addTab(self.group_tab, "Structure Control")

        self.view_tab = QWidget()
        self.view_tab_layout = QVBoxLayout(self.view_tab)
        self.view_tab_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.view_tab_layout.setSpacing(6)
        self.control_tabs.addTab(self.view_tab, "View")

        self._build_main_tab()
        self._build_structure_tab()
        self._build_view_tab()

        main_layout.addWidget(control_panel)

        # ---- Right: 3D viewer -------------------------------------------
        self.plotter = CustomQtInteractor(main_widget)
        self.plotter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.plotter)

        self.plotter.set_background('#919191')
        self.plotter.add_axes()
        self.plotter.enable_parallel_projection()

    # -- Main tab ----------------------------------------------------------

    def _build_main_tab(self):
        lay = self.main_tab_layout

        lay.addWidget(QLabel("=== File Operations ==="))
        self.load_button = QPushButton("Load File (.mol, .cif)")
        self.load_button.clicked.connect(self.load_mol_file)
        lay.addWidget(self.load_button)

        self.save_button = QPushButton("Save as CIF")
        self.save_button.clicked.connect(self.save_cif_file)
        self.save_button.setEnabled(False)
        lay.addWidget(self.save_button)

        lay.addSpacing(20)
        lay.addWidget(QLabel("=== Cell Parameters ==="))

        cell_group = QWidget()
        grid = QGridLayout(cell_group)
        self.param_inputs = {}
        params_info = {
            'a':     (10.0,  1.0, 1000.0, 0.1),
            'alpha': (90.0,  0.0,  180.0, 1.0),
            'b':     (10.0,  1.0, 1000.0, 0.1),
            'beta':  (90.0,  0.0,  180.0, 1.0),
            'c':     (10.0,  1.0, 1000.0, 0.1),
            'gamma': (90.0,  0.0,  180.0, 1.0),
        }
        row = col = 0
        for name, (default, lo, hi, step) in params_info.items():
            sb = QDoubleSpinBox()
            sb.setValue(default)
            sb.setMinimum(lo)
            sb.setMaximum(hi)
            sb.setSingleStep(step)
            sb.setDecimals(3)
            grid.addWidget(QLabel(f"{name}:"), row, col * 2)
            grid.addWidget(sb, row, col * 2 + 1)
            self.param_inputs[name] = sb
            col += 1
            if col > 1:
                col = 0
                row += 1
        lay.addWidget(cell_group)

        self.apply_cell_button = QPushButton("Apply Cell Parameters")
        self.apply_cell_button.clicked.connect(lambda: self.update_cell_and_draw(force_reset=False))
        self.apply_cell_button.setEnabled(False)
        lay.addWidget(self.apply_cell_button)

        self.optimize_button = QPushButton("Auto-fit Cell Size")
        self.optimize_button.clicked.connect(lambda: self.optimize_cell_size(draw=True))
        self.optimize_button.setEnabled(False)
        lay.addWidget(self.optimize_button)

        lay.addSpacing(20)
        lay.addWidget(QLabel("=== Structure Operations ==="))

        self.optimize_coord_button = QPushButton("Shift into Cell")
        self.optimize_coord_button.clicked.connect(self.shift_into_cell)
        self.optimize_coord_button.setEnabled(False)
        lay.addWidget(self.optimize_coord_button)

        self.fix_atom0_button = QPushButton("Rotate into Cell")
        self.fix_atom0_button.clicked.connect(self.rotate_into_cell)
        self.fix_atom0_button.setEnabled(False)
        lay.addWidget(self.fix_atom0_button)

        self.fit_to_cell_button = QPushButton("Fit Molecule to Axis")
        self.fit_to_cell_button.clicked.connect(self.open_fit_dialog)
        self.fit_to_cell_button.setEnabled(False)
        lay.addWidget(self.fit_to_cell_button)

        self.delete_atom_button = QPushButton("Delete Atom(s)")
        self.delete_atom_button.clicked.connect(self.open_delete_dialog)
        self.delete_atom_button.setEnabled(False)
        lay.addWidget(self.delete_atom_button)

        lay.addSpacing(20)
        lay.addWidget(QLabel("=== Supercell ==="))

        sc_group = QWidget()
        sc_grid = QGridLayout(sc_group)
        self.supercell_spinboxes = {}
        for i, axis in enumerate(['a', 'b', 'c']):
            sb = QDoubleSpinBox()
            sb.setDecimals(0)
            sb.setMinimum(1)
            sb.setMaximum(10)
            sb.setValue(1)
            sc_grid.addWidget(QLabel(f"n_{axis}:"), 0, i * 2)
            sc_grid.addWidget(sb, 0, i * 2 + 1)
            self.supercell_spinboxes[axis] = sb

        self.supercell_checkbox = QCheckBox("Show Supercell")
        self.supercell_checkbox.setChecked(False)
        sc_grid.addWidget(self.supercell_checkbox, 1, 0, 1, 6)
        lay.addWidget(sc_group)

        self.supercell_checkbox.stateChanged.connect(self._on_supercell_changed)
        for sb in self.supercell_spinboxes.values():
            sb.valueChanged.connect(self._on_supercell_changed)

    # -- Structure Control tab --------------------------------------------

    def _build_structure_tab(self):
        lay = self.group_tab_layout

        self.group_control_btn = QPushButton("Open Group Control")
        self.group_control_btn.clicked.connect(self.open_group_control)
        lay.addWidget(self.group_control_btn)
        lay.addSpacing(20)

        lay.addWidget(QLabel("=== Transform Controls ==="))

        # -- Translation --
        t_group = QWidget()
        t_layout = QGridLayout(t_group)

        self.translate_header_label = QLabel("Translation (Fractional):")
        t_layout.addWidget(self.translate_header_label, 0, 0, 1, 2)

        self._t_mode = QButtonGroup(t_group)
        self.translate_abc_radio = QRadioButton("ABC mode")
        self.translate_xyz_radio = QRadioButton("XYZ mode")
        self.translate_abc_radio.setChecked(True)
        self._t_mode.addButton(self.translate_abc_radio)
        self._t_mode.addButton(self.translate_xyz_radio)
        t_layout.addWidget(self.translate_abc_radio, 1, 0)
        t_layout.addWidget(self.translate_xyz_radio, 1, 1)

        self.translate_labels = []
        self.translate_spinboxes = []
        for i in range(3):
            lbl = QLabel()
            sb = QDoubleSpinBox()
            sb.setRange(-1000.0, 1000.0)
            sb.setSingleStep(0.1)
            sb.setDecimals(3)
            t_layout.addWidget(lbl, i + 2, 0)
            t_layout.addWidget(sb, i + 2, 1)
            self.translate_labels.append(lbl)
            self.translate_spinboxes.append(sb)

        self.translate_xyz_radio.toggled.connect(self._refresh_translate_labels)
        self._refresh_translate_labels()

        apply_t = QPushButton("Apply Translation")
        apply_t.clicked.connect(self.apply_translation)
        t_layout.addWidget(apply_t, 5, 0, 1, 2)
        lay.addWidget(t_group)

        # -- Rotation --
        r_group = QWidget()
        r_layout = QGridLayout(r_group)
        r_layout.addWidget(QLabel("Rotation (degrees):"), 0, 0, 1, 3)

        self._r_mode = QButtonGroup(r_group)
        self.rotate_abc_radio = QRadioButton("ABC mode")
        self.rotate_xyz_radio = QRadioButton("XYZ mode")
        self.rotate_manual_radio = QRadioButton("Manual")
        self.rotate_abc_radio.setChecked(True)
        for rb in (self.rotate_abc_radio, self.rotate_xyz_radio, self.rotate_manual_radio):
            self._r_mode.addButton(rb)
        r_layout.addWidget(self.rotate_abc_radio, 1, 0)
        r_layout.addWidget(self.rotate_xyz_radio, 1, 1)
        r_layout.addWidget(self.rotate_manual_radio, 1, 2)

        self.rotate_labels = []
        self.rotate_spinboxes = []
        for i in range(3):
            lbl = QLabel()
            sb = QDoubleSpinBox()
            sb.setRange(-360.0, 360.0)
            sb.setSingleStep(1.0)
            sb.setDecimals(2)
            r_layout.addWidget(lbl, i + 2, 0)
            r_layout.addWidget(sb, i + 2, 1, 1, 2)
            self.rotate_labels.append(lbl)
            self.rotate_spinboxes.append(sb)

        self.manual_input_widget = QWidget()
        m_row = QHBoxLayout(self.manual_input_widget)
        m_row.setContentsMargins(0, 0, 0, 0)
        m_row.addWidget(QLabel("Axis (idx1, idx2):"))
        self.manual_axis_edit = QLineEdit()
        self.manual_axis_edit.setPlaceholderText("e.g., 0, 5")
        m_row.addWidget(self.manual_axis_edit)
        r_layout.addWidget(self.manual_input_widget, 5, 0, 1, 3)
        self.manual_input_widget.setVisible(False)

        for rb in (self.rotate_xyz_radio, self.rotate_manual_radio):
            rb.toggled.connect(self._refresh_rotate_labels)
        self._refresh_rotate_labels()

        apply_r = QPushButton("Apply Rotation")
        apply_r.clicked.connect(self.apply_rotation)
        r_layout.addWidget(apply_r, 6, 0, 1, 3)
        lay.addWidget(r_group)

    # -- View tab ---------------------------------------------------------

    def _build_view_tab(self):
        lay = self.view_tab_layout
        lay.addSpacing(20)
        lay.addWidget(QLabel("=== View Controls ==="))

        self.toggle_indices_button = QPushButton("Toggle Atom Indices")
        self.toggle_indices_button.clicked.connect(self.toggle_atom_indices)
        self.toggle_indices_button.setEnabled(False)
        lay.addWidget(self.toggle_indices_button)

        self.reset_camera_button = QPushButton("Reset Camera")
        self.reset_camera_button.clicked.connect(self.reset_camera_view)
        self.reset_camera_button.setEnabled(False)
        lay.addWidget(self.reset_camera_button)

        cam_group = QWidget()
        cam_grid = QGridLayout(cam_group)
        self.camera_buttons = {}
        for i, label in enumerate(['a', 'b', 'c', 'a*', 'b*', 'c*']):
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, l=label: self.set_camera_along_axis(l))
            btn.setEnabled(False)
            cam_grid.addWidget(btn, i // 3, i % 3)
            self.camera_buttons[label] = btn
        lay.addWidget(cam_group)

    # ==========================================================================
    # Label refresh helpers
    # ==========================================================================

    def _refresh_translate_labels(self):
        if self.translate_xyz_radio.isChecked():
            self.translate_header_label.setText("Translation (Å):")
            axes = ['X', 'Y', 'Z']
        else:
            self.translate_header_label.setText("Translation (Fractional):")
            axes = ['a', 'b', 'c']
        for lbl, axis in zip(self.translate_labels, axes):
            lbl.setText(f"{axis}:")

    def _refresh_rotate_labels(self):
        is_xyz = self.rotate_xyz_radio.isChecked()
        is_manual = self.rotate_manual_radio.isChecked()
        self.manual_input_widget.setVisible(is_manual)
        for w in self.rotate_labels + self.rotate_spinboxes:
            w.setVisible(True)
            w.setEnabled(True)
        if is_xyz:
            axes = ['X', 'Y', 'Z']
        elif is_manual:
            self.rotate_labels[0].setText("Angle:")
            for w in (self.rotate_labels[1], self.rotate_spinboxes[1],
                      self.rotate_labels[2], self.rotate_spinboxes[2]):
                w.setVisible(False)
            return
        else:
            axes = ['a', 'b', 'c']
        for lbl, axis in zip(self.rotate_labels, axes):
            lbl.setText(f"Around {axis}:")

    # ==========================================================================
    # Spinbox helpers
    # ==========================================================================

    def _set_spinbox_values(self, params: dict):
        """Write cell parameter dict into the GUI spinboxes (no signals)."""
        for name, sb in self.param_inputs.items():
            if name not in params:
                continue
            sb.blockSignals(True)
            val = float(np.clip(params[name], sb.minimum(), sb.maximum()))
            sb.setValue(val)
            sb.blockSignals(False)

    def _enable_controls(self, enabled: bool):
        """Enable or disable all structure-dependent buttons at once."""
        for btn in [
            self.save_button, self.apply_cell_button, self.optimize_button,
            self.optimize_coord_button, self.fix_atom0_button,
            self.fit_to_cell_button, self.delete_atom_button,
            self.toggle_indices_button, self.reset_camera_button,
        ]:
            btn.setEnabled(enabled)
        for btn in self.camera_buttons.values():
            btn.setEnabled(enabled)
        for sb in self.translate_spinboxes + self.rotate_spinboxes:
            sb.setEnabled(enabled)

    # ==========================================================================
    # Supercell
    # ==========================================================================

    @property
    def _sc(self):
        """Return (show, n_a, n_b, n_c) tuple from the supercell controls."""
        return (
            self.supercell_checkbox.isChecked(),
            int(self.supercell_spinboxes['a'].value()),
            int(self.supercell_spinboxes['b'].value()),
            int(self.supercell_spinboxes['c'].value()),
        )

    def _on_supercell_changed(self):
        if self.atoms is None:
            return
        show, n_a, n_b, n_c = self._sc
        # Only redraw if Show Supercell is active (or checkbox just changed)
        if not show and self.sender() is not self.supercell_checkbox:
            return
        self.draw_scene_manually(force_reset=False,
                                 cell_center=np.zeros(3),
                                 draw_supercell=show)

    # ==========================================================================
    # PyVista picking helpers
    # ==========================================================================

    def enable_plot_picking(self):
        try:
            if hasattr(self.plotter, 'disable_picking'):
                try:
                    self.plotter.disable_picking()
                except Exception:
                    pass
            if hasattr(self.plotter, 'enable_point_picking'):
                self.plotter.enable_point_picking(
                    callback=self.on_atom_picked,
                    show_message=False,
                    left_clicking=True,
                    show_point=False,
                )
                self._plotter_picking_enabled = True
        except Exception as exc:
            print(f"[app] enable_point_picking failed: {exc}")

    def disable_plot_picking(self):
        try:
            if hasattr(self.plotter, 'disable_picking'):
                try:
                    self.plotter.disable_picking()
                except Exception:
                    pass
            self._plotter_picking_enabled = False
        except Exception:
            pass

    def on_atom_picked(self, picked_point):
        """Dispatch to whichever dialog's callback is currently active."""
        if self.atoms is None or picked_point is None or len(picked_point) != 3:
            return
        try:
            positions = self.atoms.get_positions()
            dists = np.linalg.norm(positions - np.array(picked_point), axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] > 1.0:   # 1 Å threshold
                return
            if self.picking_callback is not None:
                self.picking_callback(idx)
        except Exception as exc:
            print(f"[app] on_atom_picked error: {exc}")

    # ==========================================================================
    # Selection markers
    # ==========================================================================

    def draw_selection_markers(self, indices: list):
        if self.selection_actor is not None:
            try:
                self.plotter.remove_actor(self.selection_actor)
            except Exception:
                pass
            self.selection_actor = None
        if not indices or self.atoms is None:
            return
        try:
            glyphs = make_selection_glyphs(self.atoms, indices)
            if glyphs is not None:
                self.selection_actor = self.plotter.add_mesh(
                    glyphs, color='yellow', opacity=0.3,
                    pickable=False, name='selection_markers',
                )
        except Exception as exc:
            print(f"[app] selection marker error: {exc}")

    # ==========================================================================
    # File operations
    # ==========================================================================

    def load_mol_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Structure File", "",
            "Structure Files (*.mol *.cif);;All Files (*.*)",
        )
        if not file_name:
            return
        try:
            self.atoms = ase.io.read(file_name)
            cell_params = self.atoms.cell.cellpar()

            if np.any(cell_params) and not np.all(cell_params[:3] == 0):
                self._set_spinbox_values({
                    'a': cell_params[0], 'b': cell_params[1], 'c': cell_params[2],
                    'alpha': cell_params[3], 'beta': cell_params[4], 'gamma': cell_params[5],
                })
                self.atoms.set_pbc(True)
            else:
                # MOL file – auto-fit then centre
                self.optimize_cell_size(draw=False)
                cell = self.atoms.get_cell()
                cell_center = (cell[0] + cell[1] + cell[2]) / 2.0
                self.atoms.positions += cell_center - self.atoms.get_center_of_mass()
                self.draw_scene_manually(force_reset=False, cell_center=np.zeros(3))
                self.reset_camera_view()

            self._enable_controls(True)
            self.enable_plot_picking()

            if not file_name.lower().endswith('.mol'):
                self.update_cell_and_draw(force_reset=True)

        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{exc}")
            self.atoms = None
            self._enable_controls(False)

    def save_cif_file(self):
        if self.atoms is None:
            QMessageBox.warning(self, "Error", "No structure to save.")
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save as CIF", "",
            "Crystallographic Information File (*.cif)",
        )
        if file_name:
            try:
                ase.io.write(file_name, self.atoms, format='cif')
                QMessageBox.information(self, "Success",
                                        f"File saved:\n{file_name}")
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Failed to save file:\n{exc}")

    # ==========================================================================
    # Cell operations
    # ==========================================================================

    def optimize_cell_size(self, draw: bool = True):
        """Compute and apply minimum cell sizes that contain the molecule."""
        if self.atoms is None:
            return
        try:
            params = compute_autofit_cell_params(self.atoms)
            self._set_spinbox_values(params)
            new_cell = cellpar_to_cell([
                params['a'], params['b'], params['c'],
                params['alpha'], params['beta'], params['gamma'],
            ])
            self.atoms.set_cell(new_cell, scale_atoms=False)
            self.atoms.set_pbc(True)
            if draw:
                self.update_cell_and_draw(force_reset=True)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Auto-fit cell failed:\n{exc}")

    def update_cell_and_draw(self, force_reset: bool = False):
        if self.atoms is None:
            return
        try:
            p = {name: sb.value() for name, sb in self.param_inputs.items()}
            cell_matrix = cellpar_to_cell([
                p['a'], p['b'], p['c'], p['alpha'], p['beta'], p['gamma'],
            ])
            self.atoms.set_cell(cell_matrix)
            self.atoms.set_pbc(True)
            self.draw_scene_manually(force_reset=force_reset, cell_center=np.zeros(3))
        except Exception as exc:
            print(f"[app] Cell parameter error: {exc}")

    # ==========================================================================
    # Scene drawing
    # ==========================================================================

    def draw_scene_manually(self, force_reset: bool = False,
                            cell_center=None, draw_supercell=None):
        if self.atoms is None:
            return

        if cell_center is None:
            cell_center = np.zeros(3)

        show, n_a, n_b, n_c = self._sc
        use_supercell = draw_supercell if draw_supercell is not None else show

        if use_supercell and (n_a > 1 or n_b > 1 or n_c > 1):
            try:
                atoms_draw = make_supercell(self.atoms, np.diag([n_a, n_b, n_c]))
            except Exception as exc:
                print(f"[app] Supercell error: {exc}")
                atoms_draw = self.atoms
        else:
            atoms_draw = self.atoms.copy()
            if atoms_draw.pbc.any():
                try:
                    atoms_draw.wrap(pbc=atoms_draw.pbc)
                except Exception:
                    pass

        # Save camera before clearing
        if not force_reset and self.plotter.camera:
            self.camera_state = self.plotter.camera.copy()

        self.plotter.clear()
        self.plotter.set_background('#919191')
        self.plotter.add_light(pv.Light(
            position=(5, 5, 15), light_type='cameralight', intensity=1.0,
        ))

        draw_origin_label(self.plotter)
        draw_atoms(self.plotter, atoms_draw)
        draw_bonds(self.plotter, atoms_draw)

        # Atom index labels (disabled for supercell display)
        if self.show_atom_indices and not (use_supercell and (n_a > 1 or n_b > 1 or n_c > 1)):
            draw_atom_labels(self.plotter, atoms_draw.get_positions())

        draw_cell(self.plotter, self.atoms)

        try:
            self.plotter.camera.reset_clipping_range()
        except Exception:
            pass

        # Calculate visual centre for camera focal point
        if self.atoms.pbc.any():
            cell = self.atoms.get_cell()
            visual_center = (cell[0] + cell[1] + cell[2]) / 2.0
        else:
            visual_center = cell_center

        if force_reset or not self.camera_state:
            self.plotter.reset_camera()
            self.plotter.camera.focal_point = visual_center
            pos = np.array(self.plotter.camera.position)
            focal = np.array(self.plotter.camera.focal_point)
            self.plotter.camera.position = focal - (pos - focal)
            self.plotter.camera.up = (0, 1, 0)
            self.camera_state = None
        else:
            self.plotter.camera = self.camera_state
            self.plotter.camera.focal_point = visual_center

    def reset_camera_view(self):
        if self.atoms is None:
            return
        self.camera_state = None
        self.draw_scene_manually(force_reset=True, cell_center=np.zeros(3))

    def set_camera_along_axis(self, axis_label: str):
        if self.atoms is None or not self.atoms.pbc.any():
            return
        cell = self.atoms.get_cell()
        a, b, c = cell[0], cell[1], cell[2]
        axis_map = {
            'a': (a, c), 'b': (b, a), 'c': (c, b),
            'a*': (-np.cross(b, c), c),
            'b*': (-np.cross(c, a), a),
            'c*': (-np.cross(a, b), b),
        }
        if axis_label not in axis_map:
            return
        target_vec, up_vec = axis_map[axis_label]
        norm = np.linalg.norm(target_vec)
        if norm < 1e-6:
            return
        direction = target_vec / norm
        center = (a + b + c) / 2.0
        distance = np.linalg.norm(a) * 3.0
        self.plotter.camera.position = center + direction * distance
        self.plotter.camera.focal_point = center
        self.plotter.camera.up = up_vec
        self.plotter.render()

    # ==========================================================================
    # Toggle atom indices
    # ==========================================================================

    def toggle_atom_indices(self):
        if self.atoms is None:
            return
        self.show_atom_indices = not self.show_atom_indices
        self.draw_scene_manually(force_reset=False, cell_center=np.zeros(3))

    # ==========================================================================
    # Structure operations
    # ==========================================================================

    def shift_into_cell(self):
        """Translate molecule so its centre of mass sits at the cell centre."""
        if self.atoms is None:
            return
        if not self.atoms.pbc.any():
            QMessageBox.warning(self, "Warning", "Cell is not set.")
            return
        try:
            cell = self.atoms.get_cell()
            cell_center = (cell[0] + cell[1] + cell[2]) / 2.0
            shift = cell_center - self.atoms.get_center_of_mass()
            self.atoms.positions += shift
            self.draw_scene_manually(force_reset=False, cell_center=np.zeros(3))
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Shift failed:\n{exc}")

    def apply_translation(self):
        if self.atoms is None:
            return
        try:
            if self.translate_xyz_radio.isChecked():
                shift = np.array([sb.value() for sb in self.translate_spinboxes])
            else:
                cell = self.atoms.get_cell()
                shift = sum(
                    self.translate_spinboxes[i].value() * cell[i] for i in range(3)
                )
            self.atoms.positions += shift
            self.draw_scene_manually(force_reset=False, cell_center=np.zeros(3))
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Translation failed:\n{exc}")

    def apply_rotation(self):
        if self.atoms is None:
            return
        try:
            if self.rotate_xyz_radio.isChecked():
                angles = [sb.value() for sb in self.rotate_spinboxes]
                if all(abs(a) < 1e-6 for a in angles):
                    return
                idx_str, ok = QInputDialog.getText(
                    self, "Rotation Centre",
                    f"Atom index for rotation centre (0–{len(self.atoms)-1}):",
                    text="0",
                )
                if not ok:
                    return
                idx = int(idx_str)
                if not (0 <= idx < len(self.atoms)):
                    raise ValueError(f"Index must be 0–{len(self.atoms)-1}")
                center = self.atoms.positions[idx].copy()
                rot = Rotation.from_euler('xyz', angles, degrees=True)
                self.atoms.positions -= center
                self.atoms.positions = rot.apply(self.atoms.positions)
                self.atoms.positions += center

            elif self.rotate_manual_radio.isChecked():
                angle_deg = self.rotate_spinboxes[0].value()
                if abs(angle_deg) < 1e-6:
                    return
                text = self.manual_axis_edit.text().strip()
                if not text:
                    return
                parts = [x.strip() for x in text.split(',')]
                if len(parts) != 2:
                    raise ValueError("Enter exactly two atom indices.")
                idx1, idx2 = int(parts[0]), int(parts[1])
                if idx1 == idx2:
                    raise ValueError("Indices must be different.")
                if not (0 <= idx1 < len(self.atoms) and 0 <= idx2 < len(self.atoms)):
                    raise ValueError(f"Indices must be 0–{len(self.atoms)-1}")
                cell = self.atoms.get_cell()
                vec = min_image_cart_offset(cell, self.atoms.positions[idx1],
                                            self.atoms.positions[idx2])
                norm = np.linalg.norm(vec)
                if norm < 1e-6:
                    raise ValueError("Selected atoms are at the same position.")
                axis = vec / norm
                p1 = self.atoms.positions[idx1]
                center = p1 + vec / 2.0
                apply_rotation_to_atoms(self.atoms, np.radians(angle_deg), axis, center)

            else:  # ABC mode
                if not self.atoms.pbc.any():
                    QMessageBox.warning(self, "Warning",
                                        "Cell is not set. Cannot use ABC mode.")
                    return
                cell = self.atoms.get_cell()
                tolerance = 0.001
                for axis_idx, sb in enumerate(self.rotate_spinboxes):
                    angle = np.radians(sb.value())
                    if abs(angle) < 1e-6:
                        continue
                    cv = cell[axis_idx]
                    axis_dir = cv / np.linalg.norm(cv)
                    positions = self.atoms.get_positions()
                    on_axis = [
                        i for i, pos in enumerate(positions)
                        if np.linalg.norm(pos - np.dot(pos, axis_dir) * axis_dir) < tolerance
                    ]
                    if not on_axis:
                        QMessageBox.warning(
                            self, "Warning",
                            f"No atoms on {'abc'[axis_idx]}-axis within {tolerance} Å."
                        )
                        return
                    centroid = positions[on_axis].mean(axis=0)
                    rot_center = np.dot(centroid, axis_dir) * axis_dir
                    apply_rotation_to_atoms(self.atoms, angle, axis_dir, rot_center)

            self.draw_scene_manually(force_reset=False, cell_center=np.zeros(3))

        except ValueError as exc:
            QMessageBox.warning(self, "Input Error", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Rotation failed:\n{exc}")

    def rotate_into_cell(self):
        """Detect atoms on cell axes / planes and rotate the rest to fit inside."""
        if self.atoms is None:
            return
        if not self.atoms.pbc.any():
            QMessageBox.warning(self, "Warning", "Cell is not set.")
            return
        try:
            cell = self.atoms.get_cell()
            positions = self.atoms.get_positions()
            tolerance = 0.001
            candidates = []

            # Axis detection
            for axis_idx in range(3):
                cv = cell[axis_idx]
                cv_len = np.linalg.norm(cv)
                if cv_len < 1e-6:
                    continue
                axis_dir = cv / cv_len
                on_axis = [
                    i for i, pos in enumerate(positions)
                    if np.linalg.norm(pos - np.dot(pos, axis_dir) * axis_dir) < tolerance
                ]
                if len(on_axis) >= 2:
                    candidates.append({
                        'type': 'axis', 'index_1': axis_idx, 'index_2': None,
                        'name': f"{'abc'[axis_idx]}-axis",
                        'atoms': on_axis, 'count': len(on_axis), 'vector': axis_dir,
                    })

            # Plane detection (planes through origin)
            for i1, i2, pname in [(0, 1, 'ab-plane'), (1, 2, 'bc-plane'), (0, 2, 'ac-plane')]:
                normal = np.cross(cell[i1], cell[i2])
                n_len = np.linalg.norm(normal)
                if n_len < 1e-6:
                    continue
                normal /= n_len
                on_plane = [
                    i for i, pos in enumerate(positions)
                    if abs(np.dot(pos, normal)) < tolerance
                ]
                if len(on_plane) >= 3:
                    candidates.append({
                        'type': 'plane', 'index_1': i1, 'index_2': i2,
                        'name': pname, 'atoms': on_plane,
                        'count': len(on_plane), 'vector': normal,
                    })

            best = None
            force_manual = False

            if candidates:
                temp = max(candidates, key=lambda x: x['count'])
                msg = QMessageBox(self)
                msg.setWindowTitle(f"Auto-detected: {temp['name']}")
                atoms_str = ', '.join(map(str, temp['atoms'][:10]))
                if len(temp['atoms']) > 10:
                    atoms_str += '…'
                msg.setText(
                    f"Detected {temp['count']} atoms on {temp['name']}.\n"
                    f"Indices: {atoms_str}"
                )
                msg.setInformativeText("Select action:")
                btn_auto = msg.addButton("Use Auto-Detected", QMessageBox.ButtonRole.AcceptRole)
                btn_manual = msg.addButton("Manual Select", QMessageBox.ButtonRole.ActionRole)
                msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
                msg.exec()
                clicked = msg.clickedButton()
                if clicked == btn_auto:
                    best = temp
                elif clicked == btn_manual:
                    force_manual = True
                else:
                    return

            if not candidates or force_manual:
                text, ok = QInputDialog.getText(
                    self, "Manual Selection",
                    "Enter 2 atom indices for the fixed rotation axis (e.g., 0, 3):",
                )
                if not ok or not text.strip():
                    return
                manual = [int(x.strip()) for x in text.split(',') if x.strip()]
                if len(manual) != 2:
                    raise ValueError("Exactly two indices required.")
                i1, i2 = manual
                if not (0 <= i1 < len(self.atoms) and 0 <= i2 < len(self.atoms)):
                    raise ValueError("Indices out of range.")
                vec = positions[i2] - positions[i1]
                vlen = np.linalg.norm(vec)
                if vlen < 1e-6:
                    raise ValueError("Atoms are at the same position.")
                best = {
                    'type': 'manual_axis', 'name': 'Custom Vector',
                    'atoms': manual, 'count': 2, 'vector': vec / vlen,
                }

            if best is None:
                return

            rot_axis = best['vector']
            fixed_pos = self.atoms.positions[best['atoms']]

            if best['type'] == 'axis':
                check_axes = [i for i in range(3) if i != best['index_1']]
                centroid = fixed_pos.mean(axis=0)
                rot_center = np.dot(centroid, rot_axis) * rot_axis
            elif best['type'] == 'plane':
                check_axes = [best['index_1'], best['index_2']]
                rot_center = fixed_pos.mean(axis=0)
            else:
                check_axes = [0, 1, 2]
                rot_center = fixed_pos.mean(axis=0)

            best_angle = rotation_angle_search(
                self.atoms.positions, rot_axis, rot_center, cell, check_axes,
            )
            apply_rotation_to_atoms(self.atoms, best_angle, rot_axis, rot_center)

            self.draw_scene_manually(force_reset=False, cell_center=np.zeros(3))
            suffix = " (Manual)" if force_manual else " (Auto)"
            QMessageBox.information(
                self, "Success",
                f"Molecule fitted on {best['name']}{suffix}.",
            )

        except ValueError as exc:
            QMessageBox.warning(self, "Input Error", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Rotate into cell failed:\n{exc}")

    # ==========================================================================
    # Delete atoms
    # ==========================================================================

    def open_delete_dialog(self):
        if self.atoms is None:
            return
        # Wrap atoms in a mutable list so DeleteDialog can replace it
        self._atoms_ref = [self.atoms]
        dlg = DeleteDialog(
            atoms_ref=self._atoms_ref,
            redraw_fn=self._after_delete,
            mark_fn=self.draw_selection_markers,
            enable_pick_fn=self.enable_plot_picking,
            disable_pick_fn=self.disable_plot_picking,
            parent=self,
        )
        self.picking_callback = dlg.picking_callback
        dlg.show()

    def _after_delete(self):
        self.atoms = self._atoms_ref[0]
        self.picking_callback = None
        self.draw_scene_manually(force_reset=False, cell_center=np.zeros(3))

    # ==========================================================================
    # Fit molecule to axis
    # ==========================================================================

    def open_fit_dialog(self):
        if self.atoms is None:
            return
        if not self.atoms.pbc.any():
            QMessageBox.warning(self, "Warning", "Cell is not set.")
            return
        if hasattr(self, '_fit_dialog') and self._fit_dialog.isVisible():
            self._fit_dialog.close()

        self._atoms_ref = [self.atoms]
        self._fit_dialog = FitDialog(
            atoms_ref=self._atoms_ref,
            redraw_fn=self._after_fit,
            mark_fn=self.draw_selection_markers,
            enable_pick_fn=self.enable_plot_picking,
            disable_pick_fn=self.disable_plot_picking,
            update_spinboxes_fn=self._set_spinbox_values,
            parent=self,
        )
        self.picking_callback = self._fit_dialog.picking_callback
        self._fit_dialog.show()

    def _after_fit(self):
        self.atoms = self._atoms_ref[0]
        self.draw_scene_manually(force_reset=False, cell_center=np.zeros(3))

    # ==========================================================================
    # Group operations
    # ==========================================================================

    def open_group_control(self):
        if self.atoms is None:
            return
        if hasattr(self, '_group_dlg') and self._group_dlg.isVisible():
            self._group_dlg.close()

        self._atoms_ref = [self.atoms]
        self._group_dlg = GroupPickingDialog(
            atoms_ref=self._atoms_ref,
            redraw_fn=lambda: self.draw_scene_manually(False, np.zeros(3)),
            mark_fn=self.draw_selection_markers,
            enable_pick_fn=self.enable_plot_picking,
            disable_pick_fn=self.disable_plot_picking,
            open_operation_fn=self._open_group_op,
            parent=self,
        )
        self.picking_callback = self._group_dlg._pick_cb
        self._group_dlg.show()

    def _open_group_op(self, indices: list):
        self.picking_callback = None
        op_dlg = GroupOperationDialog(
            atoms_ref=[self.atoms],
            indices=indices,
            redraw_fn=lambda: self.draw_scene_manually(False, np.zeros(3)),
            parent=self,
        )
        op_dlg.show()

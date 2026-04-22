#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CelleditPy – unit-cell parameter editor for crystal structures.

Author  : Hiromichi Yokoyama
License : GPL-3.0
Repo    : https://github.com/HiroYokoyama/crystal-cell-setter
DOI     : 10.5281/zenodo.17620125

Invocation
----------
Preferred (package mode)::

    celleditpy                # installed CLI entry-point
    python -m celleditpy      # module mode via __main__.py

Also works (script mode)::

    python path/to/main.py    # adds package root to sys.path automatically
"""

import sys
import os

# ---------------------------------------------------------------------------
# Import CellSetterApp in a way that works both as a package module and as a
# directly-executed script.
#
# __package__ is set (non-empty) when Python loads this file as part of a
# package (e.g. `python -m celleditpy`, installed CLI, or `import celleditpy`).
# It is None / empty when the file is run directly with `python main.py`.
# ---------------------------------------------------------------------------
if __package__:
    # Normal package import – relative imports are available.
    from .app import CellSetterApp
else:
    # Script mode: add the directory that *contains* the celleditpy package
    # (i.e. two levels up from this file) so absolute imports resolve.
    _here = os.path.dirname(os.path.abspath(__file__))          # .../celleditpy/
    _root = os.path.dirname(os.path.dirname(_here))             # project root
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from celleditpy.app import CellSetterApp                    # absolute import


from PyQt6.QtWidgets import QApplication


def run_app():
    """Launch the CelleditPy GUI application."""
    app = QApplication(sys.argv)
    window = CellSetterApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CelleditPy – unit-cell parameter editor for crystal structures.

Author  : Hiromichi Yokoyama
License : GPL-3.0
Repo    : https://github.com/HiroYokoyama/crystal-cell-setter
DOI     : 10.5281/zenodo.17620125
"""

import sys
from PyQt6.QtWidgets import QApplication

# Support both `python main.py` (direct) and `python -m celleditpy` / installed CLI.
try:
    from .app import CellSetterApp
except ImportError:
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from celleditpy.app import CellSetterApp


def run_app():
    """Launch the CelleditPy GUI application."""
    app = QApplication(sys.argv)
    window = CellSetterApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()

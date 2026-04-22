"""
CelleditPy – unit-cell parameter editor for crystal structures.

Usage
-----
CLI::

    celleditpy              # installed entry-point
    python -m celleditpy   # run as module

Python API::

    from celleditpy import run_app
    run_app()
"""

from .constants import VERSION

__version__ = VERSION
__author__ = "Hiromichi Yokoyama"
__license__ = "GPL-3.0"

# Lazy import: run_app pulls in Qt/PyVista, so only import on demand.
def run_app():
    """Launch the CelleditPy GUI application."""
    from .main import run_app as _run
    _run()

__all__ = ["run_app", "__version__"]

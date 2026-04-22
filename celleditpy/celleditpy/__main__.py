"""
Entry point for ``python -m celleditpy``.

Invocation methods
------------------
``celleditpy``              installed CLI  (pyproject.toml [project.scripts])
``python -m celleditpy``   module mode    (this file, __package__ is set)
``python __main__.py``     direct script  (__package__ guard fixes imports)
``python main.py``         direct script  (main.py has its own guard)
"""

import sys
import os

# When executed as `python -m celleditpy` __package__ == 'celleditpy'.
# When executed as `python __main__.py`   __package__ is None/empty.
if __package__:
    from .main import run_app
else:
    _here = os.path.dirname(os.path.abspath(__file__))  # .../celleditpy/celleditpy/
    _pkg_parent = os.path.dirname(_here)                # .../celleditpy/  (contains the package)
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)
    from celleditpy.main import run_app

run_app()

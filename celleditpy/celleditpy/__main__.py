"""
``python -m celleditpy``  entry point.

Python executes this file when the package is invoked as a module::

    python -m celleditpy

This is the preferred way to run the app without installing it (i.e. from a
source checkout with the package directory on sys.path).  The installed CLI
command ``celleditpy`` uses the ``[project.scripts]`` entry point defined in
``pyproject.toml`` (``celleditpy.main:run_app``), which is equivalent.

Summary of invocation methods
------------------------------
+-----------------------------------+------------------+
| Command                           | How it works     |
+===================================+==================+
| ``celleditpy``                    | pyproject.toml   |
|                                   | entry-point      |
+-----------------------------------+------------------+
| ``python -m celleditpy``          | __main__.py      |
+-----------------------------------+------------------+
| ``python celleditpy/main.py``     | __package__ fix  |
|                                   | in main.py       |
+-----------------------------------+------------------+
"""

from .main import run_app

run_app()

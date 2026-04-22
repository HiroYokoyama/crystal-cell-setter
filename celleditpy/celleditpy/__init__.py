"""
CelleditPy package.
"""

from .constants import VERSION
from .main import run_app

__version__ = VERSION
__all__ = ["run_app", "VERSION"]

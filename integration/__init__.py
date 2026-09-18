# integration/__init__.py
# Integration module per §8.

from integration.control_loop import run
from integration.shift_log_generator import generate_shift_log

__all__ = ["generate_shift_log", "run"]

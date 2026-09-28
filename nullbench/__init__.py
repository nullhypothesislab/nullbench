"""nullbench - small, honest backtesting: point-in-time universes, realistic costs, and null-hypothesis controls."""
from .engine import equal_weight, first_of_month, on_dates, run, top_n
from .metrics import summary
from .universe import membership_mask, survivors_only, tradable
from .controls import random_selection, shuffled_signal, verdict

__version__ = "0.1.0"
__all__ = ["run", "equal_weight", "on_dates", "first_of_month", "top_n", "summary", "membership_mask",
           "tradable", "survivors_only", "random_selection", "shuffled_signal", "verdict"]

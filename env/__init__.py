"""Environment subpackage – provides price model, venue routing, and reward utilities.

Exports:
    PriceModel – Almgren‑Chriss price dynamics.
    VenueRouter – Dark‑pool / lit venue routing logic.
    compute_reward – Per‑step reward calculation.
"""

from .price_model import PriceModel
from .venue_router import VenueRouter
from .reward import compute_reward

__all__ = ["PriceModel", "VenueRouter", "compute_reward"]

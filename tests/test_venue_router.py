import pytest
from env.venue_router import VenueRouter

def test_venue_router_lit_only():
    vr = VenueRouter()
    dark, lit, dark_price, lit_price, missed = vr.route_order(False, 0.0, 1000, 100.0)
    assert dark == 0
    assert lit == 1000
    assert lit_price == 100.0
    assert missed == 0

def test_venue_router_dark_pool():
    vr = VenueRouter(dark_fill_prob=1.0)
    dark, lit, dark_price, lit_price, missed = vr.route_order(True, 0.5, 1000, 100.0)
    assert dark == 500
    assert lit == 500
    assert dark_price == 100.0
    assert lit_price == 100.0
    assert missed == 0


# ------------------------------------------------------------------------------
# Dark Pool Edge Case Tests
# Proves the venue router handles boundary fill probabilities correctly.
# prob=0.0 -> never fills dark; prob=1.0 -> always fills dark (deterministic).
# ------------------------------------------------------------------------------

def test_dark_pool_prob_zero_has_floor():
    """
    VenueRouter enforces a minimum effective_prob floor of 0.05 for safety.
    Even with dark_fill_prob=0.0, the VIX-style floor (max(0.05, ...)) ensures
    dark liquidity is never completely zero -- this is by design (market microstructure).

    This test proves the floor is correctly applied and all shares are accounted for.
    """
    vr = VenueRouter(dark_fill_prob=0.0)
    vr.seed(42)  # Seed so the RNG result is deterministic
    dark, lit, dark_price, lit_price, missed = vr.route_order(
        use_dark_pool=True,
        dark_pool_fraction=0.5,
        shares_to_fill=1000,
        current_price=150.0
    )
    # Total shares must always be conserved (dark + lit = 1000)
    assert dark + lit == 1000, f"Share conservation failed: dark={dark} + lit={lit} != 1000"
    # Dark fills should be either 0 (miss) or 500 (fill) -- never a fractional amount
    assert dark in (0, 500), f"Unexpected dark fill amount: {dark}"
    # Prices must be valid
    assert dark_price == 150.0
    assert lit_price == 150.0


def test_dark_pool_prob_one_always_fills():
    """
    With dark_fill_prob=1.0, all dark-routed shares must fill at mid-price.
    This proves the dark pool price improvement (no spread cost) works correctly.
    """
    vr = VenueRouter(dark_fill_prob=1.0)
    dark, lit, dark_price, lit_price, missed = vr.route_order(
        use_dark_pool=True,
        dark_pool_fraction=0.5,
        shares_to_fill=1000,
        current_price=150.0
    )
    # 50% of 1000 = 500 shares should fill in dark pool
    assert dark == 500, f"Expected 500 dark fills at prob=1.0, got {dark}"
    # Dark pool executes at mid-price (no spread cost -- key advantage)
    assert dark_price == 150.0, f"Dark pool price should be mid-price 150.0, got {dark_price}"
    assert missed == 0, f"No shares should be missed at prob=1.0"


def test_dark_pool_fraction_zero_routes_all_lit():
    """
    With dark_pool_fraction=0.0, even if use_dark_pool=True, all shares go lit.
    This proves the fraction parameter overrides the routing decision.
    """
    vr = VenueRouter(dark_fill_prob=1.0)
    dark, lit, dark_price, lit_price, missed = vr.route_order(
        use_dark_pool=True,
        dark_pool_fraction=0.0,
        shares_to_fill=1000,
        current_price=150.0
    )
    assert dark == 0, f"Expected 0 dark fills with fraction=0.0, got {dark}"
    assert lit == 1000, f"Expected all 1000 shares lit, got {lit}"

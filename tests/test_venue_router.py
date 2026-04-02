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

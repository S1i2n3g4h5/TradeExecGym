import pytest
import numpy as np
from env.price_model import PriceModel

def test_price_model_initialization():
    pm = PriceModel()
    assert pm.sigma == 0.02
    assert pm.eta == 0.1
    assert pm.gamma == 0.01

def test_price_model_step():
    pm = PriceModel(sigma=0.0)
    pm.reset(initial_price=100.0, seed=42)
    state = pm.step(participation_rate=0.0)
    # With sigma=0, price should be exactly 100.0 (ignoring impact which is 0 for 0 part rate)
    assert np.isclose(state.price, 100.0)
    assert pm.dt == 1.0 / 780.0

def test_price_model_impact():
    pm = PriceModel(sigma=0.0)
    pm.reset(initial_price=100.0)
    # Step 1: 0.1 participation
    state = pm.step(participation_rate=0.1)
    assert state.last_temp_impact_bps > 0
    # Permanent impact for step 1 is now last_perm_impact_bps
    assert state.last_perm_impact_bps > 0.0
    
    # Step 2: 0.0 participation (to see the carry-over from Step 1)
    state = pm.step(participation_rate=0.0)
    # Temporary impact should be 0 because current participation is 0
    assert state.last_temp_impact_bps == 0.0
    # Permanent impact for this step should be 0 because participation is 0
    assert state.last_perm_impact_bps == 0.0
    assert state.price > 100.0  # Price should have shifted up from Step 1


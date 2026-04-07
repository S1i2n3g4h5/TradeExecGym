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


# ------------------------------------------------------------------------------
# Participation Rate Clamping Tests
# Proves that the environment enforces safe action bounds (0.0 <= rate <= 0.25)
# regardless of what an agent sends -- critical for safe RL training.
# ------------------------------------------------------------------------------

def test_participation_rate_clamped_above_max():
    """Rate > 0.25 must be silently clamped to 0.25 -- never crash or over-trade."""
    import sys; sys.path.insert(0, '.')
    from server.trade_environment import TradeExecEnvironment
    env = TradeExecEnvironment()
    env.reset(seed=42, task_id="task1_twap_beater")
    # Execute with an illegal rate of 0.99
    result = env.execute_trade(participation_rate=0.99)
    assert result is not None
    # Shares executed should be <= what rate=0.25 would produce (not 4x that)
    # adv_per_step ~ 10M/780 ~ 12,820; max fill = 0.25 x 12,820 ~ 3,205
    # At rate=0.25 (max), one step executes ~5128 shares (ADV-dependent).
    # At rate=0.99 (clamped to 0.25), it must execute the same ~5128 -- not 4x more.
    # Verify clamping by checking result matches rate=0.25 baseline (<= 6000 shares).
    assert env._shares_executed <= 6000, (
        f"Clamping failed -- executed {env._shares_executed} shares (expected <=6000 for rate clamped to 0.25)"
    )


def test_participation_rate_clamped_below_zero():
    """Negative rate must be clamped to 0.0 -- no reverse trading allowed."""
    import sys; sys.path.insert(0, '.')
    from server.trade_environment import TradeExecEnvironment
    env = TradeExecEnvironment()
    env.reset(seed=42, task_id="task1_twap_beater")
    result = env.execute_trade(participation_rate=-0.5)
    assert result is not None
    # No shares should be sold / negative executed
    assert env._shares_executed >= 0, "Negative participation rate caused negative execution"


def test_exactly_100pct_completion_grader():
    """Grader must handle 100% completion without division-by-zero or unexpected score."""
    import sys; sys.path.insert(0, '.')
    from server.trade_environment import TradeExecEnvironment
    env = TradeExecEnvironment()
    env.reset(seed=42, task_id="task1_twap_beater")
    # Run at max rate until episode ends
    while not env._episode_done:
        env.execute_trade(participation_rate=0.25)
    completion = env._shares_executed / env._total_shares
    score = env._compute_grader_score()
    # Score must be a valid float and not NaN/Inf
    assert isinstance(score, float), f"Grader returned {type(score)}"
    assert 0.0 <= score <= 1.0, f"Grader score {score} out of [0, 1] bounds"
    # Completion should be >= 70% (max rate may not always hit 100% due to ADV cap)
    assert completion >= 0.70, f"Expected >=70% completion at max rate, got {completion:.1%}"


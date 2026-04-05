import pytest
from env.reward import compute_reward
from openenv.core.env_server.types import State

def test_compute_reward_baseline():
    reward = compute_reward(
        state_meta={}, 
        is_current=15.0, 
        is_baseline=20.0, 
        shares_executed=50000, 
        total_shares=100000, 
        is_done=False, 
        slippage_bps=1.0
    )
    # Dense reward: (20 - 15) * 0.1 = 0.5
    assert reward == 0.5

def test_compute_reward_optimal():
    # Check reward at completion (100% done)
    reward = compute_reward(
        state_meta={}, 
        is_current=10.0, 
        is_baseline=20.0, 
        shares_executed=100000, 
        total_shares=100000, 
        is_done=True, 
        slippage_bps=0.0
    )
    # Dense: (20-10)*0.1 = 1.0
    # Delayed: +1.0 (completion)
    # Quality: +0.5 (10.0 < 11.6 is True)
    # Total: 2.5
    assert reward == 2.5

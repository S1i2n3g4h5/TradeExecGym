import pytest
from env.reward import compute_reward
from openenv.core.env_server.types import State

def test_compute_reward_baseline():
<<<<<<< HEAD
    state = State(
        episode_id="test",
        step_count=1,
        task_id="task1_twap_beater",
        shares_remaining=100000,
        current_is_bps=10.0,
        done=False
    )
    reward = compute_reward(state, is_current=15.0, risk_penalty=2.0, tc_penalty=1.0)
    # is_penalty = 15.0 / 100.0 = 0.15
    # risk_penalty = 2.0 / 1000.0 = 0.002
    # tc_penalty = 1.0 / 100.0 = 0.01
    # total penalty = 0.162
    # base_reward = -0.162
    assert reward < 0.0

def test_compute_reward_optimal():
    state = State(
        episode_id="test",
        step_count=10,
        task_id="task1_twap_beater",
        shares_remaining=0,
        current_is_bps=0.0,
        done=True
    )
    # Check if reward scales correctly when all penalties are 0
    reward = compute_reward(state, is_current=0.0, risk_penalty=0.0, tc_penalty=0.0)
    # Reward should be close to 0
    assert reward == 0.0
=======
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
>>>>>>> gh/feature/planning-docs

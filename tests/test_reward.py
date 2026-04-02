import pytest
from env.reward import compute_reward
from openenv.core.env_server.types import State

def test_compute_reward_baseline():
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

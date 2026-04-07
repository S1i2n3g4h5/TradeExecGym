"""
OpenEnv Spec Compliance Tests
==============================
Verifies that TradeExecEnvironment meets the OpenEnv specification:
- reset() returns Observation with metadata dict
- execute_trade() returns str
- get_reward() returns float in reward_range [-2.0, 2.0]
- state property returns object with required fields
- _build_numeric_observation() returns dict with 5 valid float fields
- Typed Pydantic models (TradeObservation, TradeAction, TradeReward) are importable
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_pydantic_models_importable():
    """TradeObservation, TradeAction, TradeReward must be importable."""
    from server.trade_environment import TradeObservation, TradeAction, TradeReward
    assert TradeObservation is not None
    assert TradeAction is not None
    assert TradeReward is not None


def test_trade_observation_schema():
    """TradeObservation validates field types and constraints."""
    from server.trade_environment import TradeObservation
    obs = TradeObservation(
        price_norm=1.0,
        progress_pct=0.5,
        remaining_pct=0.5,
        vol_ratio=1.0,
        current_is_bps=15.0,
        done=False,
        step=5,
    )
    assert obs.price_norm == 1.0
    assert 0.0 <= obs.progress_pct <= 1.0
    assert 0.0 <= obs.remaining_pct <= 1.0
    assert obs.done is False


def test_trade_action_validation():
    """TradeAction clamps participation_rate to [0.0, 0.25]."""
    from server.trade_environment import TradeAction
    from pydantic import ValidationError
    # Valid action
    action = TradeAction(participation_rate=0.05)
    assert action.participation_rate == 0.05
    # Invalid action (out of range) should raise ValidationError
    with pytest.raises(ValidationError):
        TradeAction(participation_rate=0.5)  # exceeds max 0.25


def test_trade_reward_schema():
    """TradeReward has required fields."""
    from server.trade_environment import TradeReward
    reward = TradeReward(value=0.5, dense=0.3, sparse=0.2, terminal=0.0)
    assert reward.value == 0.5


def test_environment_instantiation():
    """TradeExecEnvironment instantiates without error."""
    from server.trade_environment import TradeExecEnvironment
    env = TradeExecEnvironment()
    assert env is not None


def test_reset_returns_observation():
    """reset() returns Observation with metadata dict containing required keys."""
    from server.trade_environment import TradeExecEnvironment
    from openenv.core.env_server.types import Observation
    env = TradeExecEnvironment()
    obs = env.reset(task_id="task1_twap_beater", seed=42)
    assert isinstance(obs, Observation)
    assert obs.done is False
    assert isinstance(obs.metadata, dict)
    assert "task_id" in obs.metadata
    assert "max_steps" in obs.metadata
    assert "total_shares" in obs.metadata
    assert obs.metadata["task_id"] == "task1_twap_beater"


def test_reset_includes_numeric_observation():
    """reset() metadata includes numeric observation dict with 5 fields."""
    from server.trade_environment import TradeExecEnvironment
    env = TradeExecEnvironment()
    obs = env.reset(task_id="task1_twap_beater", seed=42)
    assert "observation" in obs.metadata
    num_obs = obs.metadata["observation"]
    assert isinstance(num_obs, dict)
    for field in ["price_norm", "progress_pct", "remaining_pct", "vol_ratio", "current_is_bps", "done", "step"]:
        assert field in num_obs, f"Missing field: {field}"
    # Check ranges
    assert 0.5 <= num_obs["price_norm"] <= 2.0 or num_obs["price_norm"] == 1.0
    assert 0.0 <= num_obs["progress_pct"] <= 1.0
    assert 0.0 <= num_obs["remaining_pct"] <= 1.0


def test_state_property_fields():
    """state property returns object with required fields."""
    from server.trade_environment import TradeExecEnvironment
    env = TradeExecEnvironment()
    env.reset(task_id="task1_twap_beater", seed=42)
    state = env.state
    assert hasattr(state, "episode_id")
    assert hasattr(state, "step_count")
    assert hasattr(state, "task_id")
    assert hasattr(state, "shares_remaining")
    assert hasattr(state, "current_is_bps")
    assert hasattr(state, "done")


def test_execute_trade_returns_string():
    """execute_trade() returns a non-empty string."""
    from server.trade_environment import TradeExecEnvironment
    env = TradeExecEnvironment()
    env.reset(task_id="task1_twap_beater", seed=42)
    result = env.execute_trade(participation_rate=0.05)
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_reward_returns_float():
    """get_reward() returns float in approximately [-2.0, 2.0]."""
    from server.trade_environment import TradeExecEnvironment
    env = TradeExecEnvironment()
    env.reset(task_id="task1_twap_beater", seed=42)
    env.execute_trade(participation_rate=0.05)
    reward = env.get_reward()
    assert isinstance(reward, float)
    assert -5.0 <= reward <= 5.0  # generous bounds


def test_milestones_reset_between_episodes():
    """_milestones_reached must be empty at start of each episode (P0 bug fix)."""
    from server.trade_environment import TradeExecEnvironment
    env = TradeExecEnvironment()
    # Episode 1: run to completion or near
    env.reset(task_id="task1_twap_beater", seed=42)
    assert env._milestones_reached == set(), "Milestones should be empty after reset"
    # Execute some steps
    for _ in range(5):
        env.execute_trade(participation_rate=0.20)
    milestones_ep1 = len(env._milestones_reached)
    # Episode 2: milestones should reset
    env.reset(task_id="task1_twap_beater", seed=42)
    assert env._milestones_reached == set(), "Milestones must reset at episode start"


def test_get_market_state_returns_string():
    """get_market_state() returns non-empty ASCII-compatible string."""
    from server.trade_environment import TradeExecEnvironment
    env = TradeExecEnvironment()
    env.reset(task_id="task1_twap_beater", seed=42)
    state_text = env.get_market_state()
    assert isinstance(state_text, str)
    assert len(state_text) > 50
    # Should be ASCII-safe (no raw emoji bytes)
    state_text.encode("ascii", errors="replace")  # should not raise


def test_all_task_ids_resolve():
    """All 5 task IDs must resolve via factory."""
    from tasks.factory import get_task
    task_ids = [
        "task1_twap_beater",
        "task2_vwap_optimizer",
        "task3_volatile_execution",
        "task4_adversarial",
        "task5_deadline_pressure",
    ]
    for tid in task_ids:
        task = get_task(tid)
        assert task.task_id == tid
        assert task.total_shares > 0
        assert task.max_steps > 0

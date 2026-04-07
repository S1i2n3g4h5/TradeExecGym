import pytest
from env.reward import compute_reward
from openenv.core.env_server.types import State


def test_compute_reward_baseline():
    """Dense reward: agent IS < TWAP IS -> positive reward."""
    reward = compute_reward(
        state_meta={},
        is_current=15.0,
        is_baseline=20.0,
        shares_executed=50000,
        total_shares=100000,
        is_done=False,
        slippage_bps=1.0,
        participation_rate=0.05,  # non-zero -> no zero-rate penalty
    )
    # Dense reward: (20 - 15) * 0.1 = 0.5
    assert reward == 0.5


def test_compute_reward_optimal():
    """Check reward at completion with elite IS (beats AC Optimal)."""
    reward = compute_reward(
        state_meta={},
        is_current=10.0,
        is_baseline=20.0,
        shares_executed=100000,
        total_shares=100000,
        is_done=True,
        slippage_bps=0.0,
        participation_rate=0.05,  # non-zero -> no zero-rate penalty
    )
    # Dense: (20-10)*0.1 = 1.0
    # Delayed: +1.0 (>95% completion)
    # Quality: +0.5 (10.0 < 20.0 * 0.58 = 11.6 -> True)
    # Total: 2.5
    assert reward == 2.5


def test_compute_reward_zero_rate_penalty():
    """Zero participation_rate incurs -0.05 penalty (prevents do-nothing strategy)."""
    reward_zero = compute_reward(
        state_meta={},
        is_current=15.0,
        is_baseline=20.0,
        shares_executed=50000,
        total_shares=100000,
        is_done=False,
        slippage_bps=0.0,
        participation_rate=0.0,  # zero rate -> penalty
    )
    reward_active = compute_reward(
        state_meta={},
        is_current=15.0,
        is_baseline=20.0,
        shares_executed=50000,
        total_shares=100000,
        is_done=False,
        slippage_bps=1.0,
        participation_rate=0.05,  # active -> no penalty
    )
    # Zero-rate should be strictly less than active-rate reward
    assert reward_zero < reward_active
    # Penalty is -0.05 from the zero-rate on top of dense reward
    assert reward_zero == pytest.approx(0.5 - 0.05, abs=1e-9)


def test_compute_reward_negative_dense():
    """Agent IS > TWAP IS -> negative dense reward signal."""
    reward = compute_reward(
        state_meta={},
        is_current=30.0,
        is_baseline=20.0,
        shares_executed=50000,
        total_shares=100000,
        is_done=False,
        slippage_bps=5.0,
        participation_rate=0.05,
    )
    # Dense: (20 - 30) * 0.1 = -1.0
    assert reward == pytest.approx(-1.0, abs=1e-9)


def test_compute_reward_incomplete_terminal_penalty():
    """Less than 95% completion at terminal step -> -0.5 penalty."""
    reward = compute_reward(
        state_meta={},
        is_current=15.0,
        is_baseline=20.0,
        shares_executed=80000,   # 80% complete
        total_shares=100000,
        is_done=True,
        slippage_bps=0.0,
        participation_rate=0.05,
    )
    # Dense: (20-15) * 0.1 = 0.5
    # Terminal: -0.5 (< 95% fill)
    assert reward == pytest.approx(0.0, abs=1e-9)

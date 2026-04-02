"""Unit tests for Phase 3 task framework."""
import pytest
from tasks.factory import get_task
from tasks.base_task import BaseTradeTask
from tasks.task1_twap import TaskTwapBeater
from tasks.task2_vwap import TaskVwapOptimizer
from tasks.task3_volatile import TaskVolatileExecution
from tasks.task4_adversary import TaskAdversary
from tasks.task5_deadline import TaskDeadlinePressure


def test_factory_returns_correct_types():
    """Factory returns the right subclass for each task_id."""
    assert isinstance(get_task("task1_twap_beater"), TaskTwapBeater)
    assert isinstance(get_task("task2_vwap_optimizer"), TaskVwapOptimizer)
    assert isinstance(get_task("task3_volatile_execution"), TaskVolatileExecution)
    assert isinstance(get_task("task4_adversarial"), TaskAdversary)
    assert isinstance(get_task("task5_deadline_pressure"), TaskDeadlinePressure)


def test_factory_unknown_falls_back():
    """Unknown task_id falls back to task1."""
    task = get_task("nonexistent_task")
    assert isinstance(task, TaskTwapBeater)


def test_task_configs():
    """Each task has correct config values."""
    t1 = get_task("task1_twap_beater")
    assert t1.total_shares == 100_000
    assert t1.max_steps == 30
    assert t1.sigma == 0.02

    t3 = get_task("task3_volatile_execution")
    assert t3.total_shares == 400_000
    assert t3.sigma == 0.06  # 3x volatility

    t4 = get_task("task4_adversarial")
    assert t4.total_shares == 600_000
    assert t4.max_steps == 120

    t5 = get_task("task5_deadline_pressure")
    assert t5.total_shares == 1_000_000
    assert t5.max_steps == 80


def test_base_grader_perfect_score():
    """Base grader gives ~1.0 for perfect completion with 0 IS."""
    task = get_task("task1_twap_beater")
    score = task.get_grader_score(
        shares_executed=100_000,
        total_shares=100_000,
        current_is=0.0,
        twap_is=25.0,
        vwap_is=20.0,
    )
    assert score == 1.0


def test_base_grader_zero_completion():
    """Base grader gives ~0 for zero completion."""
    task = get_task("task1_twap_beater")
    score = task.get_grader_score(
        shares_executed=0,
        total_shares=100_000,
        current_is=50.0,
        twap_is=25.0,
        vwap_is=20.0,
    )
    assert score == 0.0


def test_adversary_no_penalty_with_varied_rates():
    """Adversary does NOT penalize if participation varies enough."""
    task = TaskAdversary()
    penalties = []
    for i in range(10):
        # Alternate between very different rates
        rate = 0.05 if i % 2 == 0 else 0.15
        p = task.on_trade_step(
            step_count=i,
            participation_rate=rate,
            current_price=150.0,
            shares_executed=i * 5000,
            shares_remaining=600_000 - i * 5000,
        )
        penalties.append(p)
    # After 5 steps of alternating rates, stdev should be > 0.005
    # so no penalty should be applied
    assert all(p == 0.0 for p in penalties)


def test_adversary_penalizes_uniform_rates():
    """Adversary DOES penalize if participation is perfectly uniform."""
    task = TaskAdversary()
    penalties = []
    for i in range(10):
        p = task.on_trade_step(
            step_count=i,
            participation_rate=0.05,  # constant → predictable
            current_price=150.0,
            shares_executed=i * 5000,
            shares_remaining=600_000 - i * 5000,
        )
        penalties.append(p)
    # After 5 steps of identical rates, stdev == 0 < 0.005 → penalty
    assert any(p > 0.0 for p in penalties), "Adversary should penalize uniform rates"


def test_deadline_grader_incomplete():
    """Deadline task returns 0.0 if < 99.9% complete."""
    task = TaskDeadlinePressure()
    score = task.get_grader_score(
        shares_executed=990_000,  # 99.0% — not enough
        total_shares=1_000_000,
        current_is=10.0,
        twap_is=25.0,
        vwap_is=20.0,
    )
    assert score == 0.0


def test_deadline_grader_complete():
    """Deadline task returns > 0 if 99.9%+ complete."""
    task = TaskDeadlinePressure()
    score = task.get_grader_score(
        shares_executed=999_999,  # 99.9999%
        total_shares=1_000_000,
        current_is=10.0,
        twap_is=25.0,
        vwap_is=20.0,
    )
    assert score > 0.0

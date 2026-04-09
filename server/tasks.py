from __future__ import annotations

import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

try:
    from models import SEED_CONFIG
except Exception:
    SEED_CONFIG: Dict[str, Any] = {}


def _clamp01(x: float) -> float:
    return round(max(0.0001, min(0.9999, float(x))), 4)


@dataclass
class EpisodeRecord:
    task_id: int
    shares_executed: int
    total_shares: int
    current_is_bps: float
    twap_is_bps: float
    vwap_is_bps: float
    ac_is_bps: float
    step_count: int = 0
    max_steps: int = 1
    participation_history: List[float] = field(default_factory=list)
    dark_pool_usage: float = 0.0


def _is_quality_score(current_is: float, ac_is: float) -> float:
    ratio = current_is / max(1.0, ac_is)
    if ratio <= 1.0:
        return 1.0
    return max(0.0, 1.0 - (ratio - 1.0) / 2.0)


def _baseline_bonus(current_is: float, twap_is: float, vwap_is: float) -> float:
    return (0.10 if current_is < twap_is else 0.0) + (0.10 if current_is < vwap_is else 0.0)


def _grade_record_task1(record: EpisodeRecord) -> float:
    """
    Easy task — TWAP Beater.
    Score = 50% IS quality (vs AC), 30% completion, 20% benchmark beating.
    """
    completion = record.shares_executed / max(1, record.total_shares)
    score = (
        0.50 * _is_quality_score(record.current_is_bps, record.ac_is_bps)
        + 0.30 * completion
        + _baseline_bonus(record.current_is_bps, record.twap_is_bps, record.vwap_is_bps)
    )
    return _clamp01(score)


def _grade_record_task2(record: EpisodeRecord) -> float:
    """
    Medium task — VWAP Optimizer.
    Slightly stronger emphasis on VWAP beating.
    """
    completion = record.shares_executed / max(1, record.total_shares)
    vwap_edge = 0.15 if record.current_is_bps < record.vwap_is_bps else 0.0
    twap_edge = 0.05 if record.current_is_bps < record.twap_is_bps else 0.0
    score = (
        0.50 * _is_quality_score(record.current_is_bps, record.ac_is_bps)
        + 0.30 * completion
        + vwap_edge
        + twap_edge
    )
    return _clamp01(score)


def _grade_record_task3(record: EpisodeRecord) -> float:
    """
    Hard task — Volatile Execution.
    Adds mild dark-pool usage reward when volatility task conditions apply.
    """
    completion = record.shares_executed / max(1, record.total_shares)
    dp_bonus = min(0.05, max(0.0, record.dark_pool_usage))
    score = (
        0.50 * _is_quality_score(record.current_is_bps, record.ac_is_bps)
        + 0.30 * completion
        + _baseline_bonus(record.current_is_bps, record.twap_is_bps, record.vwap_is_bps)
        + dp_bonus
    )
    return _clamp01(score)


def grade_episode(record: EpisodeRecord) -> float:
    """Route to the correct grader by task_id. Returns float strictly in (0.0, 1.0)."""
    # Defensive mapping: try to use the task-specific grader from the class if possible
    try:
        from server.tasks import get_task
        task_instance = get_task(f"task_{record.task_id}")
        score = task_instance.get_grader_score(
            shares_executed=record.shares_executed,
            total_shares=record.total_shares,
            current_is=record.current_is_bps,
            twap_is=record.twap_is_bps,
            vwap_is=record.vwap_is_bps,
            ac_is=record.ac_is_bps
        )
    except Exception:
        # Fallback to hardcoded logic for task 1-3
        if record.task_id == 1:
            score = _grade_record_task1(record)
        elif record.task_id == 2:
            score = _grade_record_task2(record)
        elif record.task_id == 3:
            score = _grade_record_task3(record)
        else:
            score = 0.0001
    return _clamp01(score)


class BaseTradeTask:
    def __init__(self) -> None:
        self.task_id = "base_task"
        self.total_shares = 100_000
        self.max_steps = 30
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Base trade execution task."

    def reset(self) -> None:
        pass

    def on_trade_step(
        self,
        step_count: int,
        participation_rate: float,
        current_price: float,
        shares_executed: int,
        shares_remaining: int,
    ) -> float:
        return 0.0

    def get_market_narrative(
        self,
        step_count: int,
        shares_remaining: int,
        current_is: float,
        is_high_volatility: bool,
    ) -> str:
        progress = (step_count / max(1, self.max_steps)) * 100
        return (
            f"[{self.task_id}] {progress:.0f}% done | "
            f"{shares_remaining:,} shares left | IS={current_is:.2f} bps."
        )

    def get_winning_secret(self) -> str:
        return "Keep completion pace while minimizing IS vs TWAP/VWAP baselines."

    def get_grader_score(
        self,
        shares_executed: int,
        total_shares: int,
        current_is: float,
        twap_is: float,
        vwap_is: float,
        ac_is: float = 14.0,
    ) -> float:
        completion = shares_executed / max(1, total_shares)
        score = (
            0.50 * _is_quality_score(current_is, ac_is)
            + 0.30 * completion
            + _baseline_bonus(current_is, twap_is, vwap_is)
        )
        return _clamp01(score)


class TaskTwapBeater(BaseTradeTask):
    def __init__(self) -> None:
        super().__init__()
        self.task_id = "task1_twap_beater"
        self.total_shares = 100_000
        self.max_steps = 30
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Buy 100K shares in 30 steps and beat TWAP."


class TaskVwapOptimizer(BaseTradeTask):
    def __init__(self) -> None:
        super().__init__()
        self.task_id = "task2_vwap_optimizer"
        self.total_shares = 250_000
        self.max_steps = 60
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Sell 250K shares in 60 steps and beat VWAP."


class TaskVolatileExecution(BaseTradeTask):
    def __init__(self) -> None:
        super().__init__()
        self.task_id = "task3_volatile_execution"
        self.total_shares = 400_000
        self.max_steps = 90
        self.arrival_price = 150.0
        self.sigma = 0.06
        self.description = "Buy 400K shares under 3x volatility with dark pool usage."


class TaskAdversary(BaseTradeTask):
    def __init__(self) -> None:
        super().__init__()
        self.task_id = "task4_adversarial"
        self.total_shares = 600_000
        self.max_steps = 120
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Sell 600K shares against pattern-detecting HFT adversary."
        self.participation_history: List[float] = []
        self.leakage_penalty_base = 15.0
        self._episode_seed = 42

    def reset(self) -> None:
        self.participation_history = []

    @staticmethod
    def _autocorr(data: List[float]) -> float:
        if len(data) < 3:
            return 0.0
        x = data[:-1]
        y = data[1:]
        if np.std(x) == 0 or np.std(y) == 0:
            return 1.0
        return float(np.corrcoef(x, y)[0, 1])

    def on_trade_step(
        self,
        step_count: int,
        participation_rate: float,
        current_price: float,
        shares_executed: int,
        shares_remaining: int,
    ) -> float:
        self.participation_history.append(float(participation_rate))
        if len(self.participation_history) > 6:
            self.participation_history.pop(0)
        if len(self.participation_history) < 5:
            return 0.0
        std_dev = statistics.stdev(self.participation_history)
        lag1 = self._autocorr(self.participation_history)
        if std_dev < 0.005 or abs(lag1) > 0.70:
            rng = random.Random(int(self._episode_seed) + int(step_count))
            return self.leakage_penalty_base + rng.uniform(-5.0, 5.0)
        return 0.0


class TaskDeadlinePressure(BaseTradeTask):
    def __init__(self) -> None:
        super().__init__()
        self.task_id = "task5_deadline_pressure"
        self.total_shares = 1_000_000
        self.max_steps = 80
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Buy 1M shares with strict completion pressure."

    def get_grader_score(
        self,
        shares_executed: int,
        total_shares: int,
        current_is: float,
        twap_is: float,
        vwap_is: float,
        ac_is: float = 14.0,
    ) -> float:
        completion = shares_executed / max(1, total_shares)
        if completion < 0.999:
            return _clamp01(max(0.0001, completion * 0.15))
        return super().get_grader_score(
            shares_executed=shares_executed,
            total_shares=total_shares,
            current_is=current_is,
            twap_is=twap_is,
            vwap_is=vwap_is,
            ac_is=ac_is,
        )


def get_task(task_id: str) -> BaseTradeTask:
    registry = {
        1: TaskTwapBeater,
        2: TaskVwapOptimizer,
        3: TaskVolatileExecution,
        4: TaskAdversary,
        5: TaskDeadlinePressure,
        "task_1": TaskTwapBeater,
        "task_2": TaskVwapOptimizer,
        "task_3": TaskVolatileExecution,
        "task_4": TaskAdversary,
        "task_5": TaskDeadlinePressure,
        "task1_twap_beater": TaskTwapBeater,
        "task2_vwap_optimizer": TaskVwapOptimizer,
        "task3_volatile_execution": TaskVolatileExecution,
        "task4_adversarial": TaskAdversary,
        "task5_deadline_pressure": TaskDeadlinePressure,
    }
    return registry.get(task_id, TaskTwapBeater)()


def _as_payload(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for arg in args:
        if isinstance(arg, dict):
            payload.update(arg)
        elif isinstance(arg, EpisodeRecord):
            payload.update(arg.__dict__)
        elif hasattr(arg, "__dict__"):
            payload.update(vars(arg))
    payload.update(kwargs)
    for key in ("metrics", "info", "observation", "state"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            payload.update(nested)
    return payload


def _task_num(task_id_like: Any) -> int:
    tid = str(task_id_like or "")
    if tid in {"1", "task_1", "task1_twap_beater"}:
        return 1
    if tid in {"2", "task_2", "task2_vwap_optimizer"}:
        return 2
    if tid in {"3", "task_3", "task3_volatile_execution"}:
        return 3
    return 1


def _payload_to_record(task_num: int, payload: Dict[str, Any]) -> EpisodeRecord:
    return EpisodeRecord(
        task_id=task_num,
        shares_executed=int(payload.get("shares_executed", payload.get("executed_shares", payload.get("filled_shares", 0)))),
        total_shares=int(payload.get("total_shares", 100_000 if task_num == 1 else (250_000 if task_num == 2 else 400_000))),
        current_is_bps=float(payload.get("current_is", payload.get("current_is_bps", payload.get("is_bps", 0.0)))),
        twap_is_bps=float(payload.get("twap_is", 25.0)),
        vwap_is_bps=float(payload.get("vwap_is", 20.0)),
        ac_is_bps=float(payload.get("ac_is", 14.0)),
        step_count=int(payload.get("step_count", payload.get("step", 0))),
        max_steps=int(payload.get("max_steps", 30 if task_num == 1 else (60 if task_num == 2 else 90))),
        participation_history=list(payload.get("participation_history", [])),
        dark_pool_usage=float(payload.get("dark_pool_usage", payload.get("dark_pool_fraction", 0.0))),
    )


def task_1(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    payload = _as_payload(*args, **kwargs)
    out = {"task_id": "task_1"}
    if payload.get("seed") is not None:
        out["seed"] = int(payload["seed"])
    return out


def task_2(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    payload = _as_payload(*args, **kwargs)
    out = {"task_id": "task_2"}
    if payload.get("seed") is not None:
        out["seed"] = int(payload["seed"])
    return out


def task_3(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    payload = _as_payload(*args, **kwargs)
    out = {"task_id": "task_3"}
    if payload.get("seed") is not None:
        out["seed"] = int(payload["seed"])
    return out


def generate_task_1(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return task_1(*args, **kwargs)


def generate_task_2(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return task_2(*args, **kwargs)


def generate_task_3(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return task_3(*args, **kwargs)


def grade_task1(*args: Any, **kwargs: Any) -> float:
    if args and isinstance(args[0], EpisodeRecord):
        return _grade_record_task1(args[0])
    payload = _as_payload(*args, **kwargs)
    rec = _payload_to_record(1, payload)
    return _clamp01(_grade_record_task1(rec))


def grade_task2(*args: Any, **kwargs: Any) -> float:
    if args and isinstance(args[0], EpisodeRecord):
        return _grade_record_task2(args[0])
    payload = _as_payload(*args, **kwargs)
    rec = _payload_to_record(2, payload)
    return _clamp01(_grade_record_task2(rec))


def grade_task3(*args: Any, **kwargs: Any) -> float:
    if args and isinstance(args[0], EpisodeRecord):
        return _grade_record_task3(args[0])
    payload = _as_payload(*args, **kwargs)
    rec = _payload_to_record(3, payload)
    return _clamp01(_grade_record_task3(rec))


def grade_task_1(*args: Any, **kwargs: Any) -> float:
    return grade_task1(*args, **kwargs)


def grade_task_2(*args: Any, **kwargs: Any) -> float:
    return grade_task2(*args, **kwargs)


def grade_task_3(*args: Any, **kwargs: Any) -> float:
    return grade_task3(*args, **kwargs)


def task_grader(*args: Any, **kwargs: Any) -> float:
    payload = _as_payload(*args, **kwargs)
    n = _task_num(payload.get("task_id", payload.get("id", "task_1")))
    if n == 1:
        return grade_task1(payload)
    if n == 2:
        return grade_task2(payload)
    return grade_task3(payload)


__all__ = [
    "SEED_CONFIG",
    "EpisodeRecord",
    "grade_task1",
    "grade_task2",
    "grade_task3",
    "grade_episode",
    "BaseTradeTask",
    "TaskTwapBeater",
    "TaskVwapOptimizer",
    "TaskVolatileExecution",
    "TaskAdversary",
    "TaskDeadlinePressure",
    "get_task",
    "task_1",
    "task_2",
    "task_3",
    "generate_task_1",
    "generate_task_2",
    "generate_task_3",
    "grade_task_1",
    "grade_task_2",
    "grade_task_3",
    "task_grader",
]

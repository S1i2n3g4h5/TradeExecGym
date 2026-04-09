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


@dataclass
class EpisodeRecord:
    task_id: int
    initial_money: float
    final_money: float
    storage_value: float
    total_reward: float
    days_elapsed: int
    max_days: int
    withered_count: int
    drought_days: int
    healthy_days: int
    sell_events: List[Dict[str, Any]] = field(default_factory=list)


def _clamp01(x: float) -> float:
    return round(max(0.01, min(0.99, float(x))), 4)


def _grade_record_task1(record: EpisodeRecord) -> float:
    """
    Easy task — single crop, stable climate.
    Perfect score = double your starting money.
    """
    if record.initial_money <= 0:
        return 0.01

    net_worth = record.final_money + record.storage_value
    ratio = net_worth / (record.initial_money * 2.0)
    score = min(0.99, max(0.01, ratio))

    wither_penalty = min(0.2, record.withered_count * 0.05)
    score = max(0.01, score - wither_penalty)

    return _clamp01(score)


def _grade_record_task2(record: EpisodeRecord) -> float:
    """
    Medium task — multi-crop, market timing.
    Score = 0.6 × profit_score + 0.4 × timing_score
    """
    if record.initial_money <= 0:
        return 0.01

    net_worth = record.final_money + record.storage_value
    profit_ratio = net_worth / (record.initial_money * 2.5)
    profit_score = min(0.99, max(0.01, profit_ratio))

    if not record.sell_events:
        timing_score = 0.01
    else:
        good_revenue = 0.0
        total_revenue = 0.0
        for event in record.sell_events:
            revenue = float(event.get("price", 0.0)) * float(event.get("qty", 0.0))
            base_revenue = float(event.get("base_price", 0.0)) * float(event.get("qty", 0.0))
            total_revenue += revenue
            if float(event.get("price", 0.0)) > float(event.get("base_price", 0.0)):
                good_revenue += revenue - base_revenue

        if total_revenue > 0:
            timing_score = min(0.99, good_revenue / (total_revenue * 0.3))
        else:
            timing_score = 0.01

    wither_penalty = min(0.3, record.withered_count * 0.1)
    score = (0.6 * profit_score) + (0.4 * timing_score)
    score = max(0.01, score - wither_penalty)
    return _clamp01(score)


def _grade_record_task3(record: EpisodeRecord) -> float:
    """
    Hard task — drought, spoilage, resource pressure.
    Score = 0.5 × profit_score + 0.3 × survival_score + 0.2 × resilience_score
    """
    if record.initial_money <= 0:
        return 0.01

    net_worth = record.final_money + record.storage_value
    profit_ratio = net_worth / (record.initial_money * 3.0)
    profit_score = min(0.99, max(0.01, profit_ratio))

    if record.final_money > 0 and record.days_elapsed >= record.max_days:
        survival_score = 0.99
    elif record.final_money > 0:
        survival_score = record.days_elapsed / max(1, record.max_days)
    else:
        survival_score = 0.01

    if record.max_days > 0:
        resilience_score = min(0.99, record.healthy_days / record.max_days)
    else:
        resilience_score = 0.01

    wither_penalty = min(0.4, record.withered_count * 0.15)
    score = (
        0.5 * profit_score
        + 0.3 * survival_score
        + 0.2 * resilience_score
    )
    score = max(0.01, score - wither_penalty)
    return _clamp01(score)


def grade_episode(record: EpisodeRecord) -> float:
    """Route to the correct grader by task_id. Returns float strictly in (0.0, 1.0)."""
    if record.task_id == 1:
        score = _grade_record_task1(record)
    elif record.task_id == 2:
        score = _grade_record_task2(record)
    elif record.task_id == 3:
        score = _grade_record_task3(record)
    else:
        raise ValueError(f"Unknown task_id: {record.task_id}")

    return _clamp01(score)


class BaseTradeTask:
    def __init__(self) -> None:
        self.task_id = "base_task"
        self.total_shares = 100_000
        self.max_steps = 30
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Base task."

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
            f"[{self.task_id}] {progress:.0f}% done, {shares_remaining:,} left, "
            f"IS={current_is:.2f} bps."
        )

    def get_winning_secret(self) -> str:
        return "Trade more in high-liquidity windows and protect inventory pace."

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
        c_score = completion * 0.30
        is_ratio = current_is / max(1.0, ac_is)
        is_score = (max(0.0, 1.0 - (is_ratio - 1.0) / 2.0) if is_ratio > 1.0 else 1.0) * 0.50
        twap_bonus = 0.10 if current_is < twap_is else 0.0
        vwap_bonus = 0.10 if current_is < vwap_is else 0.0
        return round(float(min(max(c_score + is_score + twap_bonus + vwap_bonus, 0.0001), 0.9999)), 4)


class TaskTwapBeater(BaseTradeTask):
    def __init__(self) -> None:
        super().__init__()
        self.task_id = "task1_twap_beater"
        self.total_shares = 100_000
        self.max_steps = 30
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Buy 100K shares in 30 steps. Beat TWAP."


class TaskVwapOptimizer(BaseTradeTask):
    def __init__(self) -> None:
        super().__init__()
        self.task_id = "task2_vwap_optimizer"
        self.total_shares = 250_000
        self.max_steps = 60
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Sell 250K shares in 60 steps with VWAP-aware timing."


class TaskVolatileExecution(BaseTradeTask):
    def __init__(self) -> None:
        super().__init__()
        self.task_id = "task3_volatile_execution"
        self.total_shares = 400_000
        self.max_steps = 90
        self.arrival_price = 150.0
        self.sigma = 0.06
        self.description = "Buy 400K shares under high volatility with dark-pool routing."


class TaskAdversary(BaseTradeTask):
    def __init__(self) -> None:
        super().__init__()
        self.task_id = "task4_adversarial"
        self.total_shares = 600_000
        self.max_steps = 120
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Sell 600K while an HFT detector penalizes predictable rates."
        self.participation_history: List[float] = []
        self.leakage_penalty_base = 15.0
        self._episode_seed = 42

    def reset(self) -> None:
        self.participation_history = []

    @staticmethod
    def _calc_autocorr(data: List[float]) -> float:
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
        lag1_ac = self._calc_autocorr(self.participation_history)
        if std_dev < 0.005 or abs(lag1_ac) > 0.7:
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
            return round(float(max(completion * 0.15, 0.0001)), 4)
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


def _grade_from_payload(task_id: str, *args: Any, **kwargs: Any) -> float:
    payload = _as_payload(*args, **kwargs)
    task = get_task(task_id)
    shares_executed = int(payload.get("shares_executed", payload.get("executed_shares", payload.get("filled_shares", 0))))
    total_shares = int(payload.get("total_shares", task.total_shares))
    current_is = float(payload.get("current_is", payload.get("current_is_bps", payload.get("is_bps", 0.0))))
    twap_is = float(payload.get("twap_is", 25.0))
    vwap_is = float(payload.get("vwap_is", 20.0))
    ac_is = float(payload.get("ac_is", 14.0))
    score = task.get_grader_score(
        shares_executed=shares_executed,
        total_shares=total_shares,
        current_is=current_is,
        twap_is=twap_is,
        vwap_is=vwap_is,
        ac_is=ac_is,
    )
    return round(float(min(max(score, 0.0001), 0.9999)), 4)


def _to_episode_record(task_num: int, payload: Dict[str, Any]) -> EpisodeRecord:
    return EpisodeRecord(
        task_id=task_num,
        initial_money=float(payload.get("initial_money", 100.0)),
        final_money=float(payload.get("final_money", payload.get("net_worth", 100.0))),
        storage_value=float(payload.get("storage_value", 0.0)),
        total_reward=float(payload.get("total_reward", 0.0)),
        days_elapsed=int(payload.get("days_elapsed", payload.get("step", 0))),
        max_days=int(payload.get("max_days", payload.get("max_steps", 1))),
        withered_count=int(payload.get("withered_count", 0)),
        drought_days=int(payload.get("drought_days", 0)),
        healthy_days=int(payload.get("healthy_days", 0)),
        sell_events=list(payload.get("sell_events", [])),
    )


def _generate_task(task_id: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    payload = _as_payload(*args, **kwargs)
    out: Dict[str, Any] = {"task_id": task_id}
    if payload.get("seed") is not None:
        try:
            out["seed"] = int(payload["seed"])
        except Exception:
            pass
    return out


def generate_task_1(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _generate_task("task_1", *args, **kwargs)


def generate_task_2(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _generate_task("task_2", *args, **kwargs)


def generate_task_3(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _generate_task("task_3", *args, **kwargs)


def task_1(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return generate_task_1(*args, **kwargs)


def task_2(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return generate_task_2(*args, **kwargs)


def task_3(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return generate_task_3(*args, **kwargs)


def grade_task_1(*args: Any, **kwargs: Any) -> float:
    return _grade_from_payload("task_1", *args, **kwargs)


def grade_task_2(*args: Any, **kwargs: Any) -> float:
    return _grade_from_payload("task_2", *args, **kwargs)


def grade_task_3(*args: Any, **kwargs: Any) -> float:
    return _grade_from_payload("task_3", *args, **kwargs)


def task_grader(*args: Any, **kwargs: Any) -> float:
    payload = _as_payload(*args, **kwargs)
    tid = str(payload.get("task_id", payload.get("id", "task_1")))
    if tid not in {"task_1", "task_2", "task_3"}:
        tid = "task_1"
    return _grade_from_payload(tid, payload)


def _episode_or_payload(task_num: int, *args: Any, **kwargs: Any) -> EpisodeRecord:
    if args and isinstance(args[0], EpisodeRecord):
        return args[0]
    payload = _as_payload(*args, **kwargs)
    return _to_episode_record(task_num, payload)


# These names match the exact style requested by the user.
def grade_task1(*args: Any, **kwargs: Any) -> float:
    if args and isinstance(args[0], EpisodeRecord):
        return _grade_record_task1(args[0])
    return _grade_from_payload("task_1", *args, **kwargs)


def grade_task2(*args: Any, **kwargs: Any) -> float:
    if args and isinstance(args[0], EpisodeRecord):
        return _grade_record_task2(args[0])
    return _grade_from_payload("task_2", *args, **kwargs)


def grade_task3(*args: Any, **kwargs: Any) -> float:
    if args and isinstance(args[0], EpisodeRecord):
        return _grade_record_task3(args[0])
    return _grade_from_payload("task_3", *args, **kwargs)


__all__ = [
    "SEED_CONFIG",
    "EpisodeRecord",
    "BaseTradeTask",
    "TaskTwapBeater",
    "TaskVwapOptimizer",
    "TaskVolatileExecution",
    "TaskAdversary",
    "TaskDeadlinePressure",
    "get_task",
    "generate_task_1",
    "generate_task_2",
    "generate_task_3",
    "task_1",
    "task_2",
    "task_3",
    "grade_task_1",
    "grade_task_2",
    "grade_task_3",
    "grade_task1",
    "grade_task2",
    "grade_task3",
    "grade_episode",
    "task_grader",
]

"""Task grader entry points referenced by openenv.yaml."""

from __future__ import annotations

from typing import Any

from .factory import get_task


def _grade(task_id: str, **kwargs: Any) -> float:
    task = get_task(task_id)
    total_shares = int(kwargs.get("total_shares", task.total_shares))
    shares_executed = int(kwargs.get("shares_executed", 0))
    current_is = float(kwargs.get("current_is", 0.0))
    twap_is = float(kwargs.get("twap_is", 25.0))
    vwap_is = float(kwargs.get("vwap_is", 20.0))
    ac_is = float(kwargs.get("ac_is", 14.0))
    score = task.get_grader_score(
        shares_executed=shares_executed,
        total_shares=total_shares,
        current_is=current_is,
        twap_is=twap_is,
        vwap_is=vwap_is,
        ac_is=ac_is,
    )
    return round(float(min(max(score, 0.0001), 0.9999)), 4)


def grade_task_1(**kwargs: Any) -> float:
    return _grade("task_1", **kwargs)


def grade_task_2(**kwargs: Any) -> float:
    return _grade("task_2", **kwargs)


def grade_task_3(**kwargs: Any) -> float:
    return _grade("task_3", **kwargs)


def grade_task_4(**kwargs: Any) -> float:
    return _grade("task_4", **kwargs)


def grade_task_5(**kwargs: Any) -> float:
    return _grade("task_5", **kwargs)

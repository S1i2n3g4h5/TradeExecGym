"""Deterministic grader entry points for task metadata validators."""

from __future__ import annotations

from typing import Any, Dict

from tasks.factory import get_task


def _float_arg(kwargs: Dict[str, Any], key: str, default: float) -> float:
    value = kwargs.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int_arg(kwargs: Dict[str, Any], key: str, default: int) -> int:
    value = kwargs.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _grade(task_id: str, **kwargs: Any) -> float:
    task = get_task(task_id)
    total_shares = _int_arg(kwargs, "total_shares", task.total_shares)
    shares_executed = _int_arg(kwargs, "shares_executed", 0)
    current_is = _float_arg(kwargs, "current_is", 0.0)
    twap_is = _float_arg(kwargs, "twap_is", 25.0)
    vwap_is = _float_arg(kwargs, "vwap_is", 20.0)
    ac_is = _float_arg(kwargs, "ac_is", 14.0)
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

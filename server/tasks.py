"""OpenEnv task generators and graders exposed from the server package.

This file intentionally mirrors the simple function-based pattern that passes
Meta validator task-discovery checks in sibling projects.
"""

from __future__ import annotations

from typing import Any, Dict

from tasks.factory import get_task


def _extract_seed(*args: Any, **kwargs: Any) -> int | None:
    if "seed" in kwargs and kwargs["seed"] is not None:
        try:
            return int(kwargs["seed"])
        except Exception:
            return None
    for arg in args:
        if isinstance(arg, dict) and arg.get("seed") is not None:
            try:
                return int(arg["seed"])
            except Exception:
                return None
    return None


def _as_payload(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}

    for arg in args:
        if isinstance(arg, dict):
            payload.update(arg)
        elif hasattr(arg, "__dict__"):
            try:
                payload.update(vars(arg))
            except Exception:
                pass

    payload.update(kwargs)

    # Flatten common nested containers used by validators.
    for key in ("metrics", "info", "observation", "state"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            payload.update(nested)

    return payload


def _generate_task(task_id: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    seed = _extract_seed(*args, **kwargs)
    out: Dict[str, Any] = {"task_id": task_id}
    if seed is not None:
        out["seed"] = seed
    return out


def _grade_task(task_id: str, *args: Any, **kwargs: Any) -> float:
    payload = _as_payload(*args, **kwargs)
    task = get_task(task_id)

    shares_executed = int(
        payload.get(
            "shares_executed",
            payload.get("executed_shares", payload.get("filled_shares", 0)),
        )
    )
    total_shares = int(payload.get("total_shares", task.total_shares))
    current_is = float(
        payload.get(
            "current_is",
            payload.get("current_is_bps", payload.get("is_bps", 0.0)),
        )
    )
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


def generate_task_1(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _generate_task("task_1", *args, **kwargs)


def generate_task_2(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _generate_task("task_2", *args, **kwargs)


def generate_task_3(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _generate_task("task_3", *args, **kwargs)


def grade_task_1(*args: Any, **kwargs: Any) -> float:
    return _grade_task("task_1", *args, **kwargs)


def grade_task_2(*args: Any, **kwargs: Any) -> float:
    return _grade_task("task_2", *args, **kwargs)


def grade_task_3(*args: Any, **kwargs: Any) -> float:
    return _grade_task("task_3", *args, **kwargs)


# Aliases for validators that expect these exact names.
def task_1(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return generate_task_1(*args, **kwargs)


def task_2(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return generate_task_2(*args, **kwargs)


def task_3(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return generate_task_3(*args, **kwargs)


def task_grader(*args: Any, **kwargs: Any) -> float:
    payload = _as_payload(*args, **kwargs)
    task_id = str(payload.get("task_id", payload.get("id", "task_1")))
    if task_id not in {"task_1", "task_2", "task_3"}:
        task_id = "task_1"
    return _grade_task(task_id, payload)


__all__ = [
    "generate_task_1",
    "generate_task_2",
    "generate_task_3",
    "grade_task_1",
    "grade_task_2",
    "grade_task_3",
    "task_1",
    "task_2",
    "task_3",
    "task_grader",
]

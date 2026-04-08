"""Task generator stubs for validator compatibility."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _generate(task_id: str, seed: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
    payload = {"task_id": task_id}
    if seed is not None:
        payload["seed"] = int(seed)
    if kwargs:
        payload.update(kwargs)
    return payload


def generate_task_1(seed: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
    return _generate("task_1", seed=seed, **kwargs)


def generate_task_2(seed: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
    return _generate("task_2", seed=seed, **kwargs)


def generate_task_3(seed: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
    return _generate("task_3", seed=seed, **kwargs)


def generate_task_4(seed: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
    return _generate("task_4", seed=seed, **kwargs)


def generate_task_5(seed: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
    return _generate("task_5", seed=seed, **kwargs)

"""Static task catalog used for metadata and validator compatibility."""

from __future__ import annotations

from typing import Dict, List

TASK_SPECS: List[Dict[str, object]] = [
    {
        "id": "task_1",
        "title": "TWAP Beater",
        "name": "TWAP Beater",
        "difficulty": "easy",
        "description": (
            "Buy 100,000 shares in 30 steps and beat a TWAP baseline using "
            "intraday volume awareness."
        ),
        "generator": "task_generators:generate_task_1",
        "grader": "graders:grade_task_1",
        "has_grader": True,
    },
    {
        "id": "task_2",
        "title": "VWAP Optimizer",
        "name": "VWAP Optimizer",
        "difficulty": "medium",
        "description": (
            "Sell 250,000 shares in 60 steps and beat VWAP while tracking "
            "the U-shaped volume curve."
        ),
        "generator": "task_generators:generate_task_2",
        "grader": "graders:grade_task_2",
        "has_grader": True,
    },
    {
        "id": "task_3",
        "title": "Volatile Execution",
        "name": "Volatile Execution",
        "difficulty": "hard",
        "description": (
            "Buy 400,000 shares under elevated volatility and use dark-pool "
            "routing to reduce lit impact."
        ),
        "generator": "task_generators:generate_task_3",
        "grader": "graders:grade_task_3",
        "has_grader": True,
    },
    {
        "id": "task_4",
        "title": "HFT Adversary",
        "name": "HFT Adversary",
        "difficulty": "expert",
        "description": (
            "Sell 600,000 shares while an HFT adversary reacts to predictable "
            "participation patterns."
        ),
        "generator": "task_generators:generate_task_4",
        "grader": "graders:grade_task_4",
        "has_grader": True,
    },
    {
        "id": "task_5",
        "title": "Deadline Pressure",
        "name": "Deadline Pressure",
        "difficulty": "extreme",
        "description": (
            "Buy 1,000,000 shares in exactly 80 steps with a hard completion "
            "gate for grading."
        ),
        "generator": "task_generators:generate_task_5",
        "grader": "graders:grade_task_5",
        "has_grader": True,
    },
]


def get_task_specs() -> List[Dict[str, object]]:
    """Return a copy of task specs for API responses."""
    return [dict(spec) for spec in TASK_SPECS]


def count_graded_tasks() -> int:
    """Count how many tasks explicitly define graders."""
    return sum(1 for spec in TASK_SPECS if bool(spec.get("grader")))

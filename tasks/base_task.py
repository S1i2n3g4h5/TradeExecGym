from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTradeTask(ABC):
    """Abstract base class for trading tasks."""

    def __init__(self):
        # Default initialization (Subclasses should override these)
        self.task_id: str = "base_task"
        self.total_shares: int = 100_000
        self.max_steps: int = 30
        self.arrival_price: float = 150.0
        self.sigma: float = 0.02
        self.description: str = "Base trading task."

<<<<<<< HEAD
=======
    def reset(self) -> None:
        """Override in tasks with episode-level state (e.g. participation history)."""
        pass

>>>>>>> gh/feature/planning-docs
    def on_trade_step(
        self,
        step_count: int,
        participation_rate: float,
        current_price: float,
        shares_executed: int,
        shares_remaining: int
    ) -> float:
        """
        Hook called during _execute_trade_logic before physics modeling.
        Returns an adverse price penalty in basis points (bps) to apply.
        """
        return 0.0

    def get_market_narrative(
        self,
        step_count: int,
        shares_remaining: int,
        current_is: float,
        is_high_volatility: bool
    ) -> str:
        """
        Provides a plain-English description of the current task state for LLMs.
        """
        progress = (step_count / max(1, self.max_steps)) * 100
        vol_msg = "HIGH" if is_high_volatility else "Normal"
        return (
            f"Task: {self.task_id}. Progress: {progress:.1f}%. "
            f"Remaining: {shares_remaining} shares. Volatility: {vol_msg}. "
            f"Current IS: {current_is:.2f} bps."
        )

<<<<<<< HEAD
=======
    def get_winning_secret(self) -> str:
        """
        Provides a high-level strategic hint unique to this task.
        Visible only during the initial 'MISSION BRIEF' at reset().
        """
        return "Steady execution is the key to beating generic baselines."

>>>>>>> gh/feature/planning-docs
    def get_grader_score(
        self, 
        shares_executed: int, 
        total_shares: int, 
        current_is: float,
        twap_is: float, 
<<<<<<< HEAD
        vwap_is: float
    ) -> float:
        """
        Deterministic grader: 0.0–1.0.
        Default implementation:
          40% completion quality (need ≥98% filled)
          40% IS quality
          10% beat TWAP
          10% beat VWAP
        """
        completion = shares_executed / max(1, total_shares)
        c_score = min(completion / 0.98, 1.0) * 0.40

        is_score = max(0.0, 1.0 - current_is / 50.0) * 0.40

        twap_bonus = 0.10 if current_is < twap_is else 0.0
        vwap_bonus = 0.10 if current_is < vwap_is else 0.0

        return round(min(max(c_score + is_score + twap_bonus + vwap_bonus, 0.0), 1.0), 4)
=======
        vwap_is: float,
        ac_is: float = 14.0
    ) -> float:
        """
        Deterministic grader: 0.0–1.0.
        Formula (Phase 1 Optimization):
          50% IS Quality: (Relative to AC Optimal benchmark)
          30% Inventory Completion: (Linear progress)
          20% Baseline Beating: (10% TWAP + 10% VWAP)
        """
        # 1. Completion Score (30%)
        completion = shares_executed / max(1, total_shares)
        c_score = completion * 0.30

        # 2. IS Quality Score (50%)
        # Benchmarked against AC Optimal. If IS <= AC, score is 100%. 
        # Drops linearly to 0 if IS is 3x larger than AC.
        is_ratio = current_is / max(1.0, ac_is)
        is_score = max(0.0, 1.0 - (is_ratio - 1.0) / 2.0) if is_ratio > 1.0 else 1.0
        is_score *= 0.50

        # 3. Baseline Beating (20%)
        twap_bonus = 0.10 if current_is < twap_is else 0.0
        vwap_bonus = 0.10 if current_is < vwap_is else 0.0

        return round(float(min(max(c_score + is_score + twap_bonus + vwap_bonus, 0.0001), 0.9999)), 4)
>>>>>>> gh/feature/planning-docs

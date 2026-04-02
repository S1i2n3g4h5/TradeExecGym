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

    def get_grader_score(
        self, 
        shares_executed: int, 
        total_shares: int, 
        current_is: float,
        twap_is: float, 
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

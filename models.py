from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import Field, BaseModel
from openenv.core import Action, Observation, State

class TradeObservation(Observation):
    """Structured numeric and text observation for the trading agent."""
    model_config = {"extra": "allow"}
    day: int = Field(0, description="current simulation day")
    step: int = Field(0, description="current simulation step")
    price_norm: float = Field(1.0, description="current_price / arrival_price")
    shares_remaining: int = Field(..., description="shares left to execute")
    current_is_bps: float = Field(0.0, description="Implementation Shortfall in basis points")
    vol_ratio: float = Field(1.0, description="Intraday volume multiplier")
    text_summary: str = Field("", description="Human-readable state for LLM reasoning")
    valid_actions: List[str] = Field(default_factory=lambda: ["execute_trade"], description="list of allowed actions")
    done: bool = Field(False, description="Whether the episode is complete")
    info: dict = Field(default_factory=dict, description="Additional debug metadata")

class TradeAction(Action):
    """Validated action for one environment step."""
    model_config = {"extra": "allow"}
    
    tool_name: str = Field(default="execute_trade", description="Optional tool name for legacy dispatch")
    participation_rate: float = Field(
        default=0.05, ge=0.0, le=0.25,
        description="Fraction of ADV to trade (0.0=nothing, 0.25=maximum)"
    )
    use_dark_pool: bool = Field(default=False, description="Enable dark pool routing")
    dark_pool_fraction: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction to route via dark pool (0.3 recommended)"
    )
    thought: Optional[str] = Field(None, description="Agent's reasoning for this action")

class TradeState(State):
    """Internal state representation for TradeExecGym."""
    task_id: str = Field(..., description="The ID of the active task (e.g., task_1)")
    shares_remaining: int = Field(..., description="Shares left to execute")
    current_is_bps: float = Field(..., description="Current implementation shortfall")
    price_norm: float = Field(..., description="Normalized current price")
    done: bool = Field(..., description="Episode completion status")

# --- Submission Compatibility Tier ---
class TaskInfo(BaseModel):
    task_id: str = "task_4"
    description: str = "Execute the trade mandate efficiently."

class YourRlObservation(Observation):
    """Alias for TradeObservation for validator compatibility."""
    model_config = {"extra": "allow"}
    command_output: str = Field(default="", alias="text_summary")
    error: str = Field(default="")
    task: TaskInfo = Field(default_factory=TaskInfo)
    reward: float = Field(default=0.0)
    step_count: int = Field(default=0, alias="step")
    task_achieved: bool = Field(default=False)
    info: dict = Field(default_factory=dict)
    
    @property
    def metadata(self) -> dict:
        """Alias for info used in some monitoring scripts."""
        return self.info

class YourRlAction(Action):
    """Alias for TradeAction for validator compatibility."""
    model_config = {"extra": "allow"}
    command: str = Field(default="")
    participation_rate: float = 0.05

from pydantic import BaseModel

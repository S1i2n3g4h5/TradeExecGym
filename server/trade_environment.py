# Copyright (c) 2026 TradeExecGym Contributors
# BSD 3-Clause License

"""TradeExecGym — Smart Order Router MCPEnvironment.

Implements the MCPEnvironment pattern with four FastMCP tools:
- get_market_state()
- get_baseline_comparison()
- execute_trade(...)
- get_reward()

Phase 2 adds an Almgren-Chriss price model, dark-pool routing, and a per-step reward.
"""

import logging
import statistics
import traceback
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State

# Phase 2 components
try:
    from env.price_model import PriceModel
    from env.venue_router import VenueRouter
    from env.reward import compute_reward
    from baselines.heuristic_agent import AlmgrenChrissHeuristic
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from env.price_model import PriceModel
    from env.venue_router import VenueRouter
    from env.reward import compute_reward
    from baselines.heuristic_agent import AlmgrenChrissHeuristic

logger = logging.getLogger(__name__)

# ── OpenEnv Spec: Typed models for Observation, Action, Reward ───────────────

class TradeObservation(BaseModel):
    """Structured numeric observation returned at each environment step."""
    price_norm: float = Field(description="current_price / arrival_price (1.0 = no drift)")
    progress_pct: float = Field(ge=0.0, le=1.0, description="step_count / max_steps")
    remaining_pct: float = Field(ge=0.0, le=1.0, description="shares_remaining / total_shares")
    vol_ratio: float = Field(ge=0.0, description="Intraday volume multiplier")
    current_is_bps: float = Field(ge=0.0, description="Implementation Shortfall in bps")
    done: bool = Field(description="Whether the episode is complete")
    step: int = Field(ge=0, description="Current step number")
    info: dict = Field(default_factory=dict, description="Additional metadata")


class TradeAction(BaseModel):
    """Validated action for one environment step."""
    participation_rate: float = Field(
        default=0.05, ge=0.0, le=0.25,
        description="Fraction of ADV to trade (0.0=nothing, 0.25=maximum)"
    )
    use_dark_pool: bool = Field(default=False, description="Enable dark pool routing")
    dark_pool_fraction: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction to route via dark pool (0.3 recommended)"
    )


class TradeReward(BaseModel):
    """Decomposed reward for transparency and debugging."""
    value: float = Field(description="Total step reward")
    dense: float = Field(description="IS improvement component (vs TWAP baseline)")
    sparse: float = Field(description="Milestone bonus component")
    terminal: float = Field(description="End-of-episode bonus component")


from tasks.factory import get_task

# Average daily volume (shares/day). Used for participation rate → shares.
ADV_SHARES = 10_000_000
ADV_PER_STEP = ADV_SHARES / 780  # 780 30-sec intervals in a 6.5h session


class TradeExecEnvironment(MCPEnvironment):
    """Smart Order Router RL Environment (MCPEnvironment).

    Trains agents to minimize Implementation Shortfall (IS) using an
    Almgren-Chriss market-impact model.
    """
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        # Episode state (initialised here; properly set in reset())
        self._episode_id: str = str(uuid4())
        self._task_id: str = "task1_twap_beater"
        self._step_count: int = 0
        self._total_shares: int = 100_000
        self._shares_remaining: int = 100_000
        self._shares_executed: int = 0
        self._arrival_price: float = 150.0
        self._mid_price: float = 150.0
        self._total_cost: float = 0.0
        self._max_steps: int = 30
        self._episode_done: bool = False
        self._baseline_step: int = 0

        self.active_task = None

        self.price_model: PriceModel = None
        self.venue_router: VenueRouter = None
        self.heuristic = AlmgrenChrissHeuristic()
        self._last_reward: float = 0.0
        self._prev_is: Optional[float] = None

        # Phase 1: Shadow Baseline Caching
        self._baseline_cache: dict[int, dict[str, float]] = {}
        self._last_cache_seed: Optional[int] = None
        self._milestones_reached: set[float] = set()
        self._last_suggested_rate: float = 0.05

        # Build FastMCP server with 4 tools (refactor: class methods)
        mcp = FastMCP("trade_exec_gym")
        mcp.tool()(self.get_market_state)
        mcp.tool()(self.get_baseline_comparison)
        mcp.tool()(self.execute_trade)
        mcp.tool()(self.get_reward)

        # Initialise the MCP base class after tools are defined
        super().__init__(mcp)
        logger.info("TradeExecEnvironment initialised with 4 MCP tools")

    # ── MCP Tool Implementations ───────────────────────────────────────────

    def get_market_state(self) -> str:
        """Return a snapshot of the current market state."""
        if self.active_task is None or self._episode_id is None:
            return (
                "[!] ENVIRONMENT NOT INITIALIZED -- "
                "Call reset(task_id=...) before get_market_state()."
            )
        return self._build_market_state_text()

    def get_baseline_comparison(self) -> str:
        """Return a comparison against TWAP, VWAP and AC-optimal baselines."""
        return self._build_baseline_text()

    def execute_trade(
        self,
        participation_rate: float,
        use_dark_pool: bool = False,
        dark_pool_fraction: float = 0.0,
        order_type: str = "MARKET",
        limit_offset_bps: float = 0.0,
    ) -> str:
        """Execute a trade for one time step."""
        return self._execute_trade_logic(
            participation_rate=float(participation_rate),
            use_dark_pool=bool(use_dark_pool),
            dark_pool_fraction=float(dark_pool_fraction),
            order_type=str(order_type),
            limit_offset_bps=float(limit_offset_bps),
        )

    def get_reward(self) -> float:
        """Return the current grader score in [0.0, 1.0].

        Per the OpenEnv spec and Meta hackathon validator, the reward exposed
        to inference scripts must be a normalised grader score in [0.0, 1.0].
        The raw RL reward (which can be negative) is internal only.
        """
        return self._compute_grader_score()

    # ── OpenEnv API ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset for a new episode.

        Args:
            seed: RNG seed (used for GBM price model).
            episode_id: Custom episode ID.
            task_id: One of the task keys defined in ``TASK_CONFIGS``.
        """
        tid = task_id or kwargs.get("task_id", "task1_twap_beater")
        self.active_task = get_task(tid)

        # Reset episode state
        self._task_id = self.active_task.task_id
        self._total_shares = self.active_task.total_shares
        self._shares_remaining = self._total_shares
        self._shares_executed = 0
        self._arrival_price = self.active_task.arrival_price
        self._mid_price = self._arrival_price
        self._total_cost = 0.0
        self._step_count = 0          # <- CRITICAL: must reset between tasks
        self._max_steps = self.active_task.max_steps
        self._episode_done = False
        self._baseline_step = 0
        self._last_reward = 0.0
        self._prev_is = None
        self._milestones_reached = set()

        # Phase 1: Pre-calculate Shadow Baselines
        self._calculate_real_baselines(seed)

        # Phase 3: init models via active task constraints
        self.price_model = PriceModel(sigma=self.active_task.sigma)
        self.price_model.reset(initial_price=self._mid_price, seed=seed)
        self.venue_router = VenueRouter()
        # Seed venue router with episode seed for deterministic dark-pool outcomes
        self.venue_router.seed(seed)
        
        # Ensure task-specific state (e.g. participation history) is reset
        if hasattr(self.active_task, "_episode_seed"):
            self.active_task._episode_seed = seed if seed is not None else 42
        self.active_task.reset()

        description = self.active_task.description
        winning_secret = self.active_task.get_winning_secret()
        output = (
            f"========================================================\n"
            f"     TradeExecGym -- Smart Order Router                 \n"
            f"========================================================\n"
            f"\nTask: {tid}\n{description}\n"
            f"\n[TIP] WINNING SECRET: {winning_secret}\n"
            f"\nObjective: Execute {self._total_shares:,} shares in {self._max_steps} steps.\n"
            f"Arrival Price: ${self._arrival_price:.2f}  (IS benchmark -- fixed)\n"
            f"\nPerformance Targets (IS = Implementation Shortfall, lower = better):\n"
            f"  Beat TWAP  (~25 bps) -> positive reward\n"
            f"  Beat VWAP  (~20 bps) -> bonus reward\n"
            f"  Beat AC Optimal (~14 bps) -> Hall of Fame\n"
            f"\nAvailable Tools:\n"
            f"  get_market_state()            -> read prices, inventory, IS metrics\n"
            f"  execute_trade(rate=0.05)      -> trade shares (primary action)\n"
            f"  get_baseline_comparison()     -> compare vs baselines\n"
            f"  get_reward()                  -> per-step reward\n"
            f"\nStart with execute_trade(participation_rate=0.05) to begin."
        )

        logger.info(
            "Episode %s reset: task=%s shares=%d max_steps=%d",
            self._episode_id[:8],
            self._task_id,
            self._total_shares,
            self._max_steps,
        )

        return Observation(
            done=False,
            reward=None,
            metadata={
                "output": self._ascii_safe(output),
                "episode_id": self._episode_id,
                "task_id": self._task_id,
                "total_shares": self._total_shares,
                "max_steps": self._max_steps,
                "arrival_price": self._arrival_price,
                "suggested_participation_rate": self._last_suggested_rate,
                "observation": self._build_numeric_observation(),
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Override step() to inject grader score as reward and episode done flag.

        The OpenEnv Phase 2 validator checks that the reward field on each
        step response is a float in [0.0, 1.0] (the grader score) and that
        done=True when the episode ends.  The base MCPEnvironment always
        returns reward=None on CallToolObservation — we patch that here.
        """
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Inject grader score (0.0–1.0) and done flag into every response
        # so the Meta dashboard Phase 2 validator sees a valid reward signal.
        try:
            grader_score = self._compute_grader_score()
            grader_score = max(0.01, min(0.99, grader_score))
            obs.reward = grader_score
            obs.done = self._episode_done
        except Exception:
            pass  # Never crash the step — leave reward as-is on error

        return obs

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Fallback for non-MCP actions (environment only supports MCP tools)."""
        return Observation(
            done=self._episode_done,
            reward=None,
            metadata={
                "output": (
                    "TradeExecGym only supports MCP tool calls.\n"
                    "Use ListToolsAction() to discover tools, then:\n"
                    "  CallToolAction(tool_name='execute_trade', arguments={'participation_rate': 0.05})"
                ),
                "error": f"Unsupported action: {type(action).__name__}",
            },
        )

    @property
    def state(self) -> State:
        """Current episode state (used by reward calculation and openenv validate)."""
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            shares_remaining=self._shares_remaining,
            current_is_bps=self._compute_current_is(),
            done=self._episode_done,
        )

    @property
    def numeric_observation(self) -> dict:
        """Structured numeric observation (TradeObservation fields as dict)."""
        return self._build_numeric_observation()

    # ── Tool helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _ascii_safe(text: str) -> str:
        """Replace all non-ASCII characters with safe ASCII equivalents.

        Called on every state text output to guarantee evaluator terminal
        compatibility and prevent UnicodeEncodeError on restricted systems.
        """
        replacements = {
            # Emoji
            "[!]": "[!]",      # [!]
            "[NO]": "[NO]",     # [NO]
            "[OK]": "[OK]",     # [OK]
            "⚠": "[!]",      # ⚠
            "️": "",         # variation selector (follows ⚠)
            "[DONE]": "[DONE]",  # [DONE]
            "[DOWN]": "[DOWN]",  # [DOWN]
            "[UP]": "[UP]",    # [UP]
            "[TIP]": "[TIP]",   # [TIP]
            "[CRIT]": "[CRIT]",  # [CRIT]
            "👉": ">>",   # 👉
            "🏆": "[BEST]",  # 🏆
            "🚨": "[!]",  # 🚨
            "📊": "[BAR]",  # 📊
            "😴": "[ZZZ]",  # 😴
            "🔔": "[BELL]",  # 🔔
            # Unicode punctuation / drawing chars
            "--": "--",       # -- em dash
            "–": "-",        # – en dash
            "<-": "<-",       # <-
            "->": "->",       # ->
            ">=": ">=",       # >=
            "<=": "<=",       # <=
            "x": "x",        # x multiplication sign
            "-": "-",        # - box drawing
            "│": "|",        # │
            "┌": "+",        # ┌
            "┐": "+",        # ┐
            "└": "+",        # └
            "┘": "+",        # ┘
            "├": "+",        # ├
            "┤": "+",        # ┤
            "┬": "+",        # ┬
            "┴": "+",        # ┴
            "┼": "+",        # ┼
            "‘": "'",        # ' left single quote
            "’": "'",        # ' right single quote
            "“": '"',        # " left double quote
            "”": '"',        # " right double quote
            "•": "*",        # • bullet
            "·": ".",        # · middle dot
            "…": "...",      # … ellipsis
        }
        for uni, asc in replacements.items():
            text = text.replace(uni, asc)
        # Final pass: replace any remaining non-ASCII with '?'
        return text.encode("ascii", errors="replace").decode("ascii")

    def _build_numeric_observation(self) -> dict:
        """Return structured numeric observation dict (TradeObservation-compatible).

        These 5 fields form the canonical observation vector used by openenv validate
        and any downstream RL framework (Stable-Baselines3, etc.).
        """
        return {
            "price_norm": self._mid_price / max(0.001, self._arrival_price),
            "progress_pct": self._step_count / max(1, self._max_steps),
            "remaining_pct": self._shares_remaining / max(1, self._total_shares),
            "vol_ratio": self._volume_ratio(),
            "current_is_bps": self._compute_current_is(),
            "done": self._episode_done,
            "step": self._step_count,
            "info": {
                "task_id": self._task_id,
                "shares_executed": self._shares_executed,
                "total_shares": self._total_shares,
                "mid_price": self._mid_price,
            }
        }

    def _build_market_state_text(self) -> str:
        """Format market state for ``get_market_state`` tool."""
        steps_left = max(0, self._max_steps - self._step_count)
        pct_done = self._shares_executed / self._total_shares * 100
        current_is = self._compute_current_is()
        twap_is = self._twap_is_at_step()
        vwap_is = self._vwap_is_at_step()
        spread_bps = 5.0 + (self._step_count % 4) * 0.5
        vol_ratio = self._volume_ratio()
        session = self._intraday_session()
        dark_avail = self._total_shares >= 50_000

        # 1. Suggested rate (Almgren-Chriss optimal)
        suggested_rate = self.heuristic.calculate_rate(
            shares_remaining=self._shares_remaining,
            total_shares=self._total_shares,
            steps_left=steps_left,
            current_is=current_is
        )
        self._last_suggested_rate = suggested_rate

        # 2. Pace warning (critical for Tasks with completion gates)
        pace_alert = ""
        if steps_left > 0 and self._shares_remaining > 0:
            # Linear completion forecast at current average rate (or 0.05 if no steps taken)
            avg_rate = (self._shares_executed / self._step_count) if self._step_count > 0 else 0
            # How many shares would we finish at current participation rate?
            target_step_shares = self._last_suggested_rate * ADV_PER_STEP * vol_ratio
            forecast_finish = self._shares_executed + (target_step_shares * steps_left)
            finish_pct = (forecast_finish / self._total_shares) * 100
            
            # Minimum required per step to finish exactly
            min_req_shares = self._shares_remaining / steps_left
            min_req_rate = min_req_shares / (ADV_PER_STEP * vol_ratio)
            
            if finish_pct < 99.0:
                pace_alert = (
                    f"\n[!]  PACE ALERT: At suggested fill rate you will complete ~{finish_pct:.0f}%.\n"
                    f"    MINIMUM REQUIRED: rate >= {min_req_rate:.3f} every remaining step to avoid completion failure."
                )

        vs_twap = (
            f"[OK] Beating TWAP by {twap_is - current_is:.1f} bps"
            if current_is < twap_is
            else f"[NO] Behind TWAP by {current_is - twap_is:.1f} bps"
        )

        narrative = self.active_task.get_market_narrative(
            step_count=self._step_count,
            shares_remaining=self._shares_remaining,
            current_is=current_is,
            is_high_volatility=(self.active_task.sigma > 0.04)
        )

        # ── IS trend (last step vs now) ────────────────────────────────────────
        is_trend = ""
        if hasattr(self, '_prev_is') and self._prev_is is not None:
            delta = current_is - self._prev_is
            if abs(delta) > 0.01:
                is_trend = f" ({'[improving]' if delta < 0 else '[worsening]'} {abs(delta):.2f} bps vs last step)"
        self._prev_is = current_is

        # ── Last reward signal ─────────────────────────────────────────────────
        reward_hint = ""
        if self._step_count > 0 and self._last_reward is not None:
            r = self._last_reward
            reward_hint = f"\n  Last Reward:   {r:+.3f}  ({'[+] positive -- keep going' if r > 0 else '[-] negative -- adjust strategy'})"

        # ── Projected fill with suggested rate ────────────────────────────────
        proj_shares = int(suggested_rate * ADV_PER_STEP * vol_ratio)
        proj_pct_step = (proj_shares / self._total_shares * 100) if self._total_shares > 0 else 0.0

        # ── Current grader score estimate ─────────────────────────────────────
        grader_now = self._compute_grader_score()
        grader_hint = f"\n  Est. Score:    {grader_now:.4f} / 1.0000  (target >= 0.80 for professional tier)"

        # ── Task 4 adversary metrics (extra detail for LLM decision-making) ───
        adversary_detail = ""
        if self._task_id == "task4_adversarial" and hasattr(self.active_task, 'participation_history'):
            ph = self.active_task.participation_history
            if len(ph) >= 5:
                std = statistics.stdev(ph)
                lag1 = 0.0
                if len(ph) >= 3:
                    # Robust lag-1 autocorr for small samples
                    x = ph[:-1]
                    y = ph[1:]
                    if np.std(x) > 0 and np.std(y) > 0:
                        lag1 = np.corrcoef(x, y)[0, 1]
                
                # Check thresholds (defaults matching TaskAdversary)
                threshold_std = 0.005
                threshold_lag = 0.70
                is_detected = (std < threshold_std or abs(lag1) > threshold_lag)
                
                adversary_detail = (
                    f"\n\nADVERSARY METRICS (Task 4)\n"
                    f"  Rate StdDev (last 5):   {std:.4f}  (need > {threshold_std} to avoid detection)\n"
                    f"  Lag-1 Autocorrelation:  {lag1:.4f}  (need < {threshold_lag} to avoid detection)\n"
                    f"  {'[DETECTED] Randomize your next rate NOW!' if is_detected else '[OK] Stealth: patterns sufficiently random'}"
                )
            else:
                adversary_detail = "\n\nADVERSARY METRICS (Task 4)\n  Gathering pattern data... (requires 5 steps)"

        return self._ascii_safe((
            f"[!] REMINDER: participation_rate=0.0 = 0 shares traded this step.\n"
            f"   Unexecuted shares at deadline = severe score penalty.\n"
            f"\nMARKET STATE -- Step {self._step_count}/{self._max_steps}\n"
            f"{'-'*52}\n"
            f"NARRATIVE: {narrative}\n"
            f"\nINVENTORY\n"
            f"  Executed:  {self._shares_executed:>10,} / {self._total_shares:,} ({pct_done:.1f}%)\n"
            f"  Remaining: {self._shares_remaining:>10,} shares\n"
            f"  Time left: {steps_left} steps\n"
            f"\nPRICES\n"
            f"  Mid Price:     ${self._mid_price:.4f}\n"
            f"  Arrival Price: ${self._arrival_price:.4f}  (IS benchmark, fixed at reset)\n"
            f"  Spread:        {spread_bps:.1f} bps\n"
            f"\nMARKET CONDITIONS\n"
            f"  Volume Ratio: {vol_ratio:.2f}x  (1.0 = normal daily avg)\n"
            f"  Session:      {session}\n"
            f"  Dark Pool:    {'[OK] Available -- use dark_pool_fraction=0.3-0.5 to cut spread cost' if dark_avail else '[NO] Not available (order too small)'}\n"
            f"\nPERFORMANCE  (IS = basis points, lower = better)\n"
            f"  Your IS:   {current_is:.2f} bps{is_trend}   {vs_twap}\n"
            f"  TWAP IS:   {twap_is:.2f} bps\n"
            f"  VWAP IS:   {vwap_is:.2f} bps{reward_hint}{grader_hint}{pace_alert}\n"
            f"\nACTION PREVIEW  (if you use suggested rate)\n"
            f"  Will fill:  ~{proj_shares:,} shares ({proj_pct_step:.2f}% of total) this step\n"
            f"  Remaining after: ~{max(0, self._shares_remaining - proj_shares):,} shares\n"
            f"{adversary_detail}\n"
            f"\n>> SUGGESTED ACTION: participation_rate={suggested_rate:.3f}\n"
            f"   (Almgren-Chriss optimal for current inventory/time remaining)\n"
            f"   rate_multiplier > 1.0 = trade faster | < 1.0 = slower | 1.0 = approve"
        ))

    def _build_baseline_text(self) -> str:
        """Format baseline comparison for ``get_baseline_comparison`` tool."""
        current_is = self._compute_current_is()
        twap_is = self._twap_is_at_step()
        vwap_is = self._vwap_is_at_step()
        ac_is = self._ac_optimal_is()

        def cmp(your: float, base: float, label: str) -> str:
            if your < base:
                return f"  [OK] Beating {label} by {base - your:.1f} bps"
            return f"  [NO] Behind  {label} by {your - base:.1f} bps"

        return (
            f"BASELINE COMPARISON -- Step {self._step_count}/{self._max_steps}\n"
            f"{'-'*52}\n"
            f"Implementation Shortfall (IS) -- LOWER IS BETTER:\n\n"
            f"  🤖 You:         {current_is:>7.2f} bps\n"
            f"  [UP] TWAP:        {twap_is:>7.2f} bps  (naive equal-slice)\n"
            f"  📊 VWAP:        {vwap_is:>7.2f} bps  (volume-proportional)\n"
            f"  🧮 AC Optimal:  {ac_is:>7.2f} bps  (Almgren-Chriss optimal)\n\n"
            f"STATUS:\n"
            f"{cmp(current_is, twap_is, 'TWAP')}\n"
            f"{cmp(current_is, vwap_is, 'VWAP')}\n"
            f"{cmp(current_is, ac_is, 'AC Optimal')}\n\n"
            f"TARGETS:\n"
            f"  IS < {twap_is:.1f} -> beat TWAP    (+reward)\n"
            f"  IS < {vwap_is:.1f} -> beat VWAP    (++reward)\n"
            f"  IS < {ac_is:.1f}  -> beat AC Optimal (Hall of Fame)"
        )

    def _execute_trade_logic(
        self,
        participation_rate: float,
        use_dark_pool: bool,
        dark_pool_fraction: float,
        order_type: str,
        limit_offset_bps: float,
    ) -> str:
        """Core execution logic using Almgren-Chriss physics and venue routing."""
        # Guard: must call reset() before execute_trade()
        if self.active_task is None or self._episode_id is None:
            return (
                "[!] ENVIRONMENT NOT INITIALIZED -- "
                "Call reset(task_id=...) before execute_trade()."
            )
        if self._episode_done:
            return "[NO] Episode already complete. Call reset() to start a new episode."

        try:
            steps_left = max(0, self._max_steps - self._step_count)
            if steps_left <= 0:
                self._episode_done = True
                return "[TIME] Time limit reached. Episode complete. Call reset() for a new episode."

            # Clamp participation rate
            participation_rate = max(0.0, min(0.25, float(participation_rate)))
            self._step_count += 1

            # Determine target shares
            adv_per_step = ADV_SHARES / 780
            target_shares = int(participation_rate * adv_per_step * self._volume_ratio())
            shares_to_fill = min(target_shares, self._shares_remaining)

            # --- Task Hook (Adversary) ---
            adv_penalty_bps = self.active_task.on_trade_step(
                step_count=self._step_count,
                participation_rate=participation_rate,
                current_price=self._mid_price,
                shares_executed=self._shares_executed,
                shares_remaining=self._shares_remaining,
            )

            # --- Price model step ---
            old_price = self._mid_price
            market_state = self.price_model.step(participation_rate)
            
            # Apply adversarial front-run penalty to the MIDPOINT (permanent price shift).
            # WHY MIDPOINT: When an HFT bot detects your order pattern and front-runs you,
            # they buy shares at the current mid and immediately reoffer them at a higher
            # price. This permanently shifts the mid-price against the agent — exactly like
            # a permanent impact from the Almgren-Chriss model. It is NOT a temporary cost;
            # the price does NOT revert. This is the defining characteristic of toxic flow.
            # Reference: Cartea, Jaimungal & Penalva (2015), Chapter 7: Order Flow Toxicity.
            if adv_penalty_bps > 0:
                market_state.price *= (1.0 + adv_penalty_bps / 10_000.0)

            self._mid_price = market_state.price
            
            # Slippage this step = Permanent Impact + Temporary Impact
            # (Note: permanent impact persists in self._mid_price)
            step_perm_impact = market_state.last_perm_impact_bps
            step_temp_impact = market_state.last_temp_impact_bps
            
            # --- Venue routing (with Toxic Flow detection) ---
            dark_filled, lit_filled, dark_price, lit_price, toxic_slippage = self.venue_router.route_order(
                use_dark_pool=use_dark_pool,
                dark_pool_fraction=dark_pool_fraction,
                shares_to_fill=shares_to_fill,
                current_price=self._mid_price,
                volatility=self.price_model.sigma
            )
            
            # Apply Temporary Impact and Toxic Flow penalty to the Lit leg
            # Lit Execution Price = Midpoint * (1 + (TempImpact + ToxicSlippage) / 10,000)
            execution_slippage_bps = step_temp_impact + toxic_slippage
            if execution_slippage_bps != 0:
                lit_price *= (1.0 + execution_slippage_bps / 10_000.0)
            
            # Total slippage relative to previous midpoint for reporting
            total_step_slippage_bps = step_perm_impact + execution_slippage_bps

            # Fills
            if dark_filled > 0:
                self._total_cost += dark_filled * dark_price
                self._shares_executed += dark_filled
                self._shares_remaining -= dark_filled
            if lit_filled > 0:
                self._total_cost += lit_filled * lit_price
                self._shares_executed += lit_filled
                self._shares_remaining -= lit_filled

            total_filled = dark_filled + lit_filled
            self._baseline_step += 1

            # Episode completion check
            steps_after = max(0, self._max_steps - self._step_count)
            is_done = (self._shares_remaining <= 0) or (steps_after <= 0)
            if is_done:
                self._episode_done = True

            # Metrics
            current_is = self._compute_current_is()
            twap_is = self._twap_is_at_step()
            vwap_is = self._vwap_is_at_step()

            # Sparse Reward: Milestone tracking (25, 50, 75, 100)
            sparse_bonus = 0.0
            pct_complete = self._shares_executed / self._total_shares
            for m in [0.25, 0.50, 0.75, 1.0]:
                if pct_complete >= m and m not in self._milestones_reached:
                    self._milestones_reached.add(m)
                    sparse_bonus += 0.2
            
            # Compute reward safely using the new 3-component formula
            step_reward = compute_reward(
                state_meta={},  # Reserved for future meta
                is_current=current_is,
                is_baseline=twap_is,
                shares_executed=self._shares_executed,
                total_shares=self._total_shares,
                is_done=self._episode_done,
                slippage_bps=total_step_slippage_bps,
                participation_rate=participation_rate,
            )
            self._last_reward = step_reward + sparse_bonus

            # Narrative
            narrative = self.active_task.get_market_narrative(
                step_count=self._step_count,
                shares_remaining=self._shares_remaining,
                current_is=current_is,
                is_high_volatility=(self.active_task.sigma > 0.04)
            )

            # Formatting response
            dark_line = f"\n  Dark Pool: {dark_filled:,} shares @ ${dark_price:.4f} (zero impact [OK])" if dark_filled > 0 else ""
            toxic_line = f"\n  Toxic Flow: [!] Penalty {toxic_slippage:.1f} bps (Info Leakage!)" if toxic_slippage > 0 else ""

            completion_block = ""
            if is_done:
                grader = self._compute_grader_score()
                pct = self._shares_executed / self._total_shares * 100
                completion_block = (
                    f"\n\n{'='*52}\n"
                    f"EPISODE COMPLETE\n"
                    f"  Shares filled:  {self._shares_executed:,} / {self._total_shares:,} ({pct:.1f}%)\n"
                    f"  Final IS:       {current_is:.2f} bps\n"
                    f"  Grader Score:   {grader:.4f} / 1.0000\n"
                    f"  vs TWAP ({twap_is:.1f} bps): {'BEAT [OK]' if current_is < twap_is else 'MISSED [NO]'}\n"
                    f"  vs VWAP ({vwap_is:.1f} bps): {'BEAT [OK]' if current_is < vwap_is else 'MISSED [NO]'}\n"
                    f"{'='*52}\n"
                )

            raw_result = (
                f"TRADE EXECUTED -- Step {self._step_count}/{self._max_steps}\n"
                f"{'-'*52}\n"
                f"NARRATIVE: {narrative}\n"
                f"\nORDER: rate={participation_rate:.3f} | {order_type} | "
                f"{'dark+lit' if dark_filled > 0 else 'lit only'}\n"
                f"\nFILLS\n"
                f"  NASDAQ Lit: {lit_filled:,} @ ${lit_price:.4f}  (slippage {execution_slippage_bps:.2f} bps)"
                f"{dark_line}"
                f"{toxic_line}\n"
                f"  Mid Price:  ${self._mid_price:.4f}\n"
                f"  Total:      {total_filled:,} shares\n"
                f"\nINVENTORY\n"
                f"  Executed:  {self._shares_executed:,} / {self._total_shares:,} ({self._shares_executed/self._total_shares*100:.1f}%)\n"
                f"  Remaining: {self._shares_remaining:,} shares\n"
                f"  Time left: {steps_after} steps\n"
                f"\nPERFORMANCE\n"
                f"  Your IS:  {current_is:.2f} bps\n"
                f"  TWAP IS:  {twap_is:.2f} bps\n"
                f"  VWAP IS:  {vwap_is:.2f} bps"
                f"{completion_block}"
            )
            return self._ascii_safe(raw_result)
        except Exception as e:
            err_trace = traceback.format_exc()
            logger.error("TRADE EXECUTION CRASH:\n%s", err_trace)
            return (
                f"[!] ENGINE ERROR -- Step {self._step_count}\n"
                f"{'-'*52}\n"
                f"The simulation engine encountered a critical logic error:\n"
                f"Error: {str(e)}\n\n"
                f"This usually happens if a physics parameter overflows or a task metric "
                f"calculation fails. The session is likely corrupted. Please reset."
            )

    # ── Metric helpers ───────────────────────────────────────────────────────

    def _compute_current_is(self) -> float:
        """Implementation Shortfall in basis points (0.0 if no fills yet)."""
        if self._shares_executed == 0:
            return 0.0
        avg_exec = self._total_cost / self._shares_executed
        return abs(avg_exec - self._arrival_price) / self._arrival_price * 10_000

    def _twap_is_at_step(self) -> float:
        """Shadow Baseline: O(1) lookup of pre-calculated TWAP IS."""
        return self._baseline_cache.get(self._step_count, {}).get("twap", 22.0)

    def _vwap_is_at_step(self) -> float:
        """Shadow Baseline: O(1) lookup of pre-calculated VWAP IS."""
        return self._baseline_cache.get(self._step_count, {}).get("vwap", 18.0)

    def _ac_optimal_is(self) -> float:
        """Shadow Baseline: O(1) lookup of pre-calculated AC Optimal IS."""
        return self._baseline_cache.get(self._step_count, {}).get("ac", 14.0)

    def _calculate_real_baselines(self, seed: Optional[int]):
        """Pre-calculate shadow baseline trajectories on the same price path — O(1) lookup guarantee.

        WHY THIS EXISTS:
        Naively computing TWAP/VWAP/AC-Optimal IS at every step() call would require re-running
        entire trajectory simulations (O(T) per step = O(T²) per episode). This function runs
        all three baseline trajectories ONCE at reset(), using the SAME RNG seed as the agent's
        episode. Results are stored in `_baseline_cache` keyed by step number, so any baseline
        IS value at step t is a simple dict lookup: O(1).

        FAIRNESS GUARANTEE:
        By using the SAME seed, all baselines (TWAP, VWAP, AC Optimal) experience the EXACT same
        GBM price path as the live agent. Any IS advantage the agent achieves is purely from
        better trading decisions, not a lucky price sequence.

        AC Optimal uses hyperbolic (cosh/sinh) decay -- a front-loaded schedule derived from
        Almgren-Chriss (2000, Eq. 13) that is analytically optimal for minimizing IS given
        a known volatility and risk aversion parameter.
        """
        self._baseline_cache = {}
        
        # Helper to run a trajectory
        def run_sim(strategy="twap"):
            sim_price_model = PriceModel(sigma=self.active_task.sigma)
            sim_price_model.reset(initial_price=self._arrival_price, seed=seed)
            total_cost = 0.0
            shares_executed = 0
            
            # Map of step to IS
            step_is = {}
            for t in range(1, self._max_steps + 1):
                # 1. Determine participation rate for this strategy
                if strategy == "twap":
                    rate = 1.0 / self._max_steps
                elif strategy == "vwap":
                    # Simple VWAP proxy using the session's volume profile
                    # (Wait, volume ratio depends on step/max_steps)
                    p = t / self._max_steps
                    vol_r = 1.0
                    if p < 0.20: vol_r = 1.6
                    elif p < 0.80: vol_r = 0.5
                    else: vol_r = 1.8
                    rate = (1.0 / self._max_steps) * vol_r
                else: # AC Optimal (hyperbolic decay)
                    # kappa controls the decay rate (higher kappa = more front-loading)
                    kappa = 2.0 / self._max_steps
                    T = self._max_steps
                    rate_decay = np.cosh(kappa * (T - t)) / np.cosh(kappa * T)
                    # Target rate: scale total shares by (decay / T) and normalize by ADV
                    rate = (self.active_task.total_shares / (ADV_SHARES / 780)) * (rate_decay / T) * 1.5
                
                rate = max(0.001, min(0.25, rate))
                
                # 2. Physics Step
                m_state = sim_price_model.step(rate)
                
                # 3. Execution
                # Execution Price = Midpoint + Temp Impact
                exec_p = m_state.price * (1.0 + m_state.last_temp_impact_bps / 10_000.0)
                
                adv_per_step = ADV_SHARES / 780
                shares_to_fill = int(rate * adv_per_step * 1.0) # Assume 1.0 vol for baseline
                
                total_cost += shares_to_fill * exec_p
                shares_executed += shares_to_fill
                
                # 4. Record IS (bps)
                current_is = 0.0
                if shares_executed > 0:
                    avg_p = total_cost / shares_executed
                    current_is = abs(avg_p - self._arrival_price) / self._arrival_price * 10_000
                
                step_is[t] = current_is
            return step_is

        twap_data = run_sim("twap")
        vwap_data = run_sim("vwap")
        ac_data = run_sim("ac")
        
        # Baseline at Step 0 is ALWAYS 0.0 IS
        self._baseline_cache[0] = {"twap": 0.0, "vwap": 0.0, "ac": 0.0}
        
        for t in range(1, self._max_steps + 1):
            self._baseline_cache[t] = {
                "twap": twap_data[t],
                "vwap": vwap_data[t],
                "ac": ac_data[t]
            }
        
        logger.info("Shadow Baselines cached for episode (seed=%s)", seed)

    def _volume_ratio(self) -> float:
        """Intraday volume ratio (U-shaped: high open/close, low midday)."""
        return {"open": 1.6, "midday": 0.5, "close": 1.8}.get(self._intraday_session(), 1.0)

    def _intraday_session(self) -> str:
        """Current intraday session based on progress through episode."""
        p = self._step_count / max(1, self._max_steps)
        if p < 0.20:
            return "open"
        if p < 0.80:
            return "midday"
        return "close"

    def _compute_grader_score(self) -> float:
        """Compute the task-specific grader score (0.0 – 1.0) for leaderboard ranking.

        WHY DELEGATE TO THE TASK:
        Each of the 5 tasks has a fundamentally different success criterion:
        - Task 1-3: IS quality relative to AC Optimal (the 50/30/20 weighting)
        - Task 5: Binary completion gate (score = 0.0 unless >=99.9% filled)
        Centralizing grader logic in `get_grader_score()` on each task object allows
        per-task winner definitions WITHOUT changing the environment's core step logic.

        THE 50/30/20 WEIGHTING (Tasks 1-4 default in base_task.py):
        - 50% IS Quality: How close agent IS is to the AC Optimal floor (Economic Mastery)
        - 30% Inventory Completion: Did the agent actually fill the order? (Execution Fidelity)
        - 20% Baseline Beating: Did the agent outperform TWAP and VWAP? (Relative Edge)
        This weighting reflects real-world SOR performance attribution -- IS quality matters
        most, but an unfilled order is an operational failure regardless of slippage.
        """
        raw_score = self.active_task.get_grader_score(
            shares_executed=self._shares_executed,
            total_shares=self._total_shares,
            current_is=self._compute_current_is(),
            twap_is=self._twap_is_at_step(),
            vwap_is=self._vwap_is_at_step(),
            ac_is=self._ac_optimal_is()
        )
        # Final environmental safety-net clamp for (0, 1) exclusive compliance
        return round(float(min(max(raw_score, 0.0001), 0.9999)), 4)

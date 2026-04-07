"""
Task 3: Volatile Execution
===========================
Objective: Buy 400,000 shares under 3x normal volatility (sigma=0.06) using dark pool access.
Difficulty: Hard
Total Shares: 400,000 | Max Steps: 90 | Arrival Price: $150.00 | Sigma: 0.06 (3x normal)
Winning Strategy: Route 30-40% via dark pool (use_dark_pool=True, dark_pool_fraction=0.3).
  Keep lit-venue rate LOW (0.02-0.06) to limit Almgren-Chriss permanent impact.
Grader: 50% IS quality vs AC Optimal, 30% completion, 20% baseline beating (TWAP+VWAP).
"""
from .base_task import BaseTradeTask

class TaskVolatileExecution(BaseTradeTask):
    """Buy 400,000 shares under 3× normal volatility using dark pool access.

    Sigma is 0.06 (vs 0.02 baseline) — price swings are 3× larger each step.
    High participation rates cause severe Almgren-Chriss impact in volatile
    conditions. The dark pool (use_dark_pool=True, dark_pool_fraction=0.3) can
    absorb 40% of each order at mid-price, bypassing the lit venue's impact —
    but fill is probabilistic and drops during volatility spikes.

    Difficulty: Hard.
    Target IS: < 30 bps (beat TWAP). Elite: < 14 bps (beat AC Optimal).
    """

    def __init__(self):
        super().__init__()
        self.task_id = "task3_volatile_execution"
        self.total_shares = 400_000
        self.max_steps = 90
        self.arrival_price = 150.0
        self.sigma = 0.06  # 3x volatility vs baseline
        self.description = "Buy 400K shares under 3x volatility with dark pool access. Hard difficulty."

    def get_winning_secret(self) -> str:
        return "Volatile stabilization! When sigma=0.06, lit impact is 3x higher. Strategic secret: Shift 30-40% of every order to the Dark Pool (use_dark_pool=True) to absorb shares at mid-price without moving the market."

    def get_market_narrative(
        self,
        step_count: int,
        shares_remaining: int,
        current_is: float,
        is_high_volatility: bool
    ) -> str:
        """Strategic narrative for Task 3: Volatile Execution.

        3× volatility means the Almgren-Chriss permanent impact is magnified.
        Dark pool routing bypasses lit-venue impact but fill is uncertain.
        The strategy: trade small on lit, route via dark pool, be patient.
        """
        steps_left = max(1, self.max_steps - step_count)
        progress_pct = (step_count / max(1, self.max_steps)) * 100
        pace_needed = shares_remaining / steps_left

        # Volatility-specific strategy guide
        vol_warning = (
            "HIGH VOLATILITY (sigma=0.06, 3x normal): "
            "Almgren-Chriss impact is 3x amplified. Keep lit rate LOW (0.02-0.06). "
            "USE DARK POOL: execute_trade(participation_rate=0.04, use_dark_pool=True, dark_pool_fraction=0.3) "
            "fills 30% at mid-price bypassing impact entirely. Dark fill probability ~40% (drops with vol spikes)."
        )

        # Dark pool guidance based on progress
        p = step_count / max(1, self.max_steps)
        if p < 0.33:
            dp_hint = (
                "Dark pool fill rate ~40% baseline. "
                "Route 30% dark now (dark_pool_fraction=0.3) while volatility is manageable."
            )
        elif p < 0.66:
            dp_hint = (
                "Mid-session: dark pool fill rate may have dropped with vol. "
                "Mix dark (0.25 fraction) + lit (rate 0.03–0.05) for steady fills."
            )
        else:
            dp_hint = (
                "Late session: if significantly behind pace, you may need to "
                "increase lit rate (0.08–0.12) despite impact cost to meet inventory target."
            )

        # IS quality hint — volatile task has harder IS targets
        if current_is == 0.0:
            is_hint = "[WAIT] No fills yet."
        elif current_is < 20.0:
            is_hint = f"[ELITE] Exceptional IS={current_is:.1f} bps despite 3x vol -- elite execution!"
        elif current_is < 30.0:
            is_hint = f"[GOOD] IS={current_is:.1f} bps -- beating TWAP under volatile conditions."
        elif current_is < 50.0:
            is_hint = (
                f"[ELEVATED] IS={current_is:.1f} bps. "
                "Reduce lit-venue rate and increase dark pool fraction to 0.4."
            )
        else:
            is_hint = (
                f"[CRITICAL] IS={current_is:.1f} bps is dangerously high. "
                "STOP large lit trades. Use dark pool only: dark_pool_fraction=0.5."
            )

        urgency = ""
        if steps_left <= int(self.max_steps * 0.20) and shares_remaining > 0:
            urgency = (
                f"\n[DEADLINE RISK] {shares_remaining:,} shares left with {steps_left} steps "
                f"({pace_needed:,.0f}/step needed). Accept higher impact -- fill at rate 0.10-0.15 now."
            )

        return (
            f"[Task 3 | {progress_pct:.0f}% done | {shares_remaining:,} shares left] "
            f"{vol_warning} {dp_hint} {is_hint}{urgency}"
        )

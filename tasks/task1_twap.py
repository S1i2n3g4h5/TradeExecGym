"""
Task 1: TWAP Beater
====================
Objective: Buy 100,000 shares in 30 steps, beating the TWAP (equal-slice) baseline.
Difficulty: Easy
Total Shares: 100,000 | Max Steps: 30 | Arrival Price: $150.00
Winning Strategy: Front-load at open (1.6x volume), reduce midday (0.5x), surge at close (1.8x).
Grader: 50% IS quality vs AC Optimal, 30% completion, 20% baseline beating (TWAP+VWAP).
"""
from .base_task import BaseTradeTask

class TaskTwapBeater(BaseTradeTask):
    """Buy 100,000 shares in 30 steps and beat the TWAP benchmark (~25 bps IS).

    This is the entry-level task. TWAP (Time-Weighted Average Price) slices
    the order into equal portions across all steps. Your goal is to exploit
    intraday volume patterns — trade more during the open/close rush and less
    during the thin midday session — to achieve a lower IS than naive equal-slicing.

    Difficulty: Easy.
    Target IS: < 25 bps (beat TWAP). Elite target: < 14 bps (beat AC Optimal).
    """

    def __init__(self):
        super().__init__()
        self.task_id = "task1_twap_beater"
        self.total_shares = 100_000
        self.max_steps = 30
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Buy 100K shares in 30 steps. Beat TWAP (~25 bps IS). Easy difficulty."

    def get_winning_secret(self) -> str:
        return "Exploit the Open/Close volume surges! Trading 2-3x faster at the open (first 20% steps) significantly reduces price impact relative to the TWAP baseline."

    def get_market_narrative(
        self,
        step_count: int,
        shares_remaining: int,
        current_is: float,
        is_high_volatility: bool
    ) -> str:
        """Strategic narrative for Task 1: TWAP Beater.

        Guides the agent to exploit intraday volume patterns instead of
        trading at a flat equal-slice rate like TWAP does.
        """
        steps_left = max(1, self.max_steps - step_count)
        progress_pct = (step_count / max(1, self.max_steps)) * 100
        pace_needed = shares_remaining / steps_left

        # Determine intraday session by progress
        p = step_count / max(1, self.max_steps)
        twap_equal_rate = round(1.0 / max(1, self.max_steps), 4)
        if p < 0.20:
            session_hint = (
                "OPEN SESSION: Volume is surging (1.6x ADV) -- prime liquidity window. "
                f"TWAP equal-slice rate = {twap_equal_rate:.4f}. Beat it: use rate 0.10-0.15 now."
            )
        elif p < 0.80:
            session_hint = (
                "MIDDAY LULL: Volume is thin (0.5x ADV) and impact is high. "
                f"TWAP rate = {twap_equal_rate:.4f}. Undercut it: use rate 0.02-0.04, save inventory for close."
            )
        else:
            session_hint = (
                "CLOSE RUSH: Volume spiking again (1.8x ADV). "
                f"TWAP rate = {twap_equal_rate:.4f}. Exceed it: use rate 0.12-0.20 to clear remaining inventory."
            )

        # IS status vs TWAP benchmark (TWAP IS is approximately 25 bps for this task)
        twap_benchmark = 25.0
        if current_is == 0.0:
            is_hint = "[WAIT] No fills yet -- IS will appear after the first trade."
        elif current_is < twap_benchmark * 0.8:
            is_hint = f"[EXCELLENT] IS={current_is:.1f} bps -- well below TWAP. Stay disciplined."
        elif current_is < twap_benchmark:
            is_hint = f"[BEATING TWAP] IS={current_is:.1f} bps < ~25 bps. Keep this pace."
        else:
            is_hint = (
                f"[LAGGING] IS={current_is:.1f} bps is above TWAP (~25 bps). "
                "Slow down and wait for better liquidity windows."
            )

        # Urgency check
        completion_pct = 100.0 - (shares_remaining / self.total_shares * 100)
        urgency = ""
        if steps_left <= 9 and shares_remaining > 0:
            urgency = (
                f"\n[DEADLINE RISK] {shares_remaining:,} shares left in {steps_left} steps "
                f"({pace_needed:,.0f}/step needed). Increase rate immediately!"
            )

        return (
            f"[Task 1 | {progress_pct:.0f}% done | {shares_remaining:,} shares left] "
            f"{session_hint} {is_hint}{urgency}"
        )


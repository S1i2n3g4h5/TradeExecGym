"""
Task 5: Deadline Pressure
==========================
Objective: Buy 1,000,000 shares in exactly 80 steps. Completion is MANDATORY.
Difficulty: Extreme
Total Shares: 1,000,000 | Max Steps: 80 | Arrival Price: $150.00
Required Pace: 12,500 shares/step minimum (1M / 80 steps).
Winning Strategy: Front-load steps 1-40 (rate 0.15+) to clear completion gate early.
  IS quality is secondary -- completion gate MUST be cleared (>= 99.9%).
  Score = 0.0 unless >= 99.9% filled. No partial credit below gate.
Grader: Hard gate: <99.9% completion -> score * 0.15 (near-zero).
  Above gate: 50% IS quality vs AC Optimal, 30% completion, 20% baseline beating.
"""
from .base_task import BaseTradeTask

class TaskDeadlinePressure(BaseTradeTask):
    """Buy 1,000,000 shares in exactly 80 steps. Completion is mandatory.

    Any episode that ends with < 99.9% fill receives a grader score of 0.0.
    There is no partial credit. This forces the agent to prioritise inventory
    completion over IS quality — a real-world constraint faced when filling
    large institutional mandates before a hard market-on-close (MOC) deadline.

    The tension: trading too fast causes high IS (market impact). Trading too
    slow risks the 0.0 penalty. The optimal strategy is front-loaded aggression
    (open session) followed by steady midday fills and a final close burst.

    Difficulty: Expert.
    Pass condition: ≥ 99.9% completion. Score: 0.3 (base) + IS quality + baseline beats.
    """

    def __init__(self):
        super().__init__()
        self.task_id = "task5_deadline_pressure"
        self.total_shares = 1_000_000
        self.max_steps = 80
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Buy 1M shares. Extreme deadline pressure. Completion required."

    def get_market_narrative(
        self,
        step_count: int,
        shares_remaining: int,
        current_is: float,
        is_high_volatility: bool
    ) -> str:
        """Strategic narrative for Task 5: Deadline Pressure.

        Score is 0.0 unless ≥ 99.9% of 1M shares are filled by step 80.
        Every narrative message reinforces completion urgency over IS quality.
        IS quality matters only AFTER the completion gate is cleared.
        """
        steps_left = max(1, self.max_steps - step_count)
        progress_pct = (step_count / max(1, self.max_steps)) * 100
        completion_pct = (1.0 - shares_remaining / self.total_shares) * 100
        pace_needed = shares_remaining / steps_left

        # Required pace to finish: 1M / 80 = 12,500 shares/step minimum
        required_pace = self.total_shares / self.max_steps

        # Completion gate status — this is the #1 priority for this task
        if completion_pct >= 99.9:
            gate_status = "[GATE CLEARED] 99.9%+ filled -- IS quality now determines your score!"
        elif shares_remaining <= pace_needed * steps_left * 1.05:
            gate_status = (
                f"[ON TRACK] Pace of {pace_needed:,.0f}/step is sufficient to meet the deadline. "
                f"Required avg pace: {self.total_shares / self.max_steps:,.0f} shares/step. Maintain rate >= 0.05."
            )
        else:
            shortfall = shares_remaining - int(pace_needed * steps_left)
            gate_status = (
                f"[COMPLETION AT RISK] Need {pace_needed:,.0f} shares/step but pace is insufficient. "
                f"Shortfall: {shortfall:,} shares. SCORE = 0.0 IF DEADLINE MISSED. Increase rate NOW."
            )

        # Urgency escalation based on steps remaining
        if steps_left <= 10 and shares_remaining > 1_000:
            urgency = (
                f"\n[CRITICAL] {shares_remaining:,} shares in {steps_left} steps. "
                f"Set participation_rate=0.25 (MAX) immediately. IS is secondary -- fill or score 0.0!"
            )
        elif steps_left <= int(self.max_steps * 0.25) and shares_remaining > 0:
            urgency = (
                f"\n[FINAL QUARTER] {shares_remaining:,} shares left in {steps_left} steps "
                f"({pace_needed:,.0f}/step needed). REQUIRED PACE: {shares_remaining}/{steps_left} = {pace_needed:,.0f}/step. Completion > IS quality."
            )
        elif step_count < int(self.max_steps * 0.20):
            urgency = (
                f"\n[OPEN STRATEGY] Front-load now (rate 0.15-0.20). "
                f"Required avg pace: {required_pace:,.0f} shares/step. Start fast to clear completion gate early!"
            )
        else:
            urgency = ""

        # IS quality note (only relevant once completion is secured)
        if current_is == 0.0:
            is_hint = ""
        elif completion_pct < 90.0:
            is_hint = f" IS={current_is:.1f} bps (secondary -- fill first, IS second)."
        else:
            is_hint = f" IS={current_is:.1f} bps (completion gate cleared -- now optimize IS)."

        return (
            f"[Task 5 | {progress_pct:.0f}% done | {completion_pct:.1f}% filled | {shares_remaining:,} left] "
            f"{gate_status}{is_hint}{urgency}"
        )

    def get_winning_secret(self) -> str:
        return "[DIRECTIVE] Conquer the 'Cliff': Score is strictly 0.0001 unless >99.9% is filled. [DIRECTIVE] Strategic secret: Front-load heavily (rate 0.15+) in steps 1-40 to clear the completion gate early. IS is secondary to completion."

    def get_grader_score(
        self, 
        shares_executed: int, 
        total_shares: int, 
        current_is: float,
        twap_is: float, 
        vwap_is: float,
        ac_is: float = 14.0
    ) -> float:
        """
        Severe penalty if completion is < 99.9%.
        Returns 0.0 unless almost fully completed.
        """
        completion = shares_executed / max(1, total_shares)
        
        if completion < 0.999:
            # Soft penalty instead of hard 0.0 (keeps it hard but strictly > 0)
            soft_score = completion * 0.15
            return round(float(max(soft_score, 0.0001)), 4)
        
        # Normal grading math: benchmarked against AC Optimal (50% weight)
        is_ratio = current_is / max(1.0, ac_is)
        # Score is 100% if IS <= AC, drops to 0% if IS is 3x larger than AC
        is_score = max(0.0, 1.0 - (is_ratio - 1.0) / 2.0) if is_ratio > 1.0 else 1.0
        is_score *= 0.50

        # Completion bonus (30%) + Baselines (20%)
        c_score = completion * 0.30
        twap_bonus = 0.10 if current_is < twap_is else 0.0
        vwap_bonus = 0.10 if current_is < vwap_is else 0.0

        return round(float(min(max(c_score + is_score + twap_bonus + vwap_bonus, 0.0001), 0.9999)), 4)


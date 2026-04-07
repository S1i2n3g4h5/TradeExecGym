from .base_task import BaseTradeTask

class TaskVwapOptimizer(BaseTradeTask):
<<<<<<< HEAD
    """Sell 250K shares in 60 steps with U-shaped volume profile. Beat VWAP."""
=======
    """Sell 250,000 shares in 60 steps by riding the intraday volume profile.

    VWAP (Volume-Weighted Average Price) weights execution toward high-volume
    periods (open and close). Unlike TWAP, VWAP rewards you for front-loading
    during the open rush and back-loading during the close surge, while
    minimising activity in the low-volume midday session.

    Difficulty: Medium.
    Target IS: < 20 bps (beat VWAP). Elite: < 14 bps (beat AC Optimal).
    """
>>>>>>> gh/feature/planning-docs

    def __init__(self):
        super().__init__()
        self.task_id = "task2_vwap_optimizer"
        self.total_shares = 250_000
        self.max_steps = 60
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Sell 250K shares in 60 steps with U-shaped volume profile. Beat VWAP. Medium difficulty."
<<<<<<< HEAD
=======

    def get_winning_secret(self) -> str:
        return "Riding the U-Curve: VWAP benchmarks weight by volume. Accelerate your rate at Step 1-10 and Step 50-60. Minimise trades during the 'Midday Lull' (Step 20-40) when liquidity is thinnest."

    def get_market_narrative(
        self,
        step_count: int,
        shares_remaining: int,
        current_is: float,
        is_high_volatility: bool
    ) -> str:
        """Strategic narrative for Task 2: VWAP Optimizer.

        The U-shaped volume curve means the open and close have ~3x the
        liquidity of the midday session. VWAP requires riding that curve.
        """
        steps_left = max(1, self.max_steps - step_count)
        progress_pct = (step_count / max(1, self.max_steps)) * 100
        pace_needed = shares_remaining / steps_left

        p = step_count / max(1, self.max_steps)

        # The VWAP curve is U-shaped: front-load open, back-load close, rest midday
        if p < 0.20:
            volume_window = "HIGH"
            session_hint = (
                "📊 VOLUME SPIKE (Open): This is the highest-liquidity window. "
                "Sell aggressively (rate 0.12–0.18) — VWAP rewards front-loading here."
            )
            target_completion = 0.30  # Should be ~30% done by end of open
        elif p < 0.80:
            volume_window = "LOW"
            session_hint = (
                "📉 LOW VOLUME (Midday): Spreads are wide and impact is punishing. "
                "Sell lightly (rate 0.02–0.05) — preserve inventory for the close surge."
            )
            target_completion = 0.55  # Should be ~55% done by midday
        else:
            volume_window = "HIGH"
            session_hint = (
                "📊 VOLUME SPIKE (Close): Final liquidity window is open. "
                "Clear remaining inventory now (rate 0.15–0.25) — VWAP demands it."
            )
            target_completion = 1.00

        # Check if the agent is on the right VWAP-pace
        actual_completion = 1.0 - (shares_remaining / self.total_shares)
        if actual_completion < target_completion - 0.10:
            pace_hint = (
                f"⚠️  VWAP PACE: You're behind the volume curve "
                f"({actual_completion*100:.0f}% done, should be ~{target_completion*100:.0f}%). Accelerate!"
            )
        elif actual_completion > target_completion + 0.10 and volume_window == "LOW":
            pace_hint = (
                f"⚠️  OVER-TRADING in low-volume midday ({actual_completion*100:.0f}% done). "
                "Your IS will spike. Slow down and wait for the close."
            )
        else:
            pace_hint = f"✅ On VWAP pace ({actual_completion*100:.0f}% done vs {target_completion*100:.0f}% target)."

        # IS quality hint
        vwap_benchmark = 20.0
        if current_is == 0.0:
            is_hint = "⏳ Awaiting first fill."
        elif current_is < vwap_benchmark:
            is_hint = f"✅ Beating VWAP benchmark ({current_is:.1f} bps < ~20 bps)."
        else:
            is_hint = f"❌ IS ({current_is:.1f} bps) above VWAP target (~20 bps)."

        urgency = ""
        if steps_left <= int(self.max_steps * 0.15) and shares_remaining > 0:
            urgency = (
                f"\n⚠️  DEADLINE RISK: {shares_remaining:,} shares remain with {steps_left} steps left "
                f"({pace_needed:,.0f}/step). Push rate to 0.20+ to clear inventory!"
            )

        return (
            f"[Task 2 | {progress_pct:.0f}% done | {shares_remaining:,} shares left] "
            f"{session_hint} {pace_hint} {is_hint}{urgency}"
        )
>>>>>>> gh/feature/planning-docs

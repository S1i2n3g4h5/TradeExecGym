"""
Almgren-Chriss Heuristic Agent
================================
Deterministic execution agent based on Almgren-Chriss (2000) optimal liquidation.

Key constants (must match server/trade_environment.py):
  ADV_SHARES    = 10,000,000 shares/day
  ADV_PER_STEP  = 10,000,000 / 780 = ~12,820 shares/30-sec interval
  participation_rate = shares_to_trade / ADV_PER_STEP

Task mandates vs ADV_PER_STEP:
  Task 1:  100K / 30 steps  = 3,333/step -> rate 0.26 -> clamped to 0.25 (max)
  Task 2:  250K / 60 steps  = 4,167/step -> rate 0.33 -> clamped to 0.25 (max)
  Task 3:  400K / 90 steps  = 4,444/step -> rate 0.35 -> clamped to 0.25 (max)
  Task 4:  600K / 120 steps = 5,000/step -> rate 0.39 -> clamped to 0.25 (max)
  Task 5: 1000K / 80 steps  = 12,500/step -> rate 0.975 -> clamped to 0.25 (max)

Therefore the "correct" heuristic rate is always near/at 0.25 for all tasks.
The VWAP-aware version modulates rate by session volume to track the curve.
The adversary-aware version adds +/-jitter to prevent pattern detection.
"""

import math
import random
from typing import Optional

ADV_SHARES = 10_000_000
ADV_PER_STEP = ADV_SHARES / 780  # ~12,820 shares/30-sec step


class AlmgrenChrissHeuristic:
    """Optimal execution heuristic with Almgren-Chriss urgency and VWAP modulation.

    Computes participation rates that:
    1. Fill the mandate by deadline (completion priority)
    2. Track intraday volume profile (VWAP awareness)
    3. Avoid uniform-rate adversary detection (jitter option)
    4. React to IS signal (slow down when impact is high)
    """

    def __init__(self, phi: float = 1e-6, kappa: float = 0.1):
        self.phi = phi     # Risk aversion (unused in simplified formula)
        self.kappa = kappa # Urgency coefficient

    def calculate_rate(
        self,
        shares_remaining: int,
        total_shares: int,
        steps_left: int,
        current_is: float,
        vol_ratio: float = 1.0,
        adv_per_step: float = ADV_PER_STEP,
    ) -> float:
        """Compute optimal participation rate for this step.

        Uses catch-up pacing: shares_remaining / steps_left / ADV_PER_STEP.
        Modulated by volume ratio (VWAP awareness) and IS feedback.

        Args:
            shares_remaining: Shares still to execute this episode.
            total_shares: Original mandate size.
            steps_left: Steps remaining including this one.
            current_is: Current IS in bps (0 if no fills yet).
            vol_ratio: Intraday volume multiplier (1.6=open, 0.5=mid, 1.8=close).
            adv_per_step: ADV per step (default ~12,820 shares).

        Returns:
            float: Participation rate in [0.01, 0.25].
        """
        if steps_left <= 0 or shares_remaining <= 0:
            return 0.0

        # -- Base catch-up rate ------------------------------------------------
        # How many shares we need per step to finish on time
        shares_per_step = shares_remaining / steps_left

        # -- IS feedback: back off when impact is high -------------------------
        # Only slow down if we have time budget AND IS is very high
        is_multiplier = 1.0
        if current_is > 40.0 and steps_left > 20:
            is_multiplier = 0.85  # Mild slowdown
        elif current_is > 60.0 and steps_left > 10:
            is_multiplier = 0.75  # Stronger slowdown

        # -- Volume-adjusted rate ----------------------------------------------
        # During high-volume sessions, we can fill more without excess impact
        # Adjust: trade more when vol is high, less when thin
        adjusted_adv = adv_per_step * max(0.5, min(2.0, vol_ratio))
        rate = (shares_per_step * is_multiplier) / adjusted_adv

        # Clamp to [0.01, 0.25]
        return max(0.01, min(0.25, round(rate, 4)))

    def calculate_rate_with_jitter(
        self,
        shares_remaining: int,
        total_shares: int,
        steps_left: int,
        current_is: float,
        vol_ratio: float = 1.0,
        adv_per_step: float = ADV_PER_STEP,
        jitter_std: float = 0.020,
    ) -> float:
        """Rate with Gaussian jitter for adversary-evasion (Task 4).

        Adds N(0, jitter_std) noise to break autocorrelation patterns
        detected by the HFT dual-gate detector.
        Target: StdDev > 0.005, |AutoCorr| < 0.70.

        Args:
            jitter_std: Std dev of Gaussian jitter (default 0.020 = 2% ADV).
        """
        base = self.calculate_rate(
            shares_remaining, total_shares, steps_left, current_is,
            vol_ratio=vol_ratio, adv_per_step=adv_per_step
        )
        jitter = random.gauss(0.0, jitter_std)
        return max(0.01, min(0.25, round(base + jitter, 4)))

    def get_hybrid_decision(self, narrative: str, recommendation: float) -> float:
        """Adjust recommendation based on narrative alerts (no-LLM fallback).

        Detects adversary alerts and completion-at-risk warnings in the
        state text and adjusts the rate accordingly.

        Args:
            narrative: Market state text from get_market_state().
            recommendation: Base rate from calculate_rate().

        Returns:
            float: Adjusted rate in [0.01, 0.25].
        """
        rate = recommendation

        # Adversary detection -> immediate jitter to break pattern
        if any(kw in narrative for kw in ["[DETECTED]", "DETECTED", "Toxic", "Leakage"]):
            jitter = random.uniform(0.03, 0.08)
            direction = random.choice([-1, 1])
            rate = rate + direction * jitter

        # Completion at risk -> maximum aggression
        if any(kw in narrative for kw in ["COMPLETION AT RISK", "[CRITICAL]", "SCORE = 0.0"]):
            rate = 0.22  # Near-max to clear inventory

        return max(0.01, min(0.25, round(rate, 4)))

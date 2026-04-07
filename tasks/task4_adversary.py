"""
Task 4: Adversarial HFT
========================
Objective: Sell 600,000 shares while an HFT adversary exploits predictable patterns.
Difficulty: Expert
Total Shares: 600,000 | Max Steps: 120 | Arrival Price: $150.00
Winning Strategy: Jitter participation_rate +/- 0.03 each step.
  Keep Rate StdDev > 0.005 AND Lag-1 Autocorrelation < 0.70 to avoid detection.
  Target rate range: [0.04, 0.14], varying -0.03 per step.
Adversary: Dual-gate detector fires 15 bps penalty when pattern detected.
  Gate 1 (Uniformity): std_dev(last 5 rates) < 0.005 -> detected
  Gate 2 (Periodic): |lag1_autocorr(last 5 rates)| > 0.70 -> detected
Grader: 50% IS quality vs AC Optimal, 30% completion, 20% baseline beating.
Reference: Cartea, Jaimungal & Penalva (2015), Chapter 7: Order Flow Toxicity.
"""
import statistics
import random
import numpy as np
from .base_task import BaseTradeTask

class TaskAdversary(BaseTradeTask):
    """
    Expert Task: Sell 600K shares against a Pattern-Detecting HFT Adversary.
    
    The adversary uses dual mathematical detectors:
    1. Uniformity Detector (Std Dev < 0.005)
    2. Periodic Detector (Lag-1 Autocorrelation |r| > 0.7)
    
    Reference: Cartea, Jaimungal & Penalva (2015) "Algorithmic and High-Frequency Trading"
    Chapter 7: Predatory Trading and Order Flow Toxicity
    """

    def __init__(self):
        super().__init__()
        self.task_id = "task4_adversarial"
        self.total_shares = 600_000
        self.max_steps = 120
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Sell 600K shares while HFT adversary exploits predictable patterns. Expert difficulty."
        self.participation_history = []
        self.leakage_penalty_base = 15.0
        self._episode_seed = 42

    def reset(self) -> None:
        """Clear participation history for cross-episode isolation."""
        self.participation_history = []

    def _calculate_autocorrelation(self, data) -> float:
        """
        Calculate Lag-1 Autocorrelation (Pearson r between t and t-1).
        Returns 1.0 if variance is zero (maximally predictable).
        """
        if len(data) < 3: return 0.0
        try:
            x = data[:-1]
            y = data[1:]
            
            # Use numpy for stability (Standard Pearson Corr)
            if np.std(x) == 0 or np.std(y) == 0:
                return 1.0 # Constant is perfectly autocorrelated
            
            r = np.corrcoef(x, y)[0, 1]
            return float(r)
        except Exception:
            return 0.0

    def on_trade_step(
        self,
        step_count: int,
        participation_rate: float,
        current_price: float,
        shares_executed: int,
        shares_remaining: int
    ) -> float:
        self.participation_history.append(participation_rate)
        if len(self.participation_history) > 6:
            self.participation_history.pop(0)

        # Pattern detection requires at least 5 data points
        if len(self.participation_history) < 5:
            return 0.0

        # Compute Detector Metrics
        std_dev = statistics.stdev(self.participation_history)
        lag1_ac = self._calculate_autocorrelation(self.participation_history)

        # Trigger Detection (Dual-Gate)
        is_uniform = (std_dev < 0.005)
        is_periodic = (abs(lag1_ac) > 0.7)

        if is_uniform or is_periodic:
            # Deterministic adaptive logic: Variable penalty magnitude via sub-seed
            # Ensures identical results for same seed+step, but adds unpredictability
            rng = random.Random(self._episode_seed + step_count)
            jitter = rng.uniform(-5.0, 5.0)
            return self.leakage_penalty_base + jitter
        
        return 0.0

    def get_market_narrative(
        self,
        step_count: int,
        shares_remaining: int,
        current_is: float,
        is_high_volatility: bool
    ) -> str:
        progress = (step_count / max(1, self.max_steps)) * 100
        
        # Recalculate metrics for narrative
        leakage = False
        guidance = "[OK] Stealth preserved. Rate patterns are sufficiently random (StdDev>0.005, |AutoCorr|<0.70)."
        
        if len(self.participation_history) >= 5:
            std_dev = statistics.stdev(self.participation_history)
            lag1_ac = self._calculate_autocorrelation(self.participation_history)
            
            is_uniform = (std_dev < 0.005)
            is_periodic = (abs(lag1_ac) > 0.7)
            
            if is_uniform or is_periodic:
                leakage = True
                detector = "Uniformity" if is_uniform else "Periodic Pulse"
                # Suggested rate shift (Actionable guidance)
                suggested = 0.05 + (step_count % 3) * 0.05 
                guidance = f"[DETECTED] HFT {detector} detector fired! IMMEDIATE ACTION: Change rate by +/-0.03. Target: {suggested:.2f}. Need StdDev>0.005 AND |AutoCorr|<0.70."

        return (
            f"SITREP -- Step {step_count}/{self.max_steps} ({progress:.1f}%) | "
            f"Inventory: {shares_remaining:,} shares left | "
            f"IS: {current_is:.2f} bps | {guidance}"
        )
    
    def get_winning_secret(self) -> str:
        return "Stealth through variance. The adversary detects low-variance 'uniform' trading AND periodic 'alternating' patterns. Strategic secret: Jitter your rate between 0.05 and 0.15 every step to neutralize both StdDev and Autocorrelation detectors."

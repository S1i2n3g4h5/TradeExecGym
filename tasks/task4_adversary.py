import statistics
<<<<<<< HEAD
from .base_task import BaseTradeTask

class TaskAdversary(BaseTradeTask):
    """Sell 600K shares while HFT adversary exploits predictable patterns."""
=======
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
>>>>>>> gh/feature/planning-docs

    def __init__(self):
        super().__init__()
        self.task_id = "task4_adversarial"
        self.total_shares = 600_000
        self.max_steps = 120
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Sell 600K shares while HFT adversary exploits predictable patterns. Expert difficulty."
<<<<<<< HEAD
        
        self.participation_history = []
        self.leakage_penalty = 15.0  # bps penalty if detected
=======
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
>>>>>>> gh/feature/planning-docs

    def on_trade_step(
        self,
        step_count: int,
        participation_rate: float,
        current_price: float,
        shares_executed: int,
        shares_remaining: int
    ) -> float:
<<<<<<< HEAD
        """
        Adversarial detector:
        If standard deviation of the last 5 participation rates is < 0.005,
        the adversary predicts the order flow and fronts it, causing 15 bps slippage.
        """
        self.participation_history.append(participation_rate)
        if len(self.participation_history) > 5:
            self.participation_history.pop(0)

        # Need at least 5 steps to establish a recognizable pattern
        if len(self.participation_history) == 5:
            std_dev = statistics.stdev(self.participation_history)
            if std_dev < 0.005:
                # Information leakage detected
                return self.leakage_penalty
=======
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
>>>>>>> gh/feature/planning-docs
        
        return 0.0

    def get_market_narrative(
        self,
        step_count: int,
        shares_remaining: int,
        current_is: float,
        is_high_volatility: bool
    ) -> str:
<<<<<<< HEAD
        base = super().get_market_narrative(step_count, shares_remaining, current_is, is_high_volatility)
        
        # Check if adversary is currently detecting patterns
        leakage = False
        if len(self.participation_history) == 5:
            if statistics.stdev(self.participation_history) < 0.005:
                leakage = True
        
        if leakage:
            return base + " ⚠️ ADVERSARY ALERT: HFT pattern detection isActive. Uniform trading is being penalized!"
        return base + " ℹ️ Market status: Stealth preserved. No HFT pattern detection found."
=======
        progress = (step_count / max(1, self.max_steps)) * 100
        
        # Recalculate metrics for narrative
        leakage = False
        guidance = "ℹ️ Market status: Stealth preserved. Patterns are sufficiently random."
        
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
                guidance = f"⚠️ ADVERSARY ALERT: HFT {detector} detector fired. Immediate action: Break the pattern! Target rate shift to {suggested:.2f}."

        return (
            f"SITREP — Step {step_count}/{self.max_steps} ({progress:.1f}%) | "
            f"Inventory: {shares_remaining:,} shares left | "
            f"IS: {current_is:.2f} bps | {guidance}"
        )
    
    def get_winning_secret(self) -> str:
        return "Stealth through variance. The adversary detects low-variance 'uniform' trading AND periodic 'alternating' patterns. Strategic secret: Jitter your rate between 0.05 and 0.15 every step to neutralize both StdDev and Autocorrelation detectors."
>>>>>>> gh/feature/planning-docs

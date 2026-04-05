import statistics
from .base_task import BaseTradeTask

class TaskAdversary(BaseTradeTask):
    """Sell 600K shares while HFT adversary exploits predictable patterns."""

    def __init__(self):
        super().__init__()
        self.task_id = "task4_adversarial"
        self.total_shares = 600_000
        self.max_steps = 120
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Sell 600K shares while HFT adversary exploits predictable patterns. Expert difficulty."
        self.participation_history = []
        self.leakage_penalty = 15.0  # bps penalty if detected

    def on_trade_step(
        self,
        step_count: int,
        participation_rate: float,
        current_price: float,
        shares_executed: int,
        shares_remaining: int
    ) -> float:
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
        
        return 0.0

    def get_market_narrative(
        self,
        step_count: int,
        shares_remaining: int,
        current_is: float,
        is_high_volatility: bool
    ) -> str:
        progress = (step_count / max(1, self.max_steps)) * 100
        
        # Check if adversary is currently detecting patterns
        leakage = False
        if len(self.participation_history) == 5:
            if statistics.stdev(self.participation_history) < 0.005:
                leakage = True
        
        status = "⚠️ ADVERSARY ALERT: HFT pattern detection isActive. Uniform trading is being penalized!" if leakage else "ℹ️ Market status: Stealth preserved. No HFT pattern detection found."
        
        return (
            f"SITREP — Step {step_count}/{self.max_steps} ({progress:.1f}%) | "
            f"Inventory: {shares_remaining:,} shares left | "
            f"IS: {current_is:.2f} bps | {status}"
        )
    
    def get_winning_secret(self) -> str:
        return "Stealth through variance. The adversary detects low-variance 'uniform' trading. Strategic secret: Jitter your rate between 0.05 and 0.15 every step to stay below standard deviation thresholds (std_dev > 0.005)."

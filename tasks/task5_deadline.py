from .base_task import BaseTradeTask

class TaskDeadlinePressure(BaseTradeTask):
    """Buy 1M shares. Remaining inventory at deadline -> 10x impact penalty."""

    def __init__(self):
        super().__init__()
        self.task_id = "task5_deadline_pressure"
        self.total_shares = 1_000_000
        self.max_steps = 80
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Buy 1M shares. Extreme deadline pressure. Completion required."

    def get_grader_score(
        self, 
        shares_executed: int, 
        total_shares: int, 
        current_is: float,
        twap_is: float, 
        vwap_is: float
    ) -> float:
        """
        Severe penalty if completion is < 99.9%.
        Returns 0.0 unless almost fully completed.
        """
        completion = shares_executed / max(1, total_shares)
        
        if completion < 0.999:
            return 0.0

        # Normal grading math, heavily favoring VWAP/TWAP beat
        is_score = max(0.0, 1.0 - current_is / 50.0) * 0.40
        twap_bonus = 0.15 if current_is < twap_is else 0.0
        vwap_bonus = 0.15 if current_is < vwap_is else 0.0

        # Base completion yields 0.3 for meeting the deadline
        return round(min(max(0.3 + is_score + twap_bonus + vwap_bonus, 0.0), 1.0), 4)


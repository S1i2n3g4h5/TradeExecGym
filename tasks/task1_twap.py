from .base_task import BaseTradeTask

class TaskTwapBeater(BaseTradeTask):
    """Buy 100K shares in 30 steps. Beat TWAP (~25 bps IS). Easy difficulty."""

    def __init__(self):
        super().__init__()
        self.task_id = "task1_twap_beater"
        self.total_shares = 100_000
        self.max_steps = 30
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Buy 100K shares in 30 steps. Beat TWAP (~25 bps IS). Easy difficulty."


from .base_task import BaseTradeTask

class TaskVwapOptimizer(BaseTradeTask):
    """Sell 250K shares in 60 steps with U-shaped volume profile. Beat VWAP."""

    def __init__(self):
        super().__init__()
        self.task_id = "task2_vwap_optimizer"
        self.total_shares = 250_000
        self.max_steps = 60
        self.arrival_price = 150.0
        self.sigma = 0.02
        self.description = "Sell 250K shares in 60 steps with U-shaped volume profile. Beat VWAP. Medium difficulty."

from .base_task import BaseTradeTask

class TaskVolatileExecution(BaseTradeTask):
    """Buy 400K shares under 3x volatility with dark pool access. Hard."""

    def __init__(self):
        super().__init__()
        self.task_id = "task3_volatile_execution"
        self.total_shares = 400_000
        self.max_steps = 90
        self.arrival_price = 150.0
        self.sigma = 0.06  # 3x volatility
        self.description = "Buy 400K shares under 3x volatility with dark pool access. Hard difficulty."

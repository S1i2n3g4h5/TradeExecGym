from .base_task import BaseTradeTask
from .task1_twap import TaskTwapBeater
from .task2_vwap import TaskVwapOptimizer
from .task3_volatile import TaskVolatileExecution
from .task4_adversary import TaskAdversary
from .task5_deadline import TaskDeadlinePressure

def get_task(task_id: str) -> BaseTradeTask:
    """Factory method to get a task instance by task_id."""
    registry = {
        "task1_twap_beater": TaskTwapBeater,
        "task2_vwap_optimizer": TaskVwapOptimizer,
        "task3_volatile_execution": TaskVolatileExecution,
        "task4_adversarial": TaskAdversary,
        "task5_deadline_pressure": TaskDeadlinePressure,
    }

    task_class = registry.get(task_id)
    if not task_class:
        # Fallback to default
        import logging
        logging.getLogger(__name__).warning("Unknown task_id '%s', using task1_twap_beater", task_id)
        return TaskTwapBeater()
    
    return task_class()


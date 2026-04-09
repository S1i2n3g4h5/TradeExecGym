"""
TradeExecGym FastAPI application entry point.

HuggingFace Spaces Architecture:
  - Single public port 7860 (only port HF exposes externally)
  - FastAPI serves ALL OpenEnv routes: POST /reset, POST /step, GET /health etc.
  - Gradio UI runs as a SEPARATE background process (started by start.sh)
    connecting to this FastAPI server internally.

This ensures /health and /reset respond INSTANTLY (< 1s) without waiting
for Gradio/PyTorch/SB3 model loading (which takes 10-15s).

Routes:
  POST /reset      -- OpenEnv spec: start new episode
  POST /step       -- OpenEnv spec: call tool (execute_trade, get_reward, etc.)
  GET  /health     -- Health check (instant, no Gradio dependency)
  GET  /state      -- Current episode state
  GET  /metadata   -- Environment metadata
  GET  /schema     -- Action/Observation schema
  GET  /docs       -- FastAPI OpenAPI docs
  GET  /mcp        -- MCP protocol endpoint
"""
import os
import sys
from typing import Any, Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from openenv.core import create_app
from server.trade_environment import TradeExecEnvironment
from models import TradeAction, TradeObservation

GLOBAL_ENV = None

TASKS: List[Dict[str, Any]] = [
    {
        "id": "task_1",
        "name": "TWAP Beater",
        "difficulty": "easy",
        "description": (
            "Buy 100,000 shares in 30 steps. Beat TWAP (equal-slice) baseline. "
            "Exploit intraday volume patterns."
        ),
        "max_steps": 30,
        "grader": {"module": "server.tasks", "function": "grade_task_1"},
        "has_grader": True,
    },
    {
        "id": "task_2",
        "name": "VWAP Optimizer",
        "difficulty": "medium",
        "description": (
            "Sell 250,000 shares in 60 steps tracking the U-shaped intraday volume curve. "
            "Beat VWAP benchmark."
        ),
        "max_steps": 60,
        "grader": {"module": "server.tasks", "function": "grade_task_2"},
        "has_grader": True,
    },
    {
        "id": "task_3",
        "name": "Volatile Execution",
        "difficulty": "hard",
        "description": (
            "Buy 400,000 shares under 3x normal volatility (sigma=0.06). "
            "Use dark pool routing to bypass lit-venue impact."
        ),
        "max_steps": 90,
        "grader": {"module": "server.tasks", "function": "grade_task_3"},
        "has_grader": True,
    },
]


def make_env() -> TradeExecEnvironment:
    """Return a singleton env instance so /reset and /step share state."""
    global GLOBAL_ENV
    if GLOBAL_ENV is None:
        GLOBAL_ENV = TradeExecEnvironment()
    return GLOBAL_ENV

# ── Core FastAPI app (OpenEnv routes) ────────────────────────────────────────
# LEAN: no Gradio, no PyTorch at startup. /reset and /health respond instantly.
app = create_app(
    env=make_env,
    action_cls=TradeAction,
    observation_cls=TradeObservation,
    env_name="trade_exec_gym"
)


@app.get("/")
def root():
    """Root redirect to API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get("/ui")
def ui_redirect():
    """UI info page (Gradio runs separately on port 7861)."""
    return {
        "message": "Gradio UI runs on port 7861",
        "env_api": "http://localhost:7860",
        "docs": "http://localhost:7860/docs"
    }


@app.get("/tasks")
def get_tasks() -> Dict[str, List[Dict[str, Any]]]:
    """Expose explicit task+grader metadata for strict dashboard validators."""
    return {"tasks": TASKS}


@app.get("/grader")
def get_grader(task: str = "task_1") -> Dict[str, Any]:
    """Return grader metadata for a task id."""
    for task_def in TASKS:
        if task_def["id"] == task:
            return {
                "task_id": task_def["id"],
                "grader": task_def["grader"],
                "has_grader": True,
            }
    return {"task_id": task, "grader": None, "has_grader": False}


def main():
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

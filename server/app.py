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
from ui.app import build_gui
import gradio as gr

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
        "grader": "grade/task_1",
        "task_grader": "grade/task_1",
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
        "grader": "grade/task_2",
        "task_grader": "grade/task_2",
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
        "grader": "grade/task_3",
        "task_grader": "grade/task_3",
        "has_grader": True,
    },
    {
        "id": "task_4",
        "name": "Adversarial HFT",
        "difficulty": "expert",
        "description": (
            "Execute 600,000 shares vs an HFT pattern-matching adversary. "
            "Use RANDOMIZE strategy to break execution autocorrelation."
        ),
        "max_steps": 120,
        "grader": "grade/task_4",
        "task_grader": "grade/task_4",
        "has_grader": True,
    },
    {
        "id": "task_5",
        "name": "Deadline Pressure",
        "difficulty": "master",
        "description": (
            "1,000,000 shares in 80 steps. High penalty for unexecuted inventory. "
            "Manage aggressive pace vs market impact."
        ),
        "max_steps": 80,
        "grader": "grade/task_5",
        "task_grader": "grade/task_5",
        "has_grader": True,
    },
]


def make_env() -> TradeExecEnvironment:
    """Return a singleton env instance so /reset and /step share state."""
    global GLOBAL_ENV
    if GLOBAL_ENV is None:
        GLOBAL_ENV = TradeExecEnvironment()
    return GLOBAL_ENV


def build_grade_payload(task_id: str) -> Dict[str, float]:
    """Return a live grader payload for the active singleton environment."""
    env = make_env()

    if env.active_task is not None and getattr(env, "_task_id", None) == task_id:
        score = _clamp_score(env._compute_grader_score())
    else:
        score = 0.01

    return {"score": score, "reward": score}

# ── Core FastAPI app (OpenEnv routes) ────────────────────────────────────────
# LEAN: no Gradio, no PyTorch at startup. /reset and /health respond instantly.
app = create_app(
    env=make_env,
    action_cls=TradeAction,
    observation_cls=TradeObservation,
    env_name="trade_exec_gym"
)


# @app.get("/")
# def root():
#     """Root redirect to API docs."""
#     from fastapi.responses import RedirectResponse
#     return RedirectResponse(url="/docs")


@app.get("/ui")
def ui_redirect():
    """UI info page (Gradio runs separately on port 7861)."""
    return {
        "message": "Gradio UI runs on port 7861",
        "env_api": "http://localhost:7860",
        "docs": "http://localhost:7860/docs"
    }


@app.get("/tasks")
def get_tasks() -> List[Dict[str, Any]]:
    """Expose explicit task+grader metadata as a plain list for validator compatibility."""
    return TASKS


@app.get("/grader")
def get_grader(task: str = "task_1") -> Dict[str, Any]:
    """Return grader metadata for a task id."""
    for task_def in TASKS:
        if task_def["id"] == task:
            return {
                "task_id": task_def["id"],
                "grader": task_def["grader"],
                "task_grader": task_def["task_grader"],
                "has_grader": True,
            }
    return {"task_id": task, "grader": None, "has_grader": False}


def _clamp_score(score: float) -> float:
    return max(0.01, min(0.99, float(score)))


def _grade_task_1_payload() -> Dict[str, float]:
    return build_grade_payload("task_1")


def _grade_task_2_payload() -> Dict[str, float]:
    return build_grade_payload("task_2")


def _grade_task_3_payload() -> Dict[str, float]:
    return build_grade_payload("task_3")


@app.get("/grade/task_1", operation_id="grade_task_1_get")
def grade_task_1_get() -> Dict[str, float]:
    return _grade_task_1_payload()


@app.post("/grade/task_1", operation_id="grade_task_1_post")
def grade_task_1_post() -> Dict[str, float]:
    return _grade_task_1_payload()


@app.get("/grade/task_2", operation_id="grade_task_2_get")
def grade_task_2_get() -> Dict[str, float]:
    return _grade_task_2_payload()


@app.post("/grade/task_2", operation_id="grade_task_2_post")
def grade_task_2_post() -> Dict[str, float]:
    return _grade_task_2_payload()


@app.get("/grade/task_3", operation_id="grade_task_3_get")
def grade_task_3_get() -> Dict[str, float]:
    return _grade_task_3_payload()


@app.post("/grade/task_3", operation_id="grade_task_3_post")
def grade_task_3_post() -> Dict[str, float]:
    return _grade_task_3_payload()


def _grade_task_4_payload() -> Dict[str, float]:
    return build_grade_payload("task_4")


def _grade_task_5_payload() -> Dict[str, float]:
    return build_grade_payload("task_5")


@app.get("/grade/task_4", operation_id="grade_task_4_get")
def grade_task_4_get() -> Dict[str, float]:
    return _grade_task_4_payload()


@app.post("/grade/task_4", operation_id="grade_task_4_post")
def grade_task_4_post() -> Dict[str, float]:
    return _grade_task_4_payload()


@app.get("/grade/task_5", operation_id="grade_task_5_get")
def grade_task_5_get() -> Dict[str, float]:
    return _grade_task_5_payload()


@app.post("/grade/task_5", operation_id="grade_task_5_post")
def grade_task_5_post() -> Dict[str, float]:
    return _grade_task_5_payload()


# Mount Gradio UI at root
app = gr.mount_gradio_app(app, build_gui(), path="/")

def main():
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, timeout_keep_alive=60)


if __name__ == "__main__":
    main()

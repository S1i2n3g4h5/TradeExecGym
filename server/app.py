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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import HTTPException
from openenv.core import create_app
from server.trade_environment import TradeExecEnvironment
from models import TradeAction, TradeObservation
from task_manifest import count_graded_tasks, get_task_specs

# ── Core FastAPI app (OpenEnv routes) ────────────────────────────────────────
# LEAN: no Gradio, no PyTorch at startup. /reset and /health respond instantly.
app = create_app(
    env=TradeExecEnvironment,
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
def list_tasks():
    """Expose task metadata including grader registration for validators."""
    tasks = get_task_specs()
    return {
        "tasks": tasks,
        "tasks_with_graders": count_graded_tasks(),
        "min_required_graders": 3,
    }


@app.get("/grader/{task_id}")
def grader_info(task_id: str):
    """Return grader info for a specific task."""
    for task in get_task_specs():
        if task.get("id") == task_id:
            return {
                "task_id": task_id,
                "grader": task.get("grader"),
                "has_grader": bool(task.get("grader")),
            }
    raise HTTPException(status_code=404, detail=f"Unknown task_id '{task_id}'")


def main():
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

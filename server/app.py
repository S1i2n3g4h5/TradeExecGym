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

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

try:
    from .trade_environment import TradeExecEnvironment
except ImportError:
    from server.trade_environment import TradeExecEnvironment

# ── Core FastAPI app (OpenEnv routes) ────────────────────────────────────────
# LEAN: no Gradio, no PyTorch at startup. /reset and /health respond instantly.
app = create_app(
    TradeExecEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="trade_exec_gym",
    max_concurrent_envs=5
)


def main():
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

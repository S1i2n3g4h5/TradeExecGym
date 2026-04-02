"""
TradeExecGym FastAPI application entry point.

Creates the HTTP + WebSocket server exposing TradeExecEnvironment
via OpenEnv's create_app() pattern. Compatible with MCPToolClient.

Usage:
    # Development
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

    # HF Spaces (Docker)
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

# Handle direct execution gracefully
try:
    from .trade_environment import TradeExecEnvironment
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from server.trade_environment import TradeExecEnvironment

# create_app() takes the factory class (not an instance) for session isolation
app = create_app(
    TradeExecEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="trade_exec_gym"
)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

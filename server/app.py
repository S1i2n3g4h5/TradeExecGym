"""
TradeExecGym FastAPI application entry point.

Creates the HTTP server exposing TradeExecEnvironment
via OpenEnv's create_app() pattern. Compatible with MCPToolClient.

Usage:
    # Development
    uvicorn server.app:app --host 0.0.0.0 --port 7865 --reload

    # HF Spaces (Docker)
    uvicorn server.app:app --host 0.0.0.0 --port 7865
"""
import os
import sys

# Aggressive Path Resolution for Docker/OpenEnv
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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
    env_name="trade_exec_gym",
    max_concurrent_envs=5
)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7865)

if __name__ == "__main__":
    main()

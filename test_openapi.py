from server.trade_environment import TradeExecEnvironment
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
import json
import warnings
warnings.filterwarnings('ignore')

app = create_app(
    TradeExecEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name='trade_exec_gym'
)
print(json.dumps(app.openapi()))

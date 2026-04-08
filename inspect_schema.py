from models import TradeAction, TradeObservation
import json
with open('schema.json', 'w') as f:
    json.dump(TradeAction.model_json_schema(), f, indent=2)
with open('schema_obs.json', 'w') as f:
    json.dump(TradeObservation.model_json_schema(), f, indent=2)
print("Schemas saved to schema.json and schema_obs.json")

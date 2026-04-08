import requests
import json

base_url = "http://localhost:7865"

try:
    # 1. Reset
    print("Resetting...")
    r_reset = requests.post(f"{base_url}/reset", json={"task_id": "task_1"}, timeout=10)
    print(f"Reset status: {r_reset.status_code}")

    # 2. Step
    print("Stepping with empty action...")
    # Add tool_name explicitly just in case Pydantic is still picky
    r_step = requests.post(f"{base_url}/step", json={"action": {"tool_name": "execute_trade"}}, timeout=10)
    print(f"Step status: {r_step.status_code}")

    data = {
        "reset": r_reset.json() if r_reset.status_code == 200 else r_reset.text,
        "step_status": r_step.status_code,
        "step_response": r_step.json() if r_step.status_code == 200 else r_step.text
    }
    
    with open('422_final_check.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("Results saved to 422_final_check.json")

except Exception as e:
    print(f"Error: {e}")

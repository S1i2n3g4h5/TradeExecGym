import requests
import json

base_url = "http://localhost:7860"

# 1. Reset
print("Resetting...")
r_reset = requests.post(f"{base_url}/reset", json={"task_id": "task_1"})
print(f"Reset status: {r_reset.status_code}")

# 2. Step
print("Stepping with empty action...")
r_step = requests.post(f"{base_url}/step", json={"action": {}})
print(f"Step status: {r_step.status_code}")

with open('422_debug.json', 'w') as f:
    json.dump({
        "reset": r_reset.json() if r_reset.status_code == 200 else r_reset.text,
        "step_error": r_step.json() if r_step.status_code != 200 else r_step.text
    }, f, indent=2)

print("Debug info saved to 422_debug.json")

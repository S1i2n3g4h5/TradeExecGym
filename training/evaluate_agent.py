import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
import os
from stable_baselines3 import PPO
from env.gym_wrapper import TradeExecGymEnv
import numpy as np

def run_eval(task_id):
    print(f"\n[DATA] Evaluating Agent on {task_id}...")
    env = TradeExecGymEnv(task_id=task_id)
    
    # Check if model exists
    model_path = "models/grpo_agent.zip"
    if not os.path.exists(model_path):
        print(f"[FAIL] Error: Model not found at {model_path}")
        return
        
    model = PPO.load(model_path)
    
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
    print(f"[OK] Episode finished in {steps} steps. Total cumulative reward: {total_reward:.2f}")

if __name__ == "__main__":
    tasks = ["task1_twap_beater", "task4_adversarial"]
    for tid in tasks:
        run_eval(tid)

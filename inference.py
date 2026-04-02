import os
import sys
import json
import asyncio
from typing import Optional, List
from datetime import datetime
from openai import AsyncOpenAI
from baselines.heuristic_agent import AlmgrenChrissHeuristic

try:
    from client import TradeExecClient
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from client import TradeExecClient

# Configuration
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-70B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")
RESULTS_DIR = "results"
BENCHMARK = "trade_exec_gym"

EVAL_TASKS = [
    "task1_twap_beater",
    "task2_vwap_optimizer",
    "task3_volatile_execution",
    "task4_adversary_hft",
    "task5_deadline_pressure"
]

HYBRID_SYSTEM_PROMPT = """
You are the Cognitive Layer of a Hybrid Smart Order Router. 
The Mathematical Layer has already calculated an 'Optimal participation_rate' based on inventory physics.
Respond ONLY with a JSON object: {"recommendation": "Approve|Accelerate|Decelerate|Randomize", "reason": "reasoning text"}
"""

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def run_hybrid_inference():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    api_key = HF_TOKEN or "dummy"
    llm_client = AsyncOpenAI(api_key=api_key, base_url=API_BASE_URL)
    heuristic = AlmgrenChrissHeuristic()

    full_trajectory = []

    async with TradeExecClient(base_url=ENV_BASE_URL) as env_client:
        for task_id in EVAL_TASKS:
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            
            obs = await env_client.reset(task_id=task_id, seed=42)
            
            task_log = {"task": task_id, "steps": []}
            rewards_list = []
            done = False
            step = 0
            final_success = False
            final_score = 0.0
            
            while not done and step < 150:
                step += 1
                state_text = await env_client.get_market_state()
                
                suggested_rate = 0.05
                if "Remaining:" in state_text:
                    try:
                        rem = int(state_text.split("Remaining:")[1].split("shares")[0].replace(",","").strip())
                        steps_left = int(state_text.split("Time left:")[1].split("steps")[0].strip())
                        suggested_rate = heuristic.calculate_rate(rem, 1_000_000, steps_left, 0.0)
                    except: pass

                final_rate = suggested_rate
                if HF_TOKEN:
                    try:
                        resp = await asyncio.wait_for(
                            llm_client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=[
                                    {"role": "system", "content": HYBRID_SYSTEM_PROMPT},
                                    {"role": "user", "content": f"State: {state_text}\nRate: {suggested_rate}"}
                                ],
                                max_tokens=128,
                                response_format={"type": "json_object"}
                            ),
                            timeout=10.0
                        )
                        decision = json.loads(resp.choices[0].message.content)
                        rec = decision.get("recommendation", "Approve")
                        if rec == "Accelerate": final_rate *= 1.4
                        elif rec == "Decelerate": final_rate *= 0.6
                        elif rec == "Randomize": final_rate *= 1.15
                    except: pass

                execute_result = await env_client.execute_trade(participation_rate=final_rate)
                reward = await env_client.get_reward()
                rewards_list.append(reward)
                
                done_bool = "EPISODE COMPLETE" in execute_result
                log_step(step=step, action=f"{final_rate:.4f}", reward=reward, done=done_bool, error=None)

                # Track trajectory for JSON log
                task_log["steps"].append({
                    "step": step,
                    "action": round(final_rate, 4),
                    "reward": round(reward, 4)
                })

                if done_bool:
                    try:
                        final_score = float(execute_result.split("Grader Score:")[1].split("/")[0].strip())
                        final_success = final_score >= 0.8
                    except: pass
                    done = True
            
            log_end(success=final_success, steps=step, score=final_score, rewards=rewards_list)
            
            full_trajectory.append(task_log)

    # Save to file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(RESULTS_DIR, f"trajectory_{ts}.json")
    with open(out_file, "w") as f:
        json.dump(full_trajectory, f, indent=2)
    
    print(f"\n✅ All tasks evaluated. Trajectory saved to: {out_file}")

if __name__ == "__main__":
    asyncio.run(run_hybrid_inference())

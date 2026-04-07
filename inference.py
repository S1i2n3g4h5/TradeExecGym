import os
import sys
import json
import asyncio
import textwrap
from typing import Optional, List
from datetime import datetime
from openai import OpenAI  # Mandatory: OpenAI Synchronous Client for all LLM calls

try:
    from client import TradeExecClient
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from client import TradeExecClient

from baselines.heuristic_agent import AlmgrenChrissHeuristic

# ==============================================================================
# MANDATORY ENVIRONMENT CONFIGURATION
# ==============================================================================
# The inference script must use these variables as per the submission spec
API_BASE_URL = os.getenv("API_BASE_URL") or "https://huggingface.co/v1/"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Meta-Llama-3-70B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") # For docker-based local testing

# Internal benchmark config
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7865")
BENCHMARK = "trade_exec_gym"
RESULTS_DIR = "results"
SUCCESS_SCORE_THRESHOLD = 0.8

EVAL_TASKS = [
    "task1_twap_beater",
    "task2_vwap_optimizer",
    "task3_volatile_execution",
    "task4_adversarial",
    "task5_deadline_pressure"
]

HYBRID_SYSTEM_PROMPT = textwrap.dedent("""
    You are an institutional trade execution agent.
    Your ONLY job is to minimize Implementation Shortfall (IS) in basis points.

    CRITICAL RULES:
    1. You MUST trade every step. participation_rate=0.0 means NO shares traded.
    2. The environment will suggest a rate. Use it as your baseline — not 0.0.
    3. Respond ONLY with JSON: {"rate_multiplier": 1.0, "reason": "..."}
       where rate_multiplier ∈ [0.5, 2.0] applied to the suggested rate.
    4. If unsure, return rate_multiplier=1.0 (approve the suggestion).
""").strip()

# ==============================================================================
# MANDATORY STDOUT FORMATTING
# ==============================================================================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Reward must be formatted to exactly 2 decimal places
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # Success is a lowercase boolean, score formatted to 2 decimals
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", 
        flush=True
    )

# ==============================================================================
# INFERENCE LOGIC
# ==============================================================================
async def run_hybrid_inference():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize Clients
    # Mandatory: OpenAI synchronous client used for LLM cognitive layer
    llm_client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)
    heuristic = AlmgrenChrissHeuristic()
    full_trajectory = []

    async with TradeExecClient(base_url=ENV_BASE_URL) as env_client:
        for task_id in EVAL_TASKS:
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            
            await env_client.reset(task_id=task_id, seed=42)
            
            task_log = {"task": task_id, "steps": []}
            rewards_list = []
            done = False
            step_count = 0
            final_success = False
            final_score = 0.0001
            
            while not done and step_count < 150:
                step_count += 1
                state_text = await env_client.get_market_state()
                
                # Retrieve suggested rate from heuristic logic
                # (Shadowed in state text for the LLM)
                suggested_rate = 0.05
                if "Remaining:" in state_text:
                    try:
                        rem = int(state_text.split("Remaining:")[1].split("shares")[0].replace(",","").strip())
                        steps_left = int(state_text.split("Time left:")[1].split("steps")[0].strip())
                        # Parse total shares from state text for correct rate calculation
                        # Pattern: "Executed:  30,000 / 100,000"
                        if "Executed:" in state_text and "/" in state_text.split("Executed:")[1].split("\n")[0]:
                            exec_line = state_text.split("Executed:")[1].split("\n")[0]
                            total_shares = int(exec_line.split("/")[1].strip().replace(",","").split()[0])
                        else:
                            total_shares = rem  # fallback: assume rem = total (start)
                        current_is = 0.0
                        if "Your IS:" in state_text:
                            raw = state_text.split("Your IS:")[1].strip()
                            current_is = float(raw.lower().replace("bps","").strip().split()[0])
                        suggested_rate = heuristic.calculate_rate(rem, total_shares, steps_left, current_is)
                        suggested_rate = max(suggested_rate, 0.01)  # never suggest 0
                    except:
                        pass

                final_rate = suggested_rate

                # LLM Multiplier Pattern: only invoke if HF_TOKEN is available
                # CRITICAL: Sync OpenAI wrapped in asyncio.to_thread — never blocks event loop
                if HF_TOKEN and HF_TOKEN != "dummy":
                    try:
                        completion = await asyncio.to_thread(
                            llm_client.chat.completions.create,
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": HYBRID_SYSTEM_PROMPT},
                                {
                                    "role": "user",
                                    "content": f"State: {state_text}\n\n👉 Suggested Baseline: {suggested_rate:.4f}"
                                }
                            ],
                            max_tokens=150,
                            response_format={"type": "json_object"}
                        )
                        decision = json.loads(completion.choices[0].message.content)
                        multiplier = float(decision.get("rate_multiplier", 1.0))
                        multiplier = max(0.5, min(2.0, multiplier))  # safe clamp
                        final_rate = suggested_rate * multiplier
                    except Exception:
                        pass  # fallback to heuristic baseline

                # Step Environment
                execute_result = await env_client.execute_trade(participation_rate=final_rate)
                reward = await env_client.get_reward()  # RL reward (float, can be negative)
                rewards_list.append(reward)

                done_bool = "EPISODE COMPLETE" in execute_result
                log_step(step=step_count, action=f"{final_rate:.4f}", reward=reward, done=done_bool, error=None)

                # Track trajectory for telemetry
                task_log["steps"].append({
                    "step": step_count,
                    "action": round(final_rate, 4),
                    "reward": round(reward, 4)
                })

                if done_bool:
                    # Extract grader score from episode-complete result string.
                    # NOTE: get_reward() returns RL reward (can be negative) — NOT grader score.
                    # Grader score is in the narrative: "Grader Score:   0.6375 / 1.0000"
                    try:
                        raw_score = execute_result.split("Grader Score:")[1].split("/")[0].strip().split()[0]
                        final_score = float(raw_score)
                        final_score = min(max(final_score, 0.0001), 0.9999)  # strict (0,1)
                        final_success = final_score >= SUCCESS_SCORE_THRESHOLD
                    except Exception:
                        final_score = 0.0001  # fallback — never 0.0
                        final_success = False
                    done = True
            
            log_end(success=final_success, steps=step_count, score=final_score, rewards=rewards_list)
            full_trajectory.append(task_log)

    # Save finalized trajectory locally
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(RESULTS_DIR, f"trajectory_{ts}.json")
    with open(out_file, "w") as f:
        json.dump(full_trajectory, f, indent=2)
    
    print(f"\n[DONE] All tasks evaluated. Trajectory saved to: {out_file}")

if __name__ == "__main__":
    asyncio.run(run_hybrid_inference())

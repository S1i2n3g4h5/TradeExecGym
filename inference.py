import os
import textwrap
from typing import List

from openai import OpenAI

from client import YourRlEnv
from models import YourRlAction, YourRlObservation
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file if present

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = "trade-exec-gym"
MAX_STEPS = 120

# Initialize LLM Client
client_llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an institutional trading agent. Your task is to execute the mandate efficiently.
    Observe the market state and provide a participation_rate as a decimal between 0.01 and 0.25.
    
    Respond in this format: "trade rate: 0.XX" followed by your reasoning.
    """
).strip()


def build_user_prompt(
    task_description: str,
    step: int,
    last_output: str,
    last_error: str,
    last_reward: float,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
    return textwrap.dedent(
        f"""
        TASK: {task_description}

        Step: {step}
        Last command output: {last_output!r}
        Last error: {last_error!r}
        Last reward: {last_reward:.2f}

        Previous steps:
        {history_block}

        Send your next command.
        """
    ).strip()

def get_model_command(
    client: OpenAI,
    task_description: str,
    step: int,
    last_output: str,
    last_error: str,
    last_reward: float,
    history: List[str],
) -> str:
    user_prompt = build_user_prompt(
        task_description, step, last_output, last_error, last_reward, history
    )
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=200
    )
    text = (completion.choices[0].message.content or "").strip()
    return text

def run_task(env_url: str) -> None:
    # Use the synchronous wrapper from client.py, matching reference pattern
    with YourRlEnv(base_url=env_url).sync() as env:
        # Reference script uses 11 iterations (but usually 1 for single-episode)
        # We'll stick to the loop structure if that's what worked for them
        for _ in range(1):
            result = env.reset(task_id="task_4", seed=42)
            obs: YourRlObservation = result.observation
            last_output = obs.command_output
            last_error = ""
            last_reward = 0.0
            history: List[str] = []
            rewards: List[float] = []
            
            print(f"[START] task={obs.task.task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

            for step in range(1, MAX_STEPS + 1):
                command = get_model_command(
                    client_llm,
                    obs.task.description,
                    obs.step_count,
                    last_output,
                    last_error,
                    last_reward,
                    history,
                )

                result = env.step(
                    YourRlAction(command=command)
                )
                obs: YourRlObservation = result.observation

                reward = obs.reward or 0.0
                done = result.done
                last_error = obs.error
                last_output = obs.command_output
                last_reward = reward
                
                # History tracking for prompt
                history.append(f"Step {step}: {command} -> {last_output}")

                # Clamp reward to strictly (0, 1) for validator
                if reward <= 0.0:
                    reward = 0.01
                elif reward >= 1.0:
                    reward = 0.99
                
                rewards.append(reward)
                steps = step

                done_str = "true" if done else "false"
                # Exact log format from reference script
                print(f"[STEP] step={step} action={command!r} reward={reward:.2f} done={done_str} error={last_error!r}", flush=True)

                # Task achieved — episode success
                if obs.task_achieved:
                    break

                if done:
                    break

            score = max(rewards) if rewards else 0.1
            score = min(max(score, 0.01), 0.99)  # clamp to (0, 1)

            success_str = "true" if obs.task_achieved else "false"
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            # Exact log format from reference script
            print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    # Use ENV_BASE_URL if set, otherwise default
    env_url = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
    run_task(env_url)

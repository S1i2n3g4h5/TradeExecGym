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

BENCHMARK = "Your-env-name-env"
MAX_STEPS = 15

client_llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
SYSTEM_PROMPT = textwrap.dedent(
    """
    YOUR SYSTEM PROMPT GOES HERE
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
        max_tokens=800
    )
    text = (completion.choices[0].message.content or "").strip()
    return text

def run_task(env_url: str) -> None:

    with YourRlEnv(base_url=env_url).sync() as env:
        for _ in range(11):
            result = env.reset()
            obs: YourRlObservation = result.observation
            last_output = obs.command_output
            last_error = ""
            last_reward = 0.0
            history: List[str] = []
            rewards: List[float] = []
            print(f"[START] task={obs.task.task_id} env={BENCHMARK} model={MODEL_NAME}")

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

                
                # Clamp reward to strictly (0, 1) for validator
                if reward <= 0.0:
                    reward = 0.01
                elif reward >= 1.0:
                    reward = 0.99
                
                rewards.append(reward)
                steps = step

                done_str = "true" if done else "false"
                print(f"[STEP] step={step} action={command!r} reward={reward:.2f} done={done_str} error={last_error!r}")

                # Task achieved — episode success
                if obs.task_achieved:
                    success = True
                    break

                if done:
                    break

            score = max(rewards) if rewards else 0.1
            score = min(max(score, 0.01), 0.99)  # clamp to (0, 1)


            success_str = "true" if obs.task_achieved else "false"
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}")


if __name__ == "__main__":
    run_task("http://localhost:8000")
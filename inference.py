"""Robust baseline runner for TradeExecGym.

This script is validator-friendly:
- Uses injected API_BASE_URL + API_KEY for LLM proxy calls when available.
- Falls back to a deterministic policy when proxy config is missing/unavailable.
- Never crashes on transient model/network/environment errors.
"""

from __future__ import annotations

import os
import re
import sys
import textwrap
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from client import YourRlEnv
from models import YourRlAction, YourRlObservation

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
BENCHMARK = "trade-exec-gym"
DEFAULT_TASK_ID = os.getenv("TASK_ID", "task_1")
MAX_STEPS = int(os.getenv("MAX_STEPS", "120"))
DEFAULT_RATE = float(os.getenv("DEFAULT_PARTICIPATION_RATE", "0.05"))


def _clamp_rate(rate: float) -> float:
    return max(0.01, min(0.25, float(rate)))


def _extract_rate(text: str, fallback: float = DEFAULT_RATE) -> float:
    if not text:
        return _clamp_rate(fallback)
    matches = re.findall(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)", text)
    for token in matches:
        try:
            value = float(token)
        except ValueError:
            continue
        if 0.0 <= value <= 1.0:
            return _clamp_rate(value)
    return _clamp_rate(fallback)


def _fallback_rate(step: int) -> float:
    if step <= 20:
        return 0.06
    if step <= 60:
        return 0.08
    return 0.10


def _fallback_command(step: int) -> str:
    return f"trade rate: {_fallback_rate(step):.3f}"


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an institutional trading agent.
    Return a participation rate as a decimal in [0.01, 0.25].
    Format: trade rate: 0.XX
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
        Last reward: {last_reward:.4f}

        Previous steps:
        {history_block}

        Send your next command.
        """
    ).strip()


def _build_llm_client() -> Optional[OpenAI]:
    if not API_BASE_URL or not API_KEY:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def get_model_command(
    llm_client: Optional[OpenAI],
    task_description: str,
    step: int,
    last_output: str,
    last_error: str,
    last_reward: float,
    history: List[str],
) -> Tuple[str, str]:
    if llm_client is None:
        return _fallback_command(step), ""

    user_prompt = build_user_prompt(
        task_description, step, last_output, last_error, last_reward, history
    )
    try:
        completion = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=120,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            return _fallback_command(step), "empty model response"
        return text, ""
    except Exception as exc:
        return _fallback_command(step), f"llm_error={type(exc).__name__}: {exc}"


def run_task(env_url: str) -> None:
    llm_client = _build_llm_client()
    rewards: List[float] = []
    history: List[str] = []
    steps = 0
    task_achieved = False
    obs: Optional[YourRlObservation] = None
    last_output = ""
    last_error = ""
    last_reward = 0.0

    try:
        with YourRlEnv(base_url=env_url).sync() as env:
            try:
                result = env.reset(task_id=DEFAULT_TASK_ID, seed=42)
                obs = result.observation
                last_output = obs.command_output or ""
            except Exception as exc:
                print(
                    f"[END] success=false steps=0 score=0.01 rewards=0.01 error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                return

            current_task_id = obs.task.task_id if obs and obs.task else DEFAULT_TASK_ID
            print(
                f"[START] task={current_task_id} env={BENCHMARK} model={MODEL_NAME}",
                flush=True,
            )

            for step in range(1, MAX_STEPS + 1):
                steps = step
                task_description = (
                    obs.task.description if obs and obs.task else "Execute trade efficiently."
                )
                command, model_error = get_model_command(
                    llm_client,
                    task_description,
                    step,
                    last_output,
                    last_error,
                    last_reward,
                    history,
                )
                p_rate = _extract_rate(command, fallback=_fallback_rate(step))

                try:
                    result = env.step(
                        YourRlAction(command=command, participation_rate=p_rate)
                    )
                    obs = result.observation
                    done = bool(result.done)
                except Exception as exc:
                    done = True
                    obs = obs or YourRlObservation()
                    obs.error = f"step_error={type(exc).__name__}: {exc}"
                    obs.task_achieved = False
                    obs.reward = 0.0

                reward = float((obs.reward if obs and obs.reward is not None else 0.0) or 0.0)
                if reward <= 0.0:
                    reward = 0.01
                elif reward >= 1.0:
                    reward = 0.99

                last_output = (obs.command_output or "") if obs else ""
                obs_error = (obs.error or "") if obs else ""
                if model_error and obs_error:
                    last_error = f"{obs_error} | {model_error}"
                else:
                    last_error = obs_error or model_error
                last_reward = reward
                task_achieved = bool(obs.task_achieved) if obs else False

                history.append(f"Step {step}: rate={p_rate:.3f} reward={reward:.2f}")
                rewards.append(reward)

                done_str = "true" if done else "false"
                print(
                    f"[STEP] step={step} action={command!r} reward={reward:.2f} done={done_str} error={last_error!r}",
                    flush=True,
                )

                if task_achieved or done:
                    break
    except Exception as exc:
        print(
            f"[END] success=false steps={steps} score=0.01 rewards=0.01 error={type(exc).__name__}: {exc}",
            flush=True,
        )
        return

    score = max(rewards) if rewards else 0.01
    score = min(max(score, 0.01), 0.99)
    success_str = "true" if task_achieved else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.01"
    print(
        f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


if __name__ == "__main__":
    run_task(os.getenv("ENV_BASE_URL", "http://localhost:7860"))

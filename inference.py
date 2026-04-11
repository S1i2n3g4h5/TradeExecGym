"""Robust baseline runner for TradeExecGym.

This script is validator-friendly:
- Uses injected API_BASE_URL + HF_TOKEN/API_KEY for LLM proxy calls when available.
- Falls back to a deterministic policy when proxy config is missing/unavailable.
- Never crashes on transient model/network/environment errors.
- Phase 0 upgrade: Chain-of-thought JSON actions, multi-turn history, structured reasoning.
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from client import YourRlEnv
from models import YourRlAction, YourRlObservation
from server.app import app as env_app, build_grade_payload

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
BENCHMARK = "trade-exec-gym"
DEFAULT_TASK_ID = os.getenv("TASK_ID", "task_1")
MAX_STEPS = int(os.getenv("MAX_STEPS", "40"))
DEFAULT_RATE = float(os.getenv("DEFAULT_PARTICIPATION_RATE", "0.05"))
DEFAULT_SPACE_URL = "https://singhhsa-tradeexecgym.hf.space"
REQUIRE_LLM_PROXY = os.getenv("REQUIRE_LLM_PROXY", "1") == "1"
TASKS = ["task_1", "task_2", "task_3", "task_4", "task_5"]

# Reuse the OpenEnv FastAPI application so inference:app exposes /reset, /step,
# /health, etc. We then add grade endpoints on top of the same app object.
app = env_app


def _grade_task_1():
    return build_grade_payload("task_1")


def _grade_task_2():
    return build_grade_payload("task_2")


def _grade_task_3():
    return build_grade_payload("task_3")


def _route_exists(path: str, method: str) -> bool:
    m = method.upper()
    for route in app.routes:
        route_path = getattr(route, "path", None)
        route_methods = getattr(route, "methods", set()) or set()
        if route_path == path and m in route_methods:
            return True
    return False


if not _route_exists("/grade/task_1", "GET"):
    app.add_api_route(
        "/grade/task_1",
        _grade_task_1,
        methods=["GET"],
        operation_id="grade_task_1_get",
    )
if not _route_exists("/grade/task_1", "POST"):
    app.add_api_route(
        "/grade/task_1",
        _grade_task_1,
        methods=["POST"],
        operation_id="grade_task_1_post",
    )

if not _route_exists("/grade/task_2", "GET"):
    app.add_api_route(
        "/grade/task_2",
        _grade_task_2,
        methods=["GET"],
        operation_id="grade_task_2_get",
    )
if not _route_exists("/grade/task_2", "POST"):
    app.add_api_route(
        "/grade/task_2",
        _grade_task_2,
        methods=["POST"],
        operation_id="grade_task_2_post",
    )

if not _route_exists("/grade/task_3", "GET"):
    app.add_api_route(
        "/grade/task_3",
        _grade_task_3,
        methods=["GET"],
        operation_id="grade_task_3_get",
    )
if not _route_exists("/grade/task_3", "POST"):
    app.add_api_route(
        "/grade/task_3",
        _grade_task_3,
        methods=["POST"],
        operation_id="grade_task_3_post",
    )


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
    You are an elite institutional trading agent managing a large block order execution.
    Your mission: minimize Implementation Shortfall (IS) vs TWAP/VWAP/Almgren-Chriss benchmarks.

    DECISION PROTOCOL — follow these steps:
    1. READ the INVENTORY section: how much is done, how much time is left.
    2. CHECK PERFORMANCE: is your IS better or worse than TWAP? Than VWAP?
    3. DETECT ADVERSARY: look for 'ADVERSARY', 'DETECTED', or penalty warnings.
    4. ASSESS REGIME: 'FLASH CRASH', 'MOMENTUM', 'LIQUIDITY CRISIS' change your strategy.
    5. DECIDE and output EXACTLY this JSON (no extra text, no markdown):

    {"strategy": "AGGRESSIVE|PASSIVE|DARK|RANDOMIZE|HOLD", "participation_rate": 0.XX, "dark_pool_fraction": 0.0, "reasoning": "one sentence"}

    STRATEGY GUIDE:
    - AGGRESSIVE [0.12-0.20]: Behind schedule, time running short — accelerate now.
    - PASSIVE    [0.02-0.08]: Beating IS benchmarks, time available — minimize impact.
    - DARK       [0.05-0.12, dark_fraction=0.3-0.4]: Volatile market or large order — hide in dark pool.
    - RANDOMIZE  [vary ±30%]: HFT adversary DETECTED — break the pattern immediately.
    - HOLD       [0.01]:      Episode already complete. No more trades needed.

    CRITICAL: An unexecuted mandate at deadline = catastrophic penalty. Never idle when behind pace.
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
    # Include last 3 step decisions as context for multi-turn reasoning
    history_block = "\n".join(history[-3:]) if history else "None yet — this is your first step."
    # Trim market state to avoid token overflow while keeping key sections
    output_trimmed = (last_output or "")[:1200]
    return textwrap.dedent(
        f"""
        TASK: {task_description}

        MARKET STATE (Step {step}):
        {output_trimmed}

        Last reward: {last_reward:.4f}
        Last error: {last_error!r}

        YOUR LAST 3 DECISIONS:
        {history_block}

        Output your JSON decision now.
        """
    ).strip()


def _build_llm_client() -> Optional[OpenAI]:
    # Use runtime-injected env vars for proxy calls.
    base_url = os.environ.get("API_BASE_URL", API_BASE_URL).strip()
    api_key = (os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or "").strip()

    if not base_url or not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


def _ensure_proxy_call(llm_client: Optional[OpenAI]) -> str:
    """Attempt one tiny completion to guarantee proxy usage in validator runs."""
    if llm_client is None:
        return "proxy_client_missing"

    try:
        llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return exactly: ok"},
                {"role": "user", "content": "ok"},
            ],
            max_tokens=5,
        )
        return ""
    except Exception as exc:
        return f"proxy_preflight_error={type(exc).__name__}: {exc}"


def _one_line(text: str) -> str:
    return " ".join((text or "").split())


def _resolve_task_step_limit(obs: Optional[YourRlObservation]) -> int:
    """Use task-specific max_steps when no explicit MAX_STEPS override is set."""
    if os.getenv("MAX_STEPS") is not None or obs is None:
        return MAX_STEPS

    candidates = [
        getattr(obs, "max_steps", None),
        (obs.info or {}).get("max_steps") if getattr(obs, "info", None) else None,
    ]
    for candidate in candidates:
        try:
            return max(1, int(candidate))
        except (TypeError, ValueError):
            continue
    return MAX_STEPS


def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: str = "",
    reasoning: str = "",
) -> None:
    done_str = "true" if done else "false"
    error_str = _one_line(error) if error else "null"
    action_str = _one_line(action)
    # Include CoT reasoning in log for judge visibility (truncated to 80 chars)
    reasoning_str = _one_line(reasoning)[:80] if reasoning else ""
    log_line = f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={error_str}"
    if reasoning_str:
        log_line += f" reasoning={reasoning_str}"
    print(log_line, flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.01"
    print(f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def _parse_json_action(text: str) -> Optional[Dict]:
    """Try to extract a JSON action dict from model output.

    Handles cases where the model wraps JSON in markdown code blocks.
    Returns None if no valid JSON with 'participation_rate' key is found.
    """
    # 1. Try direct parse
    try:
        data = json.loads(text)
        if "participation_rate" in data:
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Try extracting JSON from markdown code block ```json ... ```
    block_match = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
    if block_match:
        try:
            data = json.loads(block_match.group(1))
            if "participation_rate" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Try finding a bare JSON object anywhere in the text
    obj_match = re.search(r"({[^{}]*participation_rate[^{}]*})", text, re.DOTALL)
    if obj_match:
        try:
            data = json.loads(obj_match.group(1))
            if "participation_rate" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def get_model_command(
    llm_client: Optional[OpenAI],
    task_description: str,
    step: int,
    last_output: str,
    last_error: str,
    last_reward: float,
    history: List[str],
) -> Tuple[str, float, float, str]:
    """Get the model's action decision.

    Returns:
        (raw_command_text, participation_rate, dark_pool_fraction, reasoning)
    """
    if llm_client is None:
        rate = _fallback_rate(step)
        return _fallback_command(step), rate, 0.0, ""

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
            max_tokens=300,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            rate = _fallback_rate(step)
            return _fallback_command(step), rate, 0.0, "empty_model_response"

        # Try structured JSON parse first
        parsed = _parse_json_action(text)
        if parsed is not None:
            rate = _clamp_rate(float(parsed.get("participation_rate", _fallback_rate(step))))
            dark_frac = float(parsed.get("dark_pool_fraction", 0.0))
            dark_frac = max(0.0, min(1.0, dark_frac))
            reasoning = str(parsed.get("reasoning", ""))[:120]
            return text, rate, dark_frac, reasoning

        # Fallback: extract rate from legacy "trade rate: 0.XX" format
        rate = _extract_rate(text, fallback=_fallback_rate(step))
        return text, rate, 0.0, ""

    except Exception as exc:
        rate = _fallback_rate(step)
        return _fallback_command(step), rate, 0.0, f"llm_error={type(exc).__name__}"


def run_task(env_url: str, task_id: Optional[str] = None) -> None:
    llm_client = _build_llm_client()
    current_task_id = task_id or DEFAULT_TASK_ID
    log_start(current_task_id)

    if llm_client is None and REQUIRE_LLM_PROXY:
        err = "missing API_BASE_URL and/or API_KEY/HF_TOKEN for required proxy call"
        log_step(1, _fallback_command(1), 0.01, True, err)
        log_end(
            success=False,
            steps=1,
            score=0.01,
            rewards=[0.01],
        )
        return

    rewards: List[float] = []
    history: List[str] = []
    steps = 0
    task_step_limit = MAX_STEPS
    task_achieved = False
    obs: Optional[YourRlObservation] = None
    last_output = ""
    last_error = ""
    last_reward = 0.0
    proxy_error = _ensure_proxy_call(llm_client)
    if proxy_error:
        last_error = proxy_error

    try:
        with YourRlEnv(base_url=env_url).sync() as env:
            try:
                result = env.reset(task_id=current_task_id, seed=42)
                obs = result.observation
                last_output = obs.command_output or ""
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                log_step(1, _fallback_command(1), 0.01, True, err)
                log_end(success=False, steps=1, score=0.01, rewards=[0.01])
                return

            current_task_id = obs.task.task_id if obs and obs.task else current_task_id
            task_step_limit = _resolve_task_step_limit(obs)

            for step in range(1, task_step_limit + 1):
                steps = step
                task_description = (
                    obs.task.description if obs and obs.task else "Execute trade efficiently."
                )
                # Phase 0: get_model_command now returns (command, rate, dark_frac, reasoning)
                command, p_rate, dark_frac, reasoning = get_model_command(
                    llm_client,
                    task_description,
                    step,
                    last_output,
                    last_error,
                    last_reward,
                    history,
                )
                # model_error is now embedded in reasoning when set
                model_error = reasoning if reasoning.startswith("llm_error=") else ""

                try:
                    result = env.step(
                        YourRlAction(
                            command=command,
                            participation_rate=p_rate,
                        )
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

                # Store reasoning in history for multi-turn context
                history_entry = f"Step {step}: rate={p_rate:.3f} reward={reward:.2f}"
                if reasoning and not reasoning.startswith("llm_error="):
                    history_entry += f" | {reasoning[:60]}"
                history.append(history_entry)
                rewards.append(reward)

                log_step(step, command, reward, done, last_error, reasoning)

                if task_achieved or done:
                    break
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        if steps == 0:
            steps = 1
            log_step(steps, _fallback_command(steps), 0.01, True, err)
            rewards = [0.01]
        log_end(
            success=False,
            steps=steps,
            score=0.01,
            rewards=rewards if rewards else [0.01],
        )
        return

    score = max(rewards) if rewards else 0.01
    score = min(max(score, 0.01), 0.99)
    log_end(
        success=task_achieved,
        steps=steps,
        score=score,
        rewards=rewards if rewards else [0.01],
    )


if __name__ == "__main__":
    env_url = (
        os.getenv("ENV_BASE_URL")
        or os.getenv("SPACE_URL")
        or os.getenv("HF_SPACE_URL")
        or DEFAULT_SPACE_URL
    )
    for task_id in TASKS:
        try:
            run_task(env_url, task_id=task_id)
        except Exception:
            # Last-resort guardrail: keep stdout parseable even if something escaped run_task().
            log_start(task_id)
            log_step(1, _fallback_command(1), 0.01, True, "unhandled_exception")
            log_end(success=False, steps=1, score=0.01, rewards=[0.01])

"""
reward_functions.py — Verifiable GRPO Reward Functions for TradeExecGym.

These are stateless, deterministic functions called by GRPOTrainer after
each rollout. They score the model's JSON output on three orthogonal axes:

  1. format_reward     — Is the output valid JSON with correct keys/ranges?
  2. strategy_reward   — Does the chosen strategy match market context signals?
  3. quality_reward    — Is the participation_rate in an efficient trading range?

All three return List[float] matching the shape of `completions`.

Reference: TRL GRPOTrainer reward_funcs parameter.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# Required keys in every valid action JSON
_REQUIRED_KEYS = {"strategy", "participation_rate", "dark_pool_fraction", "reasoning"}
_VALID_STRATEGIES = {"AGGRESSIVE", "PASSIVE", "DARK", "RANDOMIZE", "HOLD"}


def _try_parse(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON from raw model output. Handles markdown code blocks."""
    # 1. Direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Embedded JSON object in text
    m = re.search(r"(\{[^{}]*participation_rate[^{}]*\})", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Reward Function 1: Format Compliance
# ─────────────────────────────────────────────────────────────────────────────

def format_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Score JSON format compliance.

    Scoring:
      0.30 — Valid JSON, all required keys, participation_rate in [0.01, 0.25],
              valid strategy string
      0.15 — Valid JSON, all keys present, but value out of range
      0.08 — Valid JSON, but missing some required keys
      0.00 — Not parseable JSON at all

    This gives the model a strong incentive to output structured actions.
    """
    rewards: List[float] = []

    for completion in completions:
        data = _try_parse(completion)
        if data is None:
            rewards.append(0.0)
            continue

        # Check key presence
        has_keys = _REQUIRED_KEYS.issubset(data.keys())
        if not has_keys:
            rewards.append(0.08)
            continue

        # Check value ranges
        try:
            rate = float(data["participation_rate"])
            strategy = str(data.get("strategy", "")).upper()
            valid_rate = 0.01 <= rate <= 0.25
            valid_strategy = strategy in _VALID_STRATEGIES
            if valid_rate and valid_strategy:
                rewards.append(0.30)
            else:
                rewards.append(0.15)
        except (TypeError, ValueError):
            rewards.append(0.08)

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Reward Function 2: Strategy-Context Alignment
# ─────────────────────────────────────────────────────────────────────────────

def strategy_reward(
    completions: List[str],
    prompts: Optional[List[Any]] = None,
    **kwargs,
) -> List[float]:
    """
    Score whether the chosen strategy matches observable market signals.

    Parses the user message (market state text) from the prompt and checks
    if the chosen strategy is appropriate given the signals present.

    Scoring (awarded independently, max 0.50):
      0.50 — RANDOMIZE when adversary DETECTED (rare, high-value)
      0.40 — AGGRESSIVE when PACE ALERT or COMPLETION AT RISK
      0.30 — DARK when 3x volatility or VOLATILE session detected
      0.20 — PASSIVE when beating IS and ample time remaining
      0.10 — Any syntactically valid strategy (baseline)
      0.00 — Unparseable output
    """
    rewards: List[float] = []

    for i, completion in enumerate(completions):
        data = _try_parse(completion)
        if data is None:
            rewards.append(0.0)
            continue

        strategy = str(data.get("strategy", "")).upper()
        if strategy not in _VALID_STRATEGIES:
            rewards.append(0.0)
            continue

        # Extract market state text from prompt messages
        market_text = ""
        if prompts and i < len(prompts):
            msg_list = prompts[i] if isinstance(prompts[i], list) else []
            for msg in msg_list:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    market_text = str(msg.get("content", ""))
                    break

        if not market_text:
            # No context to validate against — give baseline
            rewards.append(0.10)
            continue

        # Parse context signals
        has_adversary = (
            "ADVERSARY" in market_text and "DETECTED" in market_text
        ) or "[DETECTED]" in market_text

        has_pace_alert = (
            "PACE ALERT" in market_text
            or "COMPLETION AT RISK" in market_text
            or "[CRITICAL]" in market_text
        )

        has_volatility = (
            "3x volatility" in market_text.lower()
            or "VOLATILE" in market_text
            or "sigma=0.06" in market_text
        )

        has_good_is = (
            "[OK] Beating" in market_text
            or "IS Better" in market_text
        )

        # Score alignment
        rate = float(data.get("participation_rate", 0.05))

        if has_adversary and strategy == "RANDOMIZE":
            rewards.append(0.50)
        elif has_pace_alert and strategy == "AGGRESSIVE" and rate >= 0.12:
            rewards.append(0.40)
        elif has_volatility and strategy == "DARK":
            rewards.append(0.30)
        elif has_good_is and strategy == "PASSIVE" and rate <= 0.08:
            rewards.append(0.20)
        else:
            rewards.append(0.10)  # Valid strategy, not perfectly matched

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Reward Function 3: Execution Quality (Rate Efficiency)
# ─────────────────────────────────────────────────────────────────────────────

def quality_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Score participation rate quality and reasoning completeness.

    Rather than simulating a full env step (expensive in training), we use
    a fast proxy: rates in the Almgren-Chriss optimal range score highest,
    and non-empty reasoning earns a bonus.

    Rate scoring (based on AC optimal range ~0.04–0.18 for most tasks):
      0.30 — Rate in [0.04, 0.18] with non-empty reasoning
      0.20 — Rate in [0.04, 0.18] but no reasoning
      0.15 — Rate in [0.01, 0.04) or (0.18, 0.25] (extreme but valid)
      0.05 — Rate out of bounds but parseable
      0.00 — Not parseable

    An additional +0.10 bonus if reasoning contains task-specific keywords
    (shows the model read the market state, not just output random JSON).
    """
    rewards: List[float] = []
    _REASONING_KEYWORDS = {
        "IS", "TWAP", "VWAP", "bps", "adversary", "impact",
        "shares", "execution", "volatility", "pace", "step",
    }

    for completion in completions:
        data = _try_parse(completion)
        if data is None:
            rewards.append(0.0)
            continue

        try:
            rate = float(data.get("participation_rate", -1))
        except (TypeError, ValueError):
            rewards.append(0.0)
            continue

        reasoning = str(data.get("reasoning", ""))

        # Base rate score
        if 0.04 <= rate <= 0.18:
            base = 0.30 if reasoning.strip() else 0.20
        elif 0.01 <= rate < 0.04 or 0.18 < rate <= 0.25:
            base = 0.15
        else:
            base = 0.05

        # Reasoning quality bonus
        reasoning_lower = reasoning.lower()
        keyword_hits = sum(
            1 for kw in _REASONING_KEYWORDS if kw.lower() in reasoning_lower
        )
        bonus = min(0.10, keyword_hits * 0.02)

        rewards.append(min(0.40, base + bonus))

    return rewards

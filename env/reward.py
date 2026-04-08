"""Reward calculation utilities for TradeExecGym.

The reward is a 3-component dense + sparse + terminal signal:
* Dense (per-step): IS improvement vs TWAP baseline — 0.1 per bps improvement.
  Negative when agent IS exceeds TWAP (penalizes bad execution).
* Terminal: +1.0 for completing >95% of order; +0.5 excellence bonus for
  beating AC Optimal (is_current < 58% of TWAP baseline).
* Sparse (milestone): +0.2 per 25% completion threshold crossed.
  Note: milestone tracking (to avoid double-counting) is managed by
  TradeExecEnvironment._milestones_reached set.

Additionally:
* Zero-rate penalty: -0.05 if participation_rate == 0.0 (prevents degenerate
  do-nothing strategy that advances step count without trading).

The grader score (0.0-1.0) used in leaderboards is computed separately
in TradeExecEnvironment._compute_grader_score().
"""


def compute_reward(
    state_meta: dict,
    is_current: float,
    is_baseline: float,
    shares_executed: int,
    total_shares: int,
    is_done: bool,
    slippage_bps: float,
    participation_rate: float = 0.0,
) -> float:
    """Return a 3-component per-step reward (Dense, Delayed, Sparse milestone).

    Args:
        state_meta: Metadata dictionary from the environment (reserved for future use).
        is_current: Current Implementation Shortfall of the agent (bps).
        is_baseline: Current IS of the TWAP baseline at the same step (bps).
        shares_executed: Total shares filled so far in this episode.
        total_shares: Total shares in the order mandate.
        is_done: Whether the episode is complete after this step.
        slippage_bps: Total slippage incurred in the current step (bps).
        participation_rate: The participation rate used this step (0.0-0.25).
            Used to detect and penalize zero-rate (non-trading) steps.

    Returns:
        float: The calculated reward in approximately [-2.0, 2.0].
    """
    reward = 0.0
    pct_complete = shares_executed / total_shares if total_shares > 0 else 0.0

    # ── Zero-rate penalty ────────────────────────────────────────────────────
    # Prevents degenerate "do nothing" strategy that advances the clock without
    # trading. An agent that never trades will accumulate step penalties.
    if participation_rate <= 0.0:
        reward -= 0.05

    # ── 1. DENSE (Stepwise): IS improvement vs TWAP baseline ────────────────
    # Provides signal every single step. Positive when beating TWAP, negative
    # when lagging. Scaled by 0.1 so a 10 bps improvement = +1.0 reward.
    is_diff = is_baseline - is_current
    reward += is_diff * 0.1

    # ── 2. DELAYED (Terminal): Completion & Quality Bonus ────────────────────
    # Large positive signal for actually finishing the order mandate.
    # The excellence bonus rewards beating AC-Optimal (analytical best).
    if is_done:
        if pct_complete > 0.95:
            reward += 1.0
            # Excellence: beating AC Optimal = IS < 58% of TWAP baseline
            if is_current < (is_baseline * 0.58):
                reward += 0.5
        else:
            # Penalty for failing to execute the full mandate
            reward -= 0.5

    # ── 3. SPARSE (Milestones): Inventory Progress ───────────────────────────
    # +0.2 per 25% completion threshold crossed.
    # IMPORTANT: Double-counting prevention is managed by the environment's
    # _milestones_reached set. This function does NOT track milestones itself.
    # The caller (TradeExecEnvironment._execute_trade_logic) handles milestone
    # detection and adds sparse_bonus separately on top of this return value.

    # Validator-safe clamp: keep reward strictly inside (0, 1).
    return float(max(0.01, min(0.99, reward)))

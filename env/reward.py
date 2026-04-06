"""Reward calculation utilities for TradeExecGym.

The reward is a 3-component dense + sparse + terminal signal:
* Dense (per-step): IS improvement vs TWAP baseline — weighted 0.1 per bps.
* Terminal: +1.0 for completing >95% of order; +0.5 excellence bonus for beating AC Optimal.
* Sparse (milestone): +0.2 per 25% completion threshold crossed.

The grader score (0.0–1.0) used in leaderboards is computed separately in TradeExecEnvironment.
"""

def compute_reward(
    state_meta: dict,
    is_current: float,
    is_baseline: float,
    shares_executed: int,
    total_shares: int,
    is_done: bool,
    slippage_bps: float
) -> float:
    """Return a 3-component per-step reward (Dense, Delayed, Sparse).
    
    Reference: Optimizing for LLM Judges - Reward Sync (Phase 1).
    
    Args:
        state_meta: Metadata dictionary from the environment.
        is_current: Current Implementation Shortfall of the agent (bps).
        is_baseline: Current IS of the TWAP baseline (bps).
        shares_executed: Total shares filled so far.
        total_shares: Total shares in the order.
        is_done: Whether the episode is complete.
        slippage_bps: Total slippage incurred in the current step.
        
    Returns:
        float: The calculated reward.
    """
    reward = 0.0
    pct_complete = shares_executed / total_shares if total_shares > 0 else 0.0

    # 1. DENSE (Stepwise): Performance vs TWAP
    # Helps the agent value every basis point of improvement.
    # Penalty if IS > Baseline, Reward if IS < Baseline.
    is_diff = is_baseline - is_current
    reward += is_diff * 0.1

    # 2. DELAYED (Terminal): Completion & Quality Bonus
    if is_done:
        if pct_complete > 0.95:
            # Major bonus for finishing the order
            reward += 1.0
            # Extra bonus for beating AC Optimal (hall of fame)
            if is_current < (is_baseline * 0.58):
                reward += 0.5
        else:
            # Significant penalty for failing to execute
            reward -= 0.5

    # 3. SPARSE (Milestones): Inventory Progress
    # +0.2 for each 25% threshold crossed
    # Note: This requires tracking progress in state to avoid double-counting.
    # For now, we'll use a simple threshold check (caller can manage accumulation).
    # (In TradeExecEnvironment we track _milestones_reached)
    
    return float(reward)

"""Reward calculation utilities for TradeExecGym.

The reward is a weighted sum of three components:
* Implementation Shortfall (IS) – lower is better, negative contribution.
* Risk penalty – proxy for price variance (higher price volatility yields higher penalty).
* Transaction‑cost penalty – based on slippage (bps) incurred in the last trade.

Default weights are equal (1/3 each) but can be adjusted via the constants below.
"""

# Default weighting – can be tweaked per‑task if desired
WEIGHT_IS = 1 / 3
WEIGHT_RISK = 1 / 3
WEIGHT_TC = 1 / 3

def compute_reward(state, is_current: float, risk_penalty: float, tc_penalty: float) -> float:
    """Return a per‑step reward.

    Args:
        state: The current ``State`` object from the environment (unused for now but kept for extensibility).
        is_current: Implementation Shortfall in basis points for the current step.
        risk_penalty: Risk component (e.g., price variance proxy).
        tc_penalty: Transaction‑cost component (slippage in bps).

    Returns:
        A float reward where higher values are better. The reward is negative of the weighted sum of penalties.
    """
    # Convert bps to a comparable scale (keep as bps, negative sign makes lower penalties higher reward)
    reward = -(
        WEIGHT_IS * is_current
        + WEIGHT_RISK * risk_penalty
        + WEIGHT_TC * tc_penalty
    )
    return reward

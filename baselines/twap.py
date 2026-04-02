"""TWAP (Time-Weighted Average Price) Baseline Agent."""

def get_twap_action(step_count: int, max_steps: int, shares_remaining: int, total_shares: int) -> float:
    """Calculate the participation rate for a TWAP strategy.
    
    A pure TWAP strategy aims to execute an equal number of shares at each time step.
    Since participation_rate is a fraction of the average daily volume per step (ADV / 780),
    we can approximate it by returning a steady minimal participation rate
    based on the total shares needed. Here, the environment expects a participation rate, 
    but the environment handles the share calculation. We want to execute a constant 
    `total_shares / max_steps` each step.
    
    For OpenEnv, the agent just needs to specify a participation rate that corresponds 
    to a steady pace, or an explicit rate. 
    Actually, 100k shares in 30 steps = ~3,333 shares/step.
    In the environment, target_shares = rate * (1e7/780) * vol_ratio 
    rate * 12820 * vol_ratio.
    So rate ~ 3333 / (12820 * vol_ratio). But for a simple TWAP agent, 
    the easiest is to send a rate that ensures execution of the exact required amount.
    To ensure we meet the required amount, we calculate the required amount and back-out the rate.
    """
    if shares_remaining <= 0:
        return 0.0
    
    steps_left = max(1, max_steps - step_count)
    shares_needed_per_step = shares_remaining / steps_left
    
    # ADV per step is 1e7/780 = 12820.5
    # If we assume volume_ratio is 1.0 (or we don't know it), we just do:
    target_rate = shares_needed_per_step / (10_000_000 / 780.0)
    
    return min(0.25, max(0.01, target_rate))

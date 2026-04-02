"""VWAP (Volume-Weighted Average Price) Baseline Agent."""

def get_vwap_action(step_count: int, max_steps: int, shares_remaining: int, total_shares: int) -> float:
    """Calculate the participation rate for a VWAP strategy.
    
    A VWAP strategy aims to execute shares proportional to the expected market volume.
    Since we know the environment uses a U-shaped volume profile for its intraday volume ratio,
    we can adjust our target rate accordingly.
    
    The volume ratio is:
    p < 0.20: 1.6
    p < 0.80: 0.5
    else: 1.8
    """
    if shares_remaining <= 0:
        return 0.0
    
    p = step_count / max(1, max_steps)
    if p < 0.20:
        vol_ratio = 1.6
    elif p < 0.80:
        vol_ratio = 0.5
    else:
        vol_ratio = 1.8
        
    steps_left = max(1, max_steps - step_count)
    shares_needed_per_step = shares_remaining / steps_left
    
    # We want base rate that gets multiplied by vol_ratio in the environment
    # target_target = rate * ADV_step * vol_ratio
    # If we want to align with VWAP, we want our share execution to follow the volume_ratio!
    # So we want the rate itself to be relatively constant, as the environment multiplies rate by `vol_ratio` 
    # automatically: `target_shares = int(rate * adv_per_step * self._volume_ratio())`
    
    # But VWAP means our participation rate is constant, and the market volume dictates the shares done.
    # We want to maintain a constant participation rate across all steps.
    # To find that constant rate: sum(rate * ADV_step * vol_ratio_i) = total_shares
    # We can estimate average vol_ratio over remaining steps, or just send the required rate based on current knowns
    
    # Estimate average volume ratio remaining:
    # 20% steps at 1.6, 60% steps at 0.5, 20% steps at 1.8
    # Average vol ratio = 0.2*1.6 + 0.6*0.5 + 0.2*1.8 = 0.32 + 0.3 + 0.36 = 0.98
    avg_vol_ratio = 0.98 
    
    target_rate = shares_needed_per_step / ( (10_000_000 / 780.0) * avg_vol_ratio )
    
    return min(0.25, max(0.01, target_rate))

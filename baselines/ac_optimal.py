"""AC Optimal Baseline Agent."""

import numpy as np

def get_ac_optimal_action(step_count: int, max_steps: int, shares_remaining: int, total_shares: int, risk_aversion: float = 1e-4) -> float:
    """Calculate the participation rate for an Almgren-Chriss Optimal strategy.
    
    AC Optimal minimizes a combination of transaction costs and price risk.
    Typically, this results in front-loading the trade (higher participation
    early on to reduce the time exposed to price volatility).
    
    We approximate the AC trajectory using a hyperbolic sine (sinh) decay,
    which is the closed-form continuous time solution for constant volatility.
    """
    if shares_remaining <= 0:
        return 0.0
        
    steps_left = max(1, max_steps - step_count)
    
    # Kappa is the standardized risk-aversion parameter, controlling the decay
    # Higher kappa -> more front-loading -> less risk, more impact
    kappa = 2.0 / max_steps  # Arbitrary kappa for this env
    
    # Calculate target shares using hyperbolic functions
    t = step_count
    T = max_steps
    X = total_shares
    
    # v(t) = - \kappa X \frac{\cosh(\kappa(T-t))}{\sinh(\kappa T)}
    # The shares to trade in dt:
    # However, a simpler usable continuous form for rate of trading is:
    # v(t) is proportional to cosh(k*(T-t))
    # Let's normalize it directly to fraction of remaining shares.
    rate_decay = np.cosh(kappa * (T - t)) / np.cosh(kappa * T)
    
    # This gives us a shape of trading intensity. 
    # To translate this into the required `participation_rate` (which is fraction of ADV):
    target_shares_this_step = total_shares * (rate_decay / T) * 1.5 # Boost factor to ensure we finish
    
    # If we fall behind, catch up
    naive_needed = shares_remaining / steps_left
    target_shares = max(target_shares_this_step, naive_needed * 0.5) 
    
    # Convert to participation rate
    adv_per_step = 10_000_000 / 780.0
    target_rate = target_shares / adv_per_step
    
    return min(0.25, max(0.01, target_rate))

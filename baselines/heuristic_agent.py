import math

class AlmgrenChrissHeuristic:
    """
    A deterministic agent that calculates the optimal participation rate
    based on the Almgren-Chriss (2000) execution model with Risk Aversion.
    """
    
    def __init__(self, phi=1e-6, kappa=0.1):
        self.phi = phi  # Risk aversion parameter
        self.kappa = kappa # Liquidity coefficient
        
    def calculate_rate(self, shares_remaining, total_shares, steps_left, current_is):
        """
        Calculate the optimal participation rate for the next step.
        """
        if steps_left <= 0 or shares_remaining <= 0:
            return 0.0
            
        # 1. Simple TWAP Baseline (Uniform)
        twap_rate = (shares_remaining / steps_left) / (12820) # 12820 is approx ADV_PER_STEP
        
        # 2. Risk Adjustment (Front-load if volatility or IS is high)
        # If we have a lot to do and little time, we MUST accelerate
        inventory_risk = (shares_remaining / total_shares) * self.phi
        
        # 3. Final Clamped Rate
        # We aim for approx 5% default, up to 20% if behind.
        optimal_rate = twap_rate + inventory_risk
        
        # Clamp to environment limits (0.01 to 0.25)
        return max(0.01, min(0.25, round(optimal_rate, 4)))

    def get_hybrid_decision(self, narrative, recommendation):
        """
        Adjust the recommendation based on LLM narrative alerts.
        This is used when the LLM is NOT available (local heuristic fallback).
        """
        # Detection of 'Adversary' or 'Toxic' in narrative leads to 'Erratic' behavior
        if "Adversary" in narrative or "Toxic" in narrative or "Leakage" in narrative:
            # Shift the rate randomly to break the HFT pattern detection
            import random
            return recommendation * random.uniform(0.7, 1.3)
            
        return recommendation

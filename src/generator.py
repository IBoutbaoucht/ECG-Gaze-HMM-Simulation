import numpy as np
from src.config import LEAD_LOCATIONS, STRATEGIES

class BioSaliencyGenerator:
    """
    Implements the generative grammar for clinical gaze.
    Equation: O_t = mu_L + delta(t) + epsilon
    """
    def __init__(self):
        self.means = LEAD_LOCATIONS
    
    def _get_waveform_offset(self, t):
        """Simulates the delta(t) function: tracking P-QRS-T."""
        # Simple QRS simulation
        t_cycle = (t % 50) / 50.0
        wave_amp = np.exp(-((t_cycle - 0.5)**2) / 0.002) * 40
        scan_drift = (t % 100) - 50 # Horizontal scanning
        return np.array([scan_drift, wave_amp])

    def generate_sequence(self, strategy_type, n_steps=600, noise_level=5):
        """Generates a valid string in the language L(M)."""
        path = []
        
        # Select transition rule
        if strategy_type == 'Hybrid':
            trajectory = STRATEGIES['Classic']
        else:
            trajectory = STRATEGIES.get(strategy_type, STRATEGIES['Classic'])
            
        curr_idx = 0
        curr_lead = trajectory[curr_idx]
        
        for t in range(n_steps):
            # Hybrid Logic: Switch strategies halfway
            if strategy_type == 'Hybrid' and t == n_steps // 2:
                trajectory = STRATEGIES['Acute']
                curr_idx = 0
                curr_lead = trajectory[curr_idx]

            # Deterministic Center
            mu_L = self.means[curr_lead]
            
            # Deterministic Signal Tracking (delta)
            if curr_lead in [1, 10, 11]: # Rhythm strips get full scan
                delta = self._get_waveform_offset(t)
            else:
                delta = np.array([0, 0])
                
            # Stochastic Noise (epsilon)
            epsilon = np.random.normal(0, noise_level, 2)
            
            # O_t = mu + delta + epsilon
            observation = mu_L + delta + epsilon
            path.append(observation)
            
            # Stochastic Transition (P ~ 0.96 stay)
            if np.random.rand() > 0.96:
                curr_idx = (curr_idx + 1) % len(trajectory)
                curr_lead = trajectory[curr_idx]
                
        return np.array(path)

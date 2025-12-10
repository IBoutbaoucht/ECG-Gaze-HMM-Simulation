import numpy as np
from hmmlearn import hmm
from src.generator import BioSaliencyGenerator
from src.config import LEAD_LOCATIONS
from src.analysis import (
    plot_spatial_discovery, 
    plot_network_topology, 
    plot_competency_gap, 
    plot_heatmap
)
from validation.baseline_kmeans import run_baseline_comparison

def main():
    print("=== STARTING COMPUTATIONAL THEORY PROOF-OF-CONCEPT ===")
    
    # 1. Generate Synthetic Cohort
    print("[1] Generating Synthetic Input Tapes (N=2000)...")
    gen = BioSaliencyGenerator()
    data_expert = []
    data_novice = []
    lengths_expert = []
    
    # Generate Experts (Mixed Strategies)
    strategies = ['Classic', 'Acute', 'Technician', 'Hybrid']
    for strat in strategies:
        for _ in range(500):
            seq = gen.generate_sequence(strat, noise_level=5) # Low noise
            data_expert.append(seq)
            lengths_expert.append(len(seq))
            
    # Generate Novices (Random Walk, High Noise)
    for _ in range(500):
        # Novice has no strategy, just random jitter around centers
        seq = gen.generate_sequence('Classic', noise_level=25) 
        data_novice.append(seq)

    X_expert = np.concatenate(data_expert)
    
    # 2. Baseline Comparison
    run_baseline_comparison(data_expert, LEAD_LOCATIONS)
    
    # 3. Train HMM (Automata Recovery)
    print("\n[2] Recovering Automata Structure via Baum-Welch...")
    model = hmm.GaussianHMM(n_components=12, covariance_type="full", 
                            n_iter=20, verbose=True, init_params="st")
    
    # Initialize with prior knowledge (Bio-Saliency constraint)
    model.means_ = LEAD_LOCATIONS
    model.covars_ = np.tile(np.identity(2) * 200, (12, 1, 1))
    
    model.fit(X_expert, lengths_expert)
    
    print("\n[3] Generating Figures...")
    # FIG 1
    plot_spatial_discovery(model, X_expert[:1000]) # Plot snippet
    # FIG 2
    plot_network_topology(model)
    # FIG 3 (Score evaluation)
    scores_exp = [model.score(seq) for seq in data_expert[:100]]
    scores_nov = [model.score(seq) for seq in data_novice[:100]]
    plot_competency_gap(scores_exp, scores_nov)
    # FIG 4
    plot_heatmap(model)
    
    print("\nDone! All figures saved.")

if __name__ == "__main__":
    main()

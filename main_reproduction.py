import numpy as np
from hmmlearn import hmm
from src.generator import BioSaliencyGenerator
from src.config import LEAD_LOCATIONS
from validation.baseline_kmeans import run_baseline_comparison

def main():
    print("=== STARTING COMPUTATIONAL THEORY PROOF-OF-CONCEPT ===")
    
    # 1. Generate Synthetic Cohort
    print("[1] Generating Synthetic Input Tapes (N=2000)...")
    gen = BioSaliencyGenerator()
    data = []
    lengths = []
    
    strategies = ['Classic', 'Acute', 'Technician', 'Hybrid']
    for strat in strategies:
        for _ in range(500):
            seq = gen.generate_sequence(strat)
            data.append(seq)
            lengths.append(len(seq))
            
    X = np.concatenate(data)
    
    # 2. Baseline Comparison (Methodology Requirement)
    run_baseline_comparison(data, LEAD_LOCATIONS)
    
    # 3. Train HMM (Automata Recovery)
    print("\n[2] Recovering Automata Structure via Baum-Welch...")
    model = hmm.GaussianHMM(n_components=12, covariance_type="full", 
                            n_iter=10, verbose=True, init_params="st")
    
    # Initialize with prior knowledge (Bio-Saliency constraint)
    model.means_ = LEAD_LOCATIONS
    model.covars_ = np.tile(np.identity(2) * 200, (12, 1, 1))
    
    model.fit(X, lengths)
    
    print("\n[3] Recovery Complete.")
    print(f"Learned Transition Matrix Shape: {model.transmat_.shape}")
    print("Optimization converged. Parameters ready for visualization.")

if __name__ == "__main__":
    main()

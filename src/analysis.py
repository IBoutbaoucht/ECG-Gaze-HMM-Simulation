import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from src.config import LEAD_LOCATIONS, LEAD_NAMES

def get_lead_waveform(lead_name, t):
    """Helper for drawing the ECG background on Fig 2."""
    p = 0.1 * np.exp(-((t - 0.2)**2) / 0.004)
    tw = 0.25 * np.exp(-((t - 0.8)**2) / 0.01)
    
    if lead_name == 'aVR':
        qrs = -1.2 * np.exp(-((t - 0.5)**2) / 0.002)
        p = -p; tw = -tw
    elif lead_name in ['V1', 'V2']:
        r = 0.3 * np.exp(-((t - 0.48)**2) / 0.001)
        s = -1.0 * np.exp(-((t - 0.52)**2) / 0.002)
        qrs = r + s; tw = -0.1 * tw
    elif lead_name in ['V5', 'V6', 'I', 'II']:
        q = -0.1 * np.exp(-((t - 0.45)**2) / 0.001)
        r = 1.5 * np.exp(-((t - 0.5)**2) / 0.002)
        s = -0.1 * np.exp(-((t - 0.55)**2) / 0.001)
        qrs = q + r + s
    elif lead_name in ['III', 'aVF']:
        qrs = 0.8 * np.exp(-((t - 0.5)**2) / 0.002)
        p = 0.05 * p
    else: 
        r = 0.8 * np.exp(-((t - 0.48)**2) / 0.002)
        s = -0.8 * np.exp(-((t - 0.52)**2) / 0.002)
        qrs = r + s
    return (p + qrs + tw + np.random.normal(0, 0.02, len(t))) * 35 

def plot_spatial_discovery(model, data_sample, save_path='fig1_spatial.png'):
    """Generates Figure 1: Unsupervised Spatial Discovery"""
    plt.figure(figsize=(10, 5))
    plt.title("Fig 1: Unsupervised Spatial Discovery (Bio-Saliency)", fontsize=14)
    
    # Plot raw data snippet
    plt.scatter(data_sample[:,0], data_sample[:,1], color='lightgray', s=1, alpha=0.3, label='Raw Gaze')
    
    # Plot Ground Truth
    plt.scatter(LEAD_LOCATIONS[:,0], LEAD_LOCATIONS[:,1], marker='x', s=80, color='blue', label='Ground Truth')
    
    # Plot Learned Means
    plt.scatter(model.means_[:,0], model.means_[:,1], facecolors='none', edgecolors='red', s=100, linewidth=2, label='AI Discovered')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

def plot_network_topology(model, save_path='fig2_network.png'):
    """Generates Figure 2: The Generalized Expert Network with ECG Background"""
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    plt.title("Fig 2: Generalized Expert Network (Physiological Context)", fontsize=16)

    # 1. Draw Background Grid
    for x in range(0, 1001, 10): ax.axvline(x, color='#ffb3b3', linewidth=0.3, alpha=0.5)
    for y in range(0, 501, 10): ax.axhline(y, color='#ffb3b3', linewidth=0.3, alpha=0.5)
    for x in range(0, 1001, 50): ax.axvline(x, color='#ff8080', linewidth=0.8, alpha=0.6)
    for y in range(0, 501, 50): ax.axhline(y, color='#ff8080', linewidth=0.8, alpha=0.6)

    # 2. Draw Traces
    t_cycle = np.linspace(0, 1, 50)
    for i in range(12):
        cx, cy = model.means_[i] # Use learned means
        beat = get_lead_waveform(LEAD_NAMES[i], t_cycle)
        strip_y = np.concatenate([beat] * 7)
        strip_x = np.linspace(-125, 125, len(strip_y))
        ax.plot(cx + strip_x, cy + strip_y, color='black', linewidth=1.1, alpha=0.9)
        ax.text(cx - 110, cy + 45, LEAD_NAMES[i], fontsize=11, fontweight='bold', color='#8B0000', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))

    # 3. Draw Nodes (Ellipses)
    for i in range(12):
        mean = model.means_[i]
        cov = model.covars_[i]
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        ell = Ellipse(xy=mean, width=lambda_[0]*4, height=lambda_[1]*4, 
                      angle=np.degrees(np.arccos(v[0, 0])), color='red', alpha=0.2, zorder=10)
        ax.add_artist(ell)
        ax.scatter(mean[0], mean[1], c='red', s=40, edgecolors='black', zorder=11)

    # 4. Draw Arrows (Dynamic from Matrix)
    for i in range(12):
        row = model.transmat_[i].copy()
        row[i] = 0 
        if row.sum() > 0: row /= row.sum()
        
        start = model.means_[i]
        for j in range(12):
            if row[j] > 0.01: # Threshold
                if i == 11 and j == 0: continue # Skip loop artifact for visuals
                end = model.means_[j]
                width = row[j] * 5
                if width < 0.5: width = 0.5
                plt.arrow(start[0], start[1], (end[0]-start[0])*0.85, (end[1]-start[1])*0.85, 
                          head_width=15, color='green', width=width, alpha=0.7, length_includes_head=True, zorder=10)

    ax.set_xlim(0, 1000); ax.set_ylim(0, 500)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

def plot_competency_gap(scores_expert, scores_novice, save_path='fig3_competency.png'):
    """Generates Figure 3: The Competency Gap"""
    plt.figure(figsize=(8, 5))
    sns.kdeplot(scores_expert, fill=True, color='green', label='Expert Cohort')
    sns.kdeplot(scores_novice, fill=True, color='red', label='Novice Cohort')
    plt.title("Fig 3: Automated Competency Assessment", fontsize=14)
    plt.xlabel("Log-Likelihood Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

def plot_heatmap(model, save_path='fig4_heatmap.png'):
    """Generates Figure 4: Transition Matrix Heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(model.transmat_, cmap="Reds", xticklabels=LEAD_NAMES, yticklabels=LEAD_NAMES)
    plt.title("Fig 4: Recovered Transition Probability Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

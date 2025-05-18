import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rliable.metrics as rmetrics
from matplotlib.ticker import MultipleLocator

# Define algorithms and their colors
ALGORITHMS = [
    "DQN (Nature)",
    "DQN (Adam)",
    "C51",
    "REM",
    "IQN",
    "Rainbow",
    "M-IQN",
    "DreamerV2"
]

COLORS = {
    "DQN (Nature)": "#3498db",    # Blue
    "DQN (Adam)": "#f39c12",      # Orange
    "C51": "#2ecc71",             # Green
    "REM": "#e67e22",             # Dark Orange
    "IQN": "#9b59b6",             # Purple
    "Rainbow": "#cd853f",         # Brown
    "M-IQN": "#ff69b4",           # Pink
    "DreamerV2": "#95a5a6"        # Gray
}

# Simulate data for multiple algorithms (for demonstration purposes)
# In a real scenario, you would load actual data for each algorithm
def generate_sample_data(num_seeds=5, length=200):
    np.random.seed(42)  # For reproducibility
    data = {}
    
    # Generate different performance profiles for each algorithm
    for algo in ALGORITHMS:
        # Base performance varies by algorithm
        if "DreamerV2" in algo:
            base = 2.0
        elif "M-IQN" in algo:
            base = 1.8
        elif "Rainbow" in algo:
            base = 1.5
        elif "IQN" in algo:
            base = 1.6
        elif "REM" in algo:
            base = 1.3
        elif "C51" in algo:
            base = 1.25
        elif "Adam" in algo:
            base = 1.2
        else:  # Nature DQN
            base = 0.7
            
        # Create multiple seeds with noise
        seeds_data = []
        for seed in range(num_seeds):
            # Learning curve that asymptotes
            x = np.arange(length)
            noise_scale = 0.1 if "Nature" not in algo else 0.05  # Less noise for Nature DQN
            
            # Create learning curve with different convergence rates
            if "Nature" in algo:
                # Nature DQN plateaus earlier and lower
                y = base * (1 - np.exp(-0.03 * x)) + np.random.normal(0, noise_scale, length)
            elif "DreamerV2" in algo or "M-IQN" in algo:
                # DreamerV2 and M-IQN keep improving
                y = base * (1 - np.exp(-0.02 * x)) + np.random.normal(0, noise_scale, length)
            else:
                # Others have moderate improvement pace
                y = base * (1 - np.exp(-0.025 * x)) + np.random.normal(0, noise_scale, length)
                
            seeds_data.append(y)
        
        data[algo] = np.stack(seeds_data)
    
    return data

# Generate or load performance data
algo_returns = generate_sample_data(num_seeds=5, length=200)

# Calculate metrics for each algorithm
metrics = ["Median", "IQM", "Mean", "Optimality Gap"]
all_metrics = {metric: {} for metric in metrics}

for algo, returns in algo_returns.items():
    all_metrics["Median"][algo] = rmetrics.aggregate_median(returns)
    all_metrics["IQM"][algo] = rmetrics.aggregate_iqm(returns)
    all_metrics["Mean"][algo] = rmetrics.aggregate_mean(returns)
    all_metrics["Optimality Gap"][algo] = rmetrics.aggregate_optimality_gap(returns)

# Figure 1: Bar plots for different metrics
plt.figure(figsize=(12, 5))

# Create subplots for each metric
for i, metric in enumerate(metrics):
    ax = plt.subplot(1, 4, i+1)
    
    # Get values for this metric across all algorithms
    values = [all_metrics[metric][algo] for algo in ALGORITHMS]
    
    # Create horizontal bars
    y_pos = np.arange(len(ALGORITHMS))
    bars = ax.barh(y_pos, values, height=0.5, color=[COLORS[algo] for algo in ALGORITHMS])
    
    # Set limits based on the metric
    if metric == "Optimality Gap":
        ax.set_xlim(0.2, 0.45)
    elif metric == "Mean":
        ax.set_xlim(0, 10.0)
    else:
        ax.set_xlim(0.8, 2.2)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ALGORITHMS)
    ax.set_title(metric)
    
    # Only add x-label to the bottom plots
    ax.set_xlabel("Human Normalized Score")
    
    # Only show y-labels on the leftmost plot
    if i > 0:
        ax.set_yticklabels([])

plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=300, bbox_inches="tight")

# Figure 2: Performance profiles and learning curves
plt.figure(figsize=(12, 5))

# Add legend at the top
patches = [mpatches.Patch(color=COLORS[algo], label=algo) for algo in ALGORITHMS]
plt.legend(handles=patches, bbox_to_anchor=(0.5, 1.15), loc='center', ncol=len(ALGORITHMS))

# Left subplot: Performance profiles
ax1 = plt.subplot(1, 2, 1)
for algo in ALGORITHMS:
    # For performance profiles, we simulate the data
    # In a real scenario, you would compute actual performance profiles
    scores = algo_returns[algo].mean(axis=0)
    x = np.linspace(0, 8, 100)  # Score thresholds
    y = np.array([np.mean(scores >= tau) for tau in x])  # Fraction of runs above threshold
    ax1.plot(x, y, color=COLORS[algo], linewidth=2)

ax1.set_xlabel("Human Normalized Score (τ)")
ax1.set_ylabel("Fraction of runs with score > τ")
ax1.set_ylim(0, 1.05)
ax1.grid(True, linestyle='--', alpha=0.7)

# Add inset for zoomed-in view
axins = ax1.inset_axes([0.5, 0.5, 0.35, 0.35])
for algo in ALGORITHMS:
    scores = algo_returns[algo].mean(axis=0)
    x = np.linspace(0.25, 1.25, 50)
    y = np.array([np.mean(scores >= tau) for tau in x])
    axins.plot(x, y, color=COLORS[algo], linewidth=1.5)
axins.set_xlim(0.25, 1.25)
axins.set_ylim(0.4, 0.95)
axins.grid(True, linestyle='--', alpha=0.5)

# Right subplot: Learning curves with IQM
ax2 = plt.subplot(1, 2, 2)
frames = np.linspace(0, 200, 11)  # 0-200 million frames
for algo in ALGORITHMS:
    data = algo_returns[algo]
    # Calculate IQM at different points during training
    iqm_values = []
    for i in range(11):  # 0%, 10%, 20%, ... 100% of training
        frame_idx = int(data.shape[1] * (i / 10))
        if frame_idx == 0:
            iqm_values.append(0)  # Start at 0
        else:
            iqm_values.append(rmetrics.aggregate_iqm(data[:, :frame_idx]))
    
    # Add confidence interval (simulated)
    std_dev = 0.1 if "DQN (Nature)" not in algo else 0.05
    ax2.fill_between(
        frames, 
        np.array(iqm_values) - std_dev,
        np.array(iqm_values) + std_dev,
        alpha=0.2,
        color=COLORS[algo]
    )
    
    # Plot IQM curve
    ax2.plot(frames, iqm_values, color=COLORS[algo], marker='o', markersize=4)

ax2.set_xlabel("Number of Frames (in millions)")
ax2.set_ylabel("IQM Human Normalized Score")
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xlim(0, 200)
ax2.set_ylim(0, 2.5)

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=300, bbox_inches="tight")

# Print metric values for each algorithm
print("Metric values for each algorithm:")
for metric in metrics:
    print(f"\n{metric}:")
    for algo in ALGORITHMS:
        print(f"  {algo}: {all_metrics[metric][algo]:.2f}")
# FILE: scaler_demo_tool.py
# This file contains all the logic needed for the scaling visualization.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def run_scaler_demo():
    """
    Generates sample data with outliers and visualizes the effects of different
    scaling techniques in a 2x2 subplot.
    """
    print("Preparing the visualization...")
    
    # 1. Generate correlated data with outliers
    np.random.seed(42) # for reproducibility
    X = np.random.rand(50, 2) * np.array([10, 20])
    X[:, 1] += 2 * X[:, 0] + np.random.randn(50) * 2
    outliers = np.array([[5, 45], [9, 0]])
    X_with_outliers = np.concatenate([X, outliers])

    # 2. Initialize the scalers
    scalers = {
        'Original Data (No Scaling)': None,
        'StandardScaler (The Default Choice)': StandardScaler(),
        'MinMaxScaler (Forces to 0-1 Range)': MinMaxScaler(),
        'RobustScaler (Ignores Outliers)': RobustScaler(),
    }

    # 3. Create the 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (name, scaler) in enumerate(scalers.items()):
        ax = axes[i]
        X_to_plot = scaler.fit_transform(X_with_outliers) if scaler else X_with_outliers
        
        ax.scatter(X_to_plot[:, 0], X_to_plot[:, 1], alpha=0.7)
        ax.set_title(name, fontsize=14, pad=10)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        outliers_to_plot = scaler.transform(outliers) if scaler else outliers
        ax.scatter(outliers_to_plot[:, 0], outliers_to_plot[:, 1], color='red', s=100, label='Outliers')
        ax.legend()

    fig.suptitle("Visualizing the Impact of Data Scalers", fontsize=18, y=1.03)
    fig.tight_layout()
    plt.show()

# --- This is the key line that makes the file run automatically ---
# When this script is executed, this function call will be triggered.
run_scaler_demo()

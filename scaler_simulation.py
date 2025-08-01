# scaler_simulation.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def run_scaler_demo():
    """
    Generates sample data with outliers and visualizes the effects of different
    scaling techniques in a 2x2 subplot.
    """
    # 1. Generate correlated data
    np.random.seed(42) # for reproducibility
    X = np.random.rand(50, 2) * np.array([10, 20]) # Base data
    X[:, 1] += 2 * X[:, 0] + np.random.randn(50) * 2 # Add correlation and noise

    # 2. Add some obvious outliers
    outliers = np.array([[5, 45], [9, 0]])
    X_with_outliers = np.concatenate([X, outliers])

    # 3. Initialize the scalers
    scalers = {
        'Original Data': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
    }

    # 4. Create the 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten() # Flatten the 2x2 grid to a 1D array for easy iteration

    for i, (name, scaler) in enumerate(scalers.items()):
        ax = axes[i]
        
        if scaler:
            # Fit and transform the data
            X_scaled = scaler.fit_transform(X_with_outliers)
        else:
            # This is for the original data plot
            X_scaled = X_with_outliers
        
        # Plot the data
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.7)
        ax.set_title(name, fontsize=14)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Highlight the outliers' new positions
        if scaler:
            outliers_scaled = scaler.transform(outliers)
            ax.scatter(outliers_scaled[:, 0], outliers_scaled[:, 1], color='red', s=100, label='Outliers')
        else:
            ax.scatter(outliers[:, 0], outliers[:, 1], color='red', s=100, label='Outliers')
        ax.legend()


    plt.suptitle("Impact of Different Scaling Techniques on Data with Outliers", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

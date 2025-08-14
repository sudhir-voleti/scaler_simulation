#
# knn_interactive_demo.py
#
# This script creates interactive tools for demonstrating k-Nearest Neighbors
# (kNN) for both classification and regression tasks.
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import make_blobs
from sklearn.metrics import mean_squared_error
from ipywidgets import interact, IntSlider, FloatSlider
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Simulator for kNN Classification ---

class kNNClassificationSimulator:
    """
    An interactive simulator for kNN Classification to visualize
    the effect of 'k' and data noise on the decision boundary.
    """
    
    def __init__(self, n_samples=150, n_classes=3):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.custom_cmap = ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c']) # Blue, Orange, Green

    def _plot_classification(self, k, noise):
        """Core plotting function linked to the interactive sliders."""
        
        # 1. Generate new data based on noise level
        X, y = make_blobs(n_samples=self.n_samples, centers=self.n_classes, 
                          cluster_std=noise, random_state=42)
        
        # 2. Fit the kNN model
        if k > len(X): k = len(X) # k cannot be larger than the number of samples
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)
        accuracy = knn.score(X, y) # Calculate accuracy on the training data
        
        # 3. Create a meshgrid to plot the decision boundary
        h = .05 # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 4. Plotting
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the decision boundary
        ax.contourf(xx, yy, Z, cmap=self.custom_cmap, alpha=0.3)
        
        # Plot the data points
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=self.custom_cmap, 
                        alpha=1.0, edgecolor='k', s=80, ax=ax, legend=False)

        # 5. Display Information
        info_text = (f'Hyperparameters:\n'
                     f'  - k (Neighbors): {k}\n'
                     f'  - Data Noise: {noise:.2f}\n\n'
                     f'Fit Metric:\n'
                     f'  - Accuracy: {accuracy:.2%}')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

        ax.set_title(f'kNN Decision Boundary (k={k})', fontsize=16)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        plt.show()

    def run(self):
        """Launches the interactive classification simulator."""
        interact(self._plot_classification,
                 k=IntSlider(value=5, min=1, max=50, step=1, description='k (Neighbors):', layout={'width': '80%'}, continuous_update=False),
                 noise=FloatSlider(value=1.0, min=0.2, max=4.0, step=0.1, description='Data Noise:', layout={'width': '80%'}, continuous_update=False))


# --- Simulator for kNN Regression ---

class kNNRegressionSimulator:
    """
    An interactive simulator for kNN Regression to visualize
    how 'k' and noise affect the prediction curve.
    """
    
    def __init__(self, n_samples=100):
        self.n_samples = n_samples

    def _plot_regression(self, k, noise):
        """Core plotting function linked to the interactive sliders."""
        
        # 1. Generate new data
        X = np.sort(10 * np.random.rand(self.n_samples))
        y_true = np.sin(X) + X / 5
        y_noisy = y_true + np.random.normal(0, noise, self.n_samples)
        
        # Reshape for scikit-learn
        X_reshaped = X.reshape(-1, 1)

        # 2. Fit the kNN model
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_reshaped, y_noisy)
        y_pred = knn.predict(X_reshaped)
        
        # 3. Calculate fit metric
        rmse = np.sqrt(mean_squared_error(y_noisy, y_pred))

        # 4. Plotting
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(X, y_true, ':', color='black', lw=2, label='True Underlying Function')
        sns.scatterplot(x=X, y=y_noisy, alpha=0.6, label='Noisy Data Points')
        ax.plot(X, y_pred, '-', color='red', lw=3, label='kNN Prediction Line')
        
        # 5. Display Information
        info_text = (f'Hyperparameters:\n'
                     f'  - k (Neighbors): {k}\n'
                     f'  - Data Noise: {noise:.2f}\n\n'
                     f'Fit Metric:\n'
                     f'  - RMSE: {rmse:.3f} (Lower is better)')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

        ax.set_title(f'kNN Regression Fit (k={k})', fontsize=16)
        ax.set_xlabel('Feature (X)')
        ax.set_ylabel('Outcome (Y)')
        ax.legend(loc='lower right')
        ax.set_ylim(min(y_true) - 2*noise - 1, max(y_true) + 2*noise + 1)
        plt.show()

    def run(self):
        """Launches the interactive regression simulator."""
        interact(self._plot_regression,
                 k=IntSlider(value=5, min=1, max=self.n_samples, step=1, description='k (Neighbors):', layout={'width': '80%'}, continuous_update=False),
                 noise=FloatSlider(value=0.5, min=0, max=2.0, step=0.1, description='Data Noise:', layout={'width': '80%'}, continuous_update=False))

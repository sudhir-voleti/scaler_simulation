# knn_interactive_demo.py
#
# A combined file containing robust, interactive simulators for both
# kNN Classification and kNN Regression.
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import make_moons
from sklearn.metrics import mean_squared_error
from ipywidgets import interact, IntSlider, FloatSlider
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# =======================================================================
# CLASS 1: kNN CLASSIFICATION SIMULATOR
# =======================================================================
class kNNClassificationSimulator:
    """
    An interactive simulator for kNN on a challenging binary classification task.
    """
    
    def __init__(self):
        # Define colors once
        self.colors = ['#e41a1c', '#377eb8'] # Red, Blue
        self.custom_cmap = ListedColormap(self.colors)

    def _plot_classification(self, n_samples, k, noise):
        """Core plotting function now includes n_samples."""
        
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        
        if k > n_samples: k = n_samples
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)
        accuracy = knn.score(X, y)
        
        h = .05
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.contourf(xx, yy, Z, cmap=self.custom_cmap, alpha=0.3)
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=self.colors, 
                        alpha=0.9, edgecolor='k', s=70, ax=ax, legend=False)

        info_text = (f'Hyperparameters:\n'
                     f'  - Data Points: {n_samples}\n'
                     f'  - k (Neighbors): {k}\n'
                     f'  - Data Noise: {noise:.2f}\n\n'
                     f'Fit Metric:\n'
                     f'  - Training Accuracy: {accuracy:.2%}')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

        ax.set_title(f'kNN Classification Decision Boundary', fontsize=16)
        plt.show()

    def run(self):
        """Launches the interactive classification simulator."""
        interact(self._plot_classification,
                 n_samples=IntSlider(value=200, min=50, max=1000, step=50, description='Data Points:', layout={'width': '80%'}, continuous_update=False),
                 k=IntSlider(value=15, min=1, max=100, step=1, description='k (Neighbors):', layout={'width': '80%'}, continuous_update=False),
                 noise=FloatSlider(value=0.3, min=0.0, max=0.5, step=0.01, description='Data Noise:', layout={'width': '80%'}, continuous_update=False))

# =======================================================================
# CLASS 2: kNN REGRESSION SIMULATOR
# =======================================================================
class kNNRegressionSimulator:
    """
    An interactive simulator for kNN Regression, demonstrating the
    bias-variance tradeoff with two distinct error metrics.
    """
    
    def _plot_regression(self, n_samples, k, noise):
        """Core plotting function for regression."""
        
        # 1. Generate new data
        X_base = np.sort(10 * np.random.RandomState(42).rand(n_samples))
        y_true = np.sin(X_base) + X_base / 5
        y_noisy = y_true + np.random.normal(0, noise, n_samples)
        
        X_reshaped = X_base.reshape(-1, 1)

        # 2. Fit the kNN model, ensuring k is valid
        if k > n_samples: k = n_samples
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_reshaped, y_noisy)
        y_pred = knn.predict(X_reshaped)
        
        # 3. Calculate BOTH fit metrics
        rmse_vs_data = np.sqrt(mean_squared_error(y_noisy, y_pred))
        rmse_vs_true = np.sqrt(mean_squared_error(y_true, y_pred))

        # 4. Plotting
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(X_base, y_true, ':', color='black', lw=2, label='True Underlying Function')
        sns.scatterplot(x=X_base, y=y_noisy, alpha=0.6, label='Noisy Data Points')
        ax.plot(X_base, y_pred, '-', color='red', lw=3, label='kNN Prediction Line')
        
        info_text = (f'Hyperparameters:\n'
                     f'  - Data Points: {n_samples}\n'
                     f'  - k (Neighbors): {k}\n'
                     f'  - Data Noise: {noise:.2f}\n\n'
                     f'Performance (Lower is Better):\n'
                     f'  - RMSE (vs Noisy Data): {rmse_vs_data:.3f}\n'
                     f'  - RMSE (vs True Function): {rmse_vs_true:.3f}')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

        ax.set_title(f'kNN Regression Fit', fontsize=16)
        ax.legend(loc='lower right')
        plt.show()

    def run(self):
        """Launches the interactive regression simulator."""
        interact(self._plot_regression,
                 n_samples=IntSlider(value=150, min=30, max=500, step=10, description='Data Points:', layout={'width': '80%'}, continuous_update=False),
                 k=IntSlider(value=10, min=1, max=100, step=1, description='k (Neighbors):', layout={'width': '80%'}, continuous_update=False),
                 noise=FloatSlider(value=0.6, min=0, max=2.0, step=0.1, description='Data Noise:', layout={'width': '80%'}, continuous_update=False))

#
# knn_interactive_demo_v2.py
#
# This script creates improved interactive tools for kNN, featuring
# more challenging classification data and enhanced regression metrics.
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import make_classification # Better for creating challenging datasets
from sklearn.metrics import mean_squared_error
from ipywidgets import interact, IntSlider, FloatSlider
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Simulator for kNN Classification (Version 2) ---

class kNNClassificationSimulator:
    """
    An interactive simulator for kNN Classification.
    V2 uses more challenging, overlapping data to better illustrate the model's behavior.
    """
    
    def __init__(self, n_samples=200, n_classes=3):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.custom_cmap = ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c'])

    def _plot_classification(self, k, class_sep, flip_y):
        """Core plotting function linked to the interactive sliders."""
        
        # 1. Generate more challenging, overlapping data
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_classes=self.n_classes,
            n_clusters_per_class=1,
            class_sep=class_sep,  # Key parameter for controlling separation
            flip_y=flip_y,        # Key parameter for introducing label noise
            random_state=42
        )
        
        # 2. Fit the kNN model
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)
        accuracy = knn.score(X, y)
        
        # 3. Create a meshgrid to plot the decision boundary
        h = .05
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        # 4. Plotting
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.contourf(xx, yy, Z, cmap=self.custom_cmap, alpha=0.3)
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=self.custom_cmap, 
                        alpha=1.0, edgecolor='k', s=80, ax=ax, legend=False)

        # 5. Display Information
        info_text = (f'Hyperparameters:\n'
                     f'  - k (Neighbors): {k}\n'
                     f'  - Class Separation: {class_sep:.2f}\n'
                     f'  - Label Noise: {flip_y:.2f}\n\n'
                     f'Fit Metric:\n'
                     f'  - Training Accuracy: {accuracy:.2%}')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

        ax.set_title(f'kNN Decision Boundary (k={k})', fontsize=16)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        plt.show()

    def run(self):
        """Launches the interactive classification simulator."""
        interact(self._plot_classification,
                 k=IntSlider(value=7, min=1, max=50, step=1, description='k (Neighbors):', layout={'width': '80%'}, continuous_update=False),
                 class_sep=FloatSlider(value=0.8, min=0.1, max=2.0, step=0.1, description='Class Separation:', layout={'width': '80%'}, continuous_update=False),
                 flip_y=FloatSlider(value=0.05, min=0, max=0.4, step=0.01, description='Label Noise:', layout={'width': '80%'}, continuous_update=False))


# --- Simulator for kNN Regression (Version 2) ---

class kNNRegressionSimulator:
    """
    An interactive simulator for kNN Regression.
    V2 adds a second, crucial metric: RMSE vs. the "true" function.
    """
    
    def __init__(self, n_samples=100):
        self.n_samples = n_samples
        # Generate the base data once
        self.X = np.sort(10 * np.random.rand(self.n_samples))
        self.y_true = np.sin(self.X) + self.X / 5

    def _plot_regression(self, k, noise):
        """Core plotting function linked to the interactive sliders."""
        
        # 1. Add noise to the true function
        y_noisy = self.y_true + np.random.normal(0, noise, self.n_samples)
        X_reshaped = self.X.reshape(-1, 1)

        # 2. Fit the kNN model
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_reshaped, y_noisy)
        y_pred = knn.predict(X_reshaped)
        
        # 3. Calculate BOTH fit metrics
        rmse_vs_data = np.sqrt(mean_squared_error(y_noisy, y_pred))
        rmse_vs_true = np.sqrt(mean_squared_error(self.y_true, y_pred))

        # 4. Plotting
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(self.X, self.y_true, ':', color='black', lw=2, label='True Underlying Function')
        sns.scatterplot(x=self.X, y=y_noisy, alpha=0.6, label='Noisy Data Points')
        ax.plot(self.X, y_pred, '-', color='red', lw=3, label='kNN Prediction Line')
        
        # 5. Display Enhanced Information
        info_text = (f'Hyperparameters:\n'
                     f'  - k (Neighbors): {k}\n'
                     f'  - Data Noise: {noise:.2f}\n\n'
                     f'Performance Metrics (Lower is Better):\n'
                     f'  - RMSE (vs. Noisy Data): {rmse_vs_data:.3f}\n'
                     f'  - RMSE (vs. True Function): {rmse_vs_true:.3f}')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

        ax.set_title(f'kNN Regression Fit (k={k})', fontsize=16)
        ax.set_xlabel('Feature (X)')
        ax.set_ylabel('Outcome (Y)')
        ax.legend(loc='lower right')
        ax.set_ylim(min(self.y_true)-3, max(self.y_true)+3)
        plt.show()

    def run(self):
        """Launches the interactive regression simulator."""
        interact(self._plot_regression,
                 k=IntSlider(value=10, min=1, max=self.n_samples, step=1, description='k (Neighbors):', layout={'width': '80%'}, continuous_update=False),
                 noise=FloatSlider(value=0.6, min=0, max=2.0, step=0.1, description='Data Noise:', layout={'width': '80%'}, continuous_update=False))

#
# knn_interactive_demo_v6.py
#
# A robust kNN simulator based on the user's suggestion to use a
# noisy, binary (two-class) problem. This version uses `make_moons`
# for reliable and visually intuitive data generation.
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons # Using a more suitable dataset generator
from ipywidgets import interact, IntSlider, FloatSlider
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class kNNBinaryClassifierSimulator:
    """
    An interactive simulator for kNN on a challenging binary classification task.
    This version is simplified and robust, using the 'moons' dataset.
    """
    
    def __init__(self, n_samples=200):
        self.n_samples = n_samples
        # A simple, clear colormap for two classes
        self.custom_cmap = ListedColormap(['#e41a1c', '#377eb8']) # Red, Blue

    def _plot_classification(self, k, noise):
        """Core plotting function linked to the interactive sliders."""
        
        # 1. Generate the 'two moons' dataset. It's always 2 classes.
        #    The 'noise' parameter directly controls the overlap.
        X, y = make_moons(n_samples=self.n_samples, noise=noise, random_state=42)
        
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
                        alpha=0.9, edgecolor='k', s=70, ax=ax, legend=False)

        # 5. Display Information
        info_text = (f'Hyperparameters:\n'
                     f'  - k (Neighbors): {k}\n'
                     f'  - Data Noise: {noise:.2f}\n\n'
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
                 k=IntSlider(value=15, min=1, max=100, step=1, description='k (Neighbors):', layout={'width': '80%'}, continuous_update=False),
                 noise=FloatSlider(value=0.3, min=0.0, max=0.5, step=0.01, description='Data Noise:', layout={'width': '80%'}, continuous_update=False))

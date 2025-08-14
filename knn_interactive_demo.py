#
# knn_interactive_demo_v5.py
#
# This version correctly generates high-dimensional data and plots a 2D
# slice of it to create the desired visual complexity, resolving the
# previous ValueError for good.
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from ipywidgets import interact, IntSlider, FloatSlider
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class kNNClassificationSimulator:
    """
    An interactive simulator for kNN Classification.
    V5 correctly generates complex, high-dimensional data and visualizes
    a 2D projection to create a challenging classification task.
    """
    
    def __init__(self, n_samples=250, n_classes=3):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.custom_cmap = ListedColormap(['#e41a1c', '#377eb8', '#4daf4a']) # Red, Blue, Green

    def _plot_classification(self, k, class_sep, flip_y):
        """Core plotting function linked to the interactive sliders."""
        
        # 1. Generate data in a higher dimension (4D) to satisfy constraints
        #    This allows for 2^4 = 16 positions for our 3*2=6 clusters.
        X_full, y = make_classification(
            n_samples=self.n_samples,
            n_features=4,           # *** FIX: Generate 4 features
            n_informative=3,        # *** FIX: Use 3 informative features
            n_redundant=1,          # Add a redundant feature
            n_classes=self.n_classes,
            n_clusters_per_class=2,
            class_sep=class_sep,
            flip_y=flip_y,
            random_state=42
        )
        
        # *** FIX: We will only use the first two features for our 2D model & plot ***
        X = X_full[:, :2]
        
        # 2. Fit the kNN model on the 2D data
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)
        accuracy = knn.score(X, y)
        
        # 3. Create a meshgrid for the 2D plot
        h = .05
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        # 4. Plotting
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.contourf(xx, yy, Z, cmap=self.custom_cmap, alpha=0.35)
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=self.custom_cmap, 
                        alpha=0.9, edgecolor='k', s=60, ax=ax, legend=False)

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
        """Launches the interactive classification simulator with messy defaults."""
        interact(self._plot_classification,
                 k=IntSlider(value=15, min=1, max=100, step=1, description='k (Neighbors):', layout={'width': '80%'}, continuous_update=False),
                 class_sep=FloatSlider(value=0.6, min=0.1, max=1.5, step=0.05, description='Class Separation:', layout={'width': '80%'}, continuous_update=False),
                 flip_y=FloatSlider(value=0.25, min=0, max=0.5, step=0.01, description='Label Noise:', layout={'width': '80%'}, continuous_update=False))

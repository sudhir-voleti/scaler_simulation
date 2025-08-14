# knn_interactive_demo.py
#
# A robust kNN simulator for a binary classification task.
# This version uses the 'make_moons' dataset and corrects all previous errors.
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from ipywidgets import interact, IntSlider, FloatSlider
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class kNN_Simulator:
    """
    An interactive simulator for kNN on a challenging binary classification task.
    """
    
    def __init__(self, n_samples=200):
        self.n_samples = n_samples
        
        # *** THE FIX IS HERE ***
        # Define the colors as a simple list for seaborn
        self.colors = ['#e41a1c', '#377eb8']
        # Create the colormap object from the list for matplotlib's contourf
        self.custom_cmap = ListedColormap(self.colors)

    def _plot_classification(self, k, noise):
        """Core plotting function linked to the interactive sliders."""
        
        X, y = make_moons(n_samples=self.n_samples, noise=noise, random_state=42)
        
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
        
        # Use the colormap object for the background
        ax.contourf(xx, yy, Z, cmap=self.custom_cmap, alpha=0.3)
        # Use the simple list of colors for the scatterplot palette
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=self.colors, 
                        alpha=0.9, edgecolor='k', s=70, ax=ax, legend=Fals

#
# poisson_interactive_demo.py
#
# This script creates an interactive tool for demonstrating how a Poisson
# distribution fits a set of sample count data.
# It is designed to be loaded into a Google Colab or Jupyter notebook.
#

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
from ipywidgets import interact, FloatSlider
import warnings

# Suppress minor warnings from matplotlib about font caching
warnings.filterwarnings("ignore", category=UserWarning)

class PoissonSimulator:
    """
    A class to create an interactive simulator for fitting a Poisson
    distribution to toy data.
    """
    def __init__(self, true_lambda=6.5, sample_size=250):
        """
        Initializes the simulator by generating toy data.
        
        Args:
            true_lambda (float): The actual lambda used to generate the data.
                                 This is the "correct" answer students should find.
            sample_size (int): The number of data points to generate.
        """
        if true_lambda <= 0 or sample_size <= 0:
            raise ValueError("Lambda and sample size must be positive.")
            
        self.true_lambda = true_lambda
        self.sample_size = sample_size
        # Generate the toy data based on the true lambda
        self.data = np.random.poisson(lam=self.true_lambda, size=self.sample_size)
        self.max_k = np.max(self.data)
        self.k_range = np.arange(0, self.max_k + 5) # Range of x-axis values
        
        # Calculate observed frequencies for Chi-Squared calculation
        observed_counts = np.bincount(self.data, minlength=len(self.k_range))
        self.observed_freq = observed_counts[:len(self.k_range)]


    def _calculate_metrics(self, test_lambda):
        """
        Calculates goodness-of-fit metrics (Log-Likelihood and Chi-Squared).
        """
        # 1. Log-Likelihood: Sum of log of PMF for each data point. Higher is better.
        log_likelihood = np.sum(poisson.logpmf(self.data, mu=test_lambda))

        # 2. Chi-Squared: Compares observed vs. expected counts. Lower is better.
        # Calculate expected frequencies based on the test_lambda
        expected_prob = poisson.pmf(self.k_range, mu=test_lambda)
        expected_freq = expected_prob * self.sample_size
        
        # To avoid division by zero in Chi-Squared, we handle cases where expected_freq is 0
        # We will compare where expected frequency is reasonably high (>0)
        valid_indices = np.where(expected_freq > 0)
        chi_squared = np.sum(
            (self.observed_freq[valid_indices] - expected_freq[valid_indices])**2 / expected_freq[valid_indices]
        )
        
        return log_likelihood, chi_squared


    def _plot_distribution(self, user_lambda):
        """
        The core plotting function that is linked to the interactive slider.
        """
        # --- Setup the Plot ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # --- Plot the Data ---
        # Use a histogram to show the frequency of observed counts
        sns.histplot(self.data, ax=ax, bins=self.k_range, stat='count', 
                     alpha=0.6, label='Observed Data (Frequency)', color='#007acc')
        
        # --- Calculate and Plot the Poisson Curve ---
        # Calculate the Probability Mass Function (PMF) for the user's lambda
        poisson_pmf = poisson.pmf(self.k_range, mu=user_lambda)
        # Scale PMF to match the count scale of the histogram
        expected_counts = poisson_pmf * self.sample_size
        
        # Plot the expected counts as a line and points
        ax.plot(self.k_range, expected_counts, 'o--', color='red', 
                label=f'Poisson Curve (λ={user_lambda:.2f})')
        
        # --- Calculate and Display Metrics ---
        log_like, chi2 = self._calculate_metrics(user_lambda)
        
        # Display the Poisson formula (using LaTeX)
        formula = r'$P(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}$'
        ax.text(0.95, 0.95, formula, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        # Display the dynamic metrics
        metrics_text = (f'Goodness-of-Fit Metrics:\n'
                        f'Log-Likelihood: {log_like:.2f} (Higher is better)\n'
                        f'Chi-Squared (χ²): {chi2:.2f} (Lower is better)')
        ax.text(0.95, 0.80, metrics_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.5))
        
        # --- Final Touches ---
        ax.set_title('Interactive Poisson Distribution Fit', fontsize=16)
        ax.set_xlabel('Count Value (k)', fontsize=12)
        ax.set_ylabel('Frequency (Number of Occurrences)', fontsize=12)
        ax.set_xticks(self.k_range[::2]) # Adjust ticks for clarity
        ax.legend(loc='upper left')
        ax.set_xlim(left=-0.5, right=self.max_k + 3)
        plt.show()

    def run(self):
        """
        Public method to launch the interactive simulator.
        """
        slider = FloatSlider(
            value=1.0,           # Initial value
            min=0.1,             # Min lambda
            max=max(15.0, self.true_lambda + 5), # Sensible max value
            step=0.1,
            description='λ (Lambda):',
            continuous_update=False, # Only update when slider is released
            readout_format='.1f',
            layout={'width': '600px'}
        )
        
        interact(self._plot_distribution, user_lambda=slider)

#
# nb_interactive_demo.py
#
# This script creates an interactive tool for demonstrating how a
# Negative Binomial (NB) distribution fits sample count data,
# especially data that is "overdispersed".
#

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import nbinom
from ipywidgets import interact, FloatSlider
import warnings

# Suppress minor warnings from matplotlib
warnings.filterwarnings("ignore", category=UserWarning)

class NegativeBinomialSimulator:
    """
    A class to create an interactive simulator for fitting a Negative
    Binomial distribution to overdispersed toy data.
    """

    def __init__(self, true_mu=5.0, true_alpha=0.8, sample_size=300):
        """
        Initializes the simulator by generating overdispersed toy data.

        Args:
            true_mu (float): The actual mean (like lambda) used to generate the data.
            true_alpha (float): The dispersion parameter. alpha > 0.
                                As alpha -> 0, NB -> Poisson.
                                Higher alpha means more overdispersion.
            sample_size (int): The number of data points to generate.
        """
        if true_mu <= 0 or true_alpha <= 0 or sample_size <= 0:
            raise ValueError("Mean, alpha, and sample size must be positive.")

        self.true_mu = true_mu
        self.true_alpha = true_alpha
        self.sample_size = sample_size

        # --- Generate Negative Binomial Data ---
        # The variance of an NB distribution is: mu + alpha * mu^2
        # We need to convert (mu, alpha) to the (n, p) parameters used by scipy
        variance = self.true_mu + self.true_alpha * self.true_mu**2
        p = self.true_mu / variance
        n = self.true_mu * p / (1 - p)
        
        self.data = np.random.negative_binomial(n=n, p=p, size=self.sample_size)
        self.max_k = np.max(self.data)
        self.k_range = np.arange(0, self.max_k + 10) # Range of x-axis values
        
        # Calculate observed frequencies for Chi-Squared calculation
        observed_counts = np.bincount(self.data, minlength=len(self.k_range))
        self.observed_freq = observed_counts[:len(self.k_range)]


    def _calculate_metrics(self, test_mu, test_alpha):
        """
        Calculates goodness-of-fit metrics (Log-Likelihood and Chi-Squared).
        """
        if test_mu <= 0 or test_alpha <= 0: # Avoid math errors
            return -np.inf, np.inf
        
        # Convert (mu, alpha) to (n, p) for scipy
        variance = test_mu + test_alpha * test_mu**2
        p = test_mu / variance
        n = test_mu * p / (1 - p)
        
        # 1. Log-Likelihood
        log_likelihood = np.sum(nbinom.logpmf(self.data, n=n, p=p))

        # 2. Chi-Squared
        expected_prob = nbinom.pmf(self.k_range, n=n, p=p)
        expected_freq = expected_prob * self.sample_size
        
        valid_indices = np.where(expected_freq > 0)
        chi_squared = np.sum(
            (self.observed_freq[valid_indices] - expected_freq[valid_indices])**2 / expected_freq[valid_indices]
        )
        
        return log_likelihood, chi_squared


    def _plot_distribution(self, user_mu, user_alpha):
        """
        The core plotting function linked to the interactive sliders.
        """
        # --- Setup the Plot ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # --- Plot the Data ---
        sns.histplot(self.data, ax=ax, bins=self.k_range, stat='count', 
                     alpha=0.6, label='Observed Overdispersed Data', color='#6d4c41')
        
        # --- Calculate and Plot the NB Curve ---
        variance = user_mu + user_alpha * user_mu**2
        p = user_mu / variance
        n = user_mu * p / (1 - p)
        
        nb_pmf = nbinom.pmf(self.k_range, n=n, p=p)
        expected_counts = nb_pmf * self.sample_size
        
        ax.plot(self.k_range, expected_counts, 'o--', color='#d95f02', 
                label=f'NB Curve (μ={user_mu:.2f}, α={user_alpha:.2f})')
        
        # --- Calculate and Display Metrics ---
        log_like, chi2 = self._calculate_metrics(user_mu, user_alpha)
        
        # Display the relationship between mean, variance, and alpha
        formula = (r'$Var(k) = \mu + \alpha\mu^2$' '\n' r'($\alpha$ is the dispersion parameter)')
        ax.text(0.95, 0.95, formula, transform=ax.transAxes, fontsize=16,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        metrics_text = (f'Goodness-of-Fit Metrics:\n'
                        f'Log-Likelihood: {log_like:.2f} (Higher is better)\n'
                        f'Chi-Squared (χ²): {chi2:.2f} (Lower is better)')
        ax.text(0.95, 0.78, metrics_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.5))
        
        # --- Final Touches ---
        ax.set_title('Interactive Negative Binomial Distribution Fit', fontsize=16)
        ax.set_xlabel('Count Value (k)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_xticks(self.k_range[::2])
        ax.legend(loc='upper left')
        ax.set_xlim(left=-0.5, right=self.max_k + 5)
        ax.set_ylim(bottom=0)
        plt.show()

    def run(self):
        """
        Public method to launch the interactive simulator with two sliders.
        """
        # Slider for the mean (mu)
        mu_slider = FloatSlider(
            value=1.0, min=0.1, max=max(20.0, self.true_mu + 5), step=0.1,
            description='μ (Mean):', continuous_update=False,
            readout_format='.1f', layout={'width': '600px'}
        )
        
        # Slider for the dispersion (alpha)
        alpha_slider = FloatSlider(
            value=0.1, min=0.01, max=max(3.0, self.true_alpha + 1), step=0.05,
            description='α (Dispersion):', continuous_update=False,
            readout_format='.2f', layout={'width': '600px'}
        )
        
        interact(self._plot_distribution, user_mu=mu_slider, user_alpha=alpha_slider)

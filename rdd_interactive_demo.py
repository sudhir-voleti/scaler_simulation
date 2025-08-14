#
# rdd_interactive_demo.py
#
# This script creates an interactive tool for demonstrating the principles of
# a Sharp Regression Discontinuity Design (RDD).
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from ipywidgets import interact, FloatSlider, IntSlider
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class RDDSimulator:
    """
    A class to create an interactive simulator for understanding the
    mechanics of a Regression Discontinuity Design.
    """

    def __init__(self, cutoff=1000, running_var_range=(500, 1500)):
        """
        Initializes the simulator with fixed parameters.
        
        Args:
            cutoff (int): The fixed, known cutoff point for the treatment.
            running_var_range (tuple): The min and max range for the running variable.
        """
        self.cutoff = cutoff
        self.running_var_range = running_var_range

    def _generate_data(self, treatment_effect, sample_size, noise, slope_before, slope_after):
        """Generates sample data based on the user-controlled 'true' world."""
        # 1. Create the running variable
        running_variable = np.random.uniform(self.running_var_range[0], self.running_var_range[1], sample_size)
        
        # 2. Define the true underlying relationship (without noise)
        true_outcome = np.zeros(sample_size)
        
        # Apply different slopes before and after the cutoff
        before_cutoff_indices = np.where(running_variable < self.cutoff)
        after_cutoff_indices = np.where(running_variable >= self.cutoff)
        
        true_outcome[before_cutoff_indices] = 10 + slope_before * (running_variable[before_cutoff_indices] - self.running_var_range[0])
        true_outcome[after_cutoff_indices] = 10 + slope_after * (running_variable[after_cutoff_indices] - self.running_var_range[0])
        
        # 3. Assign treatment based on the sharp cutoff rule
        treated = (running_variable >= self.cutoff).astype(int)
        
        # 4. Add the true treatment effect to the treated group
        true_outcome += treated * treatment_effect
        
        # 5. Add random noise to create the observed outcome
        observed_outcome = true_outcome + np.random.normal(0, noise, sample_size)
        
        # 6. Create a DataFrame
        df = pd.DataFrame({
            'running_variable': running_variable,
            'outcome': observed_outcome,
            'treated': treated
        })
        return df

    def _plot_rdd(self, treatment_effect, sample_size, noise, slope_before, slope_after):
        """The core plotting function linked to interactive sliders."""
        
        # --- Generate data based on slider inputs ---
        df = self._generate_data(treatment_effect, sample_size, noise, slope_before, slope_after)
        
        # --- Run the RDD regression model on the generated data ---
        # This is the same model from your notebook, but run on our simulated data
        df['distance_to_cutoff'] = df['running_variable'] - self.cutoff
        df['interaction'] = df['distance_to_cutoff'] * df['treated']
        formula = 'outcome ~ treated + distance_to_cutoff + interaction'
        rdd_results = smf.ols(formula, data=df).fit()
        
        # Get the key result: the estimated treatment effect and its p-value
        estimated_effect = rdd_results.params['treated']
        p_value = rdd_results.pvalues['treated']
        
        # --- Plotting ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot of the generated data
        sns.scatterplot(data=df, x='running_variable', y='outcome', hue='treated', 
                        palette=['#007acc', '#d95f02'], alpha=0.7, ax=ax)
        
        # Plot the estimated regression lines
        sns.regplot(data=df[df['treated']==0], x='running_variable', y='outcome', ax=ax,
                    color='#007acc', scatter=False, ci=None, label='Fit (Untreated)')
        sns.regplot(data=df[df['treated']==1], x='running_variable', y='outcome', ax=ax,
                    color='#d95f02', scatter=False, ci=None, label='Fit (Treated)')
        
        # The crucial cutoff line
        ax.axvline(x=self.cutoff, color='red', linestyle='--', lw=2, label=f'Cutoff = {self.cutoff}')

        # --- Display Information ---
        true_params_text = (f'True Parameters (Set by you):\n'
                            f'  - True Effect: {treatment_effect:.2f}\n'
                            f'  - Noise (Std Dev): {noise:.2f}\n'
                            f'  - Sample Size: {sample_size}')
        ax.text(0.02, 0.98, true_params_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

        estimated_params_text = (f'RDD Regression Results:\n'
                                 f'  - Estimated Effect: {estimated_effect:.2f}\n'
                                 f'  - P-value: {p_value:.3f}')
        ax.text(0.98, 0.98, estimated_params_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle='round', fc='lightgreen' if p_value < 0.05 else 'lightcoral', alpha=0.5))

        # --- Final Touches ---
        ax.set_title('Interactive Regression Discontinuity (RDD) Simulator', fontsize=16)
        ax.set_xlabel('Running Variable (e.g., Test Score)', fontsize=12)
        ax.set_ylabel('Outcome Variable (e.g., Success Metric)', fontsize=12)
        ax.legend(loc='lower right')
        plt.show()

    def run(self):
        """Public method to launch the interactive simulator."""
        interact(self._plot_rdd,
                 treatment_effect=FloatSlider(value=10, min=-20, max=20, step=1, description='True Effect Size:', layout={'width': '80%'}, continuous_update=False),
                 sample_size=IntSlider(value=500, min=50, max=2000, step=50, description='Sample Size:', layout={'width': '80%'}, continuous_update=False),
                 noise=FloatSlider(value=8, min=0.1, max=30, step=0.5, description='Data Noise Level:', layout={'width': '80%'}, continuous_update=False),
                 slope_before=FloatSlider(value=0.02, min=-0.1, max=0.1, step=0.01, description='Slope (Before):', layout={'width': '80%'}, continuous_update=False),
                 slope_after=FloatSlider(value=0.02, min=-0.1, max=0.1, step=0.01, description='Slope (After):', layout={'width': '80%'}, continuous_update=False)
                )

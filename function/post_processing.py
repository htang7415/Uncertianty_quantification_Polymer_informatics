# post_processing.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def create_errorbar_plot(x, y, yerr, color, label, filename=None):
    plt.figure(figsize=(4.5, 4.0))
    plt.errorbar(x, y, 
                 yerr=yerr, 
                 fmt='o', ecolor=f'light{color}', mec=color, mfc=f'light{color}', 
                 alpha=0.7, capsize=5, label=f'{label} Prediction')
    plt.plot(x, x, 'r--', label='Perfect predictions')
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()

def plot_abs_error_vs_std(abs_error, std, set_name, color):
    plt.figure(figsize=(4.5, 4.0))
    plt.scatter(abs_error, std, alpha=0.5, color=color)
    plt.xlabel('Absolute Error', fontsize=14)
    plt.ylabel('Standard Deviation', fontsize=14)
    plt.title(f'{set_name} Set: Absolute Error vs. Std Dev', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_std_histogram(std_data, set_name, color, fontsize=14):
    plt.figure(figsize=(4.5, 4.0))
    n_bins = int(np.ceil(np.log2(len(std_data))) + 1)
    plt.hist(std_data, bins=n_bins, color=color, alpha=0.7, edgecolor='black')
    plt.xlabel('Standard Deviation', fontsize=fontsize)
    plt.ylabel('Count', fontsize=fontsize)
    plt.title(f'{set_name} Set Standard Deviation Distribution', fontsize=fontsize+2)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.tight_layout()
    plt.show()

def plot_calibration_curve(confidence_levels, observed_confidence, label):
    plt.figure(figsize=(4.5, 4.0))
    plt.plot(confidence_levels, observed_confidence, marker='o', label=f'Calibration curve ({label})')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Expected confidence')
    plt.ylabel('Observed confidence')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    # plt.grid(True)
    plt.show()
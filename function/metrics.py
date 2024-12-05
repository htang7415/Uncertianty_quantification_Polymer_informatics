# metrics.py
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
import numpy as np
import scipy.stats

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calculate_spearman(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

def calculate_spearman(y_true, y_pred, std_dev):
    # Calculate absolute errors
    abs_errors = np.abs(y_true - y_pred)
    
    # Calculate Spearman correlation between absolute errors and standard deviations
    correlation, _ = spearmanr(abs_errors, std_dev)
    return correlation if not np.isnan(correlation) else 0.0

def calculate_observed_confidence(y_true, mean_pred, std_pred, confidence_levels):
    z_scores = scipy.stats.norm.ppf((1 + confidence_levels) / 2)
    lower_bound = mean_pred[:, None] - z_scores * std_pred[:, None] / 2
    upper_bound = mean_pred[:, None] + z_scores * std_pred[:, None] / 2
    return np.mean((y_true[:, None] >= lower_bound) & (y_true[:, None] <= upper_bound), axis=0)

def calculate_calibration_area(expected_confidence, observed_confidence):
    # Calculate the difference between observed and expected confidence
    difference = np.abs(observed_confidence - expected_confidence)
    
    # Use numpy's trapz function to calculate the area
    area = np.trapz(difference, expected_confidence)
    
    return area

def calculate_metrics(y_true, y_pred, std_dev=None, confidence_levels=None):
    # Reshape y_true to match y_pred and std_dev
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    if std_dev is not None:
        std_dev = np.array(std_dev).flatten()
    
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    
    # Update Spearman calculation to use std_dev
    spearman = calculate_spearman(y_true, y_pred, std_dev) if std_dev is not None else None
    
    observed_confidence = calculate_observed_confidence(y_true, y_pred, std_dev, confidence_levels)
    calibration_area = calculate_calibration_area(confidence_levels, observed_confidence)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Spearman': spearman,
        'Calibration Area': calibration_area
    }
        
    return metrics
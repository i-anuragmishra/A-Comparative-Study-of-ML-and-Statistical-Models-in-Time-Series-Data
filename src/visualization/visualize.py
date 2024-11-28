"""Visualization utilities for time series analysis.

Provides functions for creating various plots and visualizations
for time series data and model results.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_time_series(data, title='Time Series Plot'):
    """Plot time series data."""
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    return plt

def plot_model_comparison(actual, predictions_dict):
    """Plot actual vs predicted values for multiple models."""
    plt.figure(figsize=(15, 8))
    plt.plot(actual, label='Actual', color='black')
    
    colors = sns.color_palette('husl', len(predictions_dict))
    for (name, pred), color in zip(predictions_dict.items(), colors):
        plt.plot(pred, label=f'Predicted ({name})', color=color)
    
    plt.title('Model Predictions Comparison')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    return plt

def plot_error_distribution(errors, model_name):
    """Plot error distribution for a model."""
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title(f'Error Distribution - {model_name}')
    plt.xlabel('Error')
    plt.ylabel('Count')
    return plt

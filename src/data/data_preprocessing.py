"""Data preprocessing module for climate time series data.

This module handles the preprocessing of raw climate and pollution data,
including cleaning, normalization, and feature engineering.
"""

import pandas as pd
import numpy as np

def load_data(filepath):
    """Load raw data from specified path."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean the dataset by handling missing values and outliers."""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.interpolate(method='linear')
    
    return df

def feature_engineering(df):
    """Create relevant features for time series analysis."""
    # Add time-based features
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['day'] = pd.DatetimeIndex(df['date']).day
    
    # Add rolling averages
    df['rolling_mean'] = df['value'].rolling(window=7).mean()
    
    return df

def normalize_data(df):
    """Normalize the features using MinMax scaling."""
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    # Normalize numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, scaler

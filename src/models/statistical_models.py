"""Statistical models implementation for time series analysis.

Implements various statistical models including ARIMA, SARIMA,
and regression models for time series prediction.
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

class TimeSeriesModel:
    """Base class for time series models."""
    
    def __init__(self):
        self.model = None
    
    def train(self, data):
        """Train the model with given data."""
        raise NotImplementedError
    
    def predict(self, steps):
        """Make predictions for specified steps."""
        raise NotImplementedError

class ARIMAModel(TimeSeriesModel):
    """ARIMA model implementation."""
    
    def __init__(self, order=(1,1,1)):
        super().__init__()
        self.order = order
    
    def train(self, data):
        """Train ARIMA model."""
        self.model = ARIMA(data, order=self.order)
        self.fitted = self.model.fit()
        return self.fitted
    
    def predict(self, steps):
        """Generate predictions."""
        return self.fitted.forecast(steps)

class SARIMAModel(TimeSeriesModel):
    """SARIMA model implementation."""
    
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
    
    def train(self, data):
        """Train SARIMA model."""
        self.model = SARIMAX(data,
                           order=self.order,
                           seasonal_order=self.seasonal_order)
        self.fitted = self.model.fit()
        return self.fitted
    
    def predict(self, steps):
        """Generate predictions."""
        return self.fitted.forecast(steps)

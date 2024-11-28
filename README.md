# A Comparative Study of ML and Statistical Models in Time Series Data

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Research Paper
[Analyzing the Impact of Climate Change With Major Emphasis on Pollution: A Comparative Study of ML and Statistical Models in Time Series Data](https://arxiv.org/abs/2405.15835)

## 🎯 Project Overview
This research project analyzes climate change impacts through time series data analysis, comparing traditional statistical approaches with modern machine learning methodologies.

### Quick Start
```bash
# Clone the repository
git clone https://github.com/i-anuragmishra/A-Comparative-Study-of-ML-and-Statistical-Models-in-Time-Series-Data.git

# Install dependencies
pip install -r requirements.txt

# Run experiments
python src/main.py
```

## 📁 Repository Structure
```
├── data/
│   ├── raw/              # Original, immutable data
│   ├── processed/         # Cleaned and processed data
│   └── external/          # External source data
├── docs/                  # Documentation
│   ├── paper/            # Research paper and related materials
│   └── figures/          # Figures and visualizations
├── models/               # Trained and serialized models
├── notebooks/            # Jupyter notebooks for exploration and demonstration
├── src/                  # Source code
│   ├── data/             # Data processing scripts
│   ├── features/         # Feature engineering code
│   ├── models/           # Model implementations
│   └── visualization/    # Visualization code
├── tests/                # Test cases
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## 🔍 Research Components

### Data Processing
- Time series data cleaning and preprocessing
- Feature engineering for climate metrics
- Data validation and quality checks

### Models Implemented
1. Statistical Models
   - ARIMA
   - SARIMA
   - Statistical Regression

2. Machine Learning Models
   - LSTM Networks
   - Random Forests
   - XGBoost

### Evaluation Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared Score

## 📊 Results
Detailed results and comparisons are available in:
- `/notebooks/analysis.ipynb`
- `/docs/results.md`

## 🛠️ Technologies Used
- Python 3.8+
- TensorFlow 2.x
- PyTorch
- scikit-learn
- pandas
- numpy

## 👥 Contributors
Research conducted by [Anurag Mishra](https://github.com/i-anuragmishra)

## 📚 Citation
```bibtex
@article{mishra2024analyzing,
  title={Analyzing the Impact of Climate Change With Major Emphasis on Pollution: A Comparative Study of ML and Statistical Models in Time Series Data},
  author={Mishra, Anurag},
  journal={arXiv preprint arXiv:2405.15835},
  year={2024}
}
```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
© 2024 All Rights Reserved
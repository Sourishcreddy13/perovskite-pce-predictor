# Perovskite Solar Cell PCE Prediction - ML Pipeline

A comprehensive machine learning pipeline for predicting Power Conversion Efficiency (PCE) of perovskite solar cells using composition, processing, and material data.

## Overview

This project implements an end-to-end machine learning workflow that:
- Cleans and validates perovskite solar cell data from the Perovskite Database
- Performs stoichiometric validation (ABX₃ structure)
- Engineers domain-specific features
- Benchmarks multiple ML algorithms
- Predicts PCE with interpretable models
- Generates comprehensive analysis reports

## Features

### Data Cleaning Pipeline (`clean_perovskite_data.py`)
- **Stoichiometry Validation**: Enforces ABX₃ structure with configurable tolerances
- **Ion Filtering**: 
  - A-site: Cs, FA (Formamidinium), MA (Methylammonium)
  - B-site: Pb (Lead)
  - C-site: Cl, Br, I (Halides)
- **Data Deduplication**: Removes duplicate deposition procedures
- **Unknown Value Filtering**: Regex-based removal of placeholder values
- **Stack Sequence Extraction**: Extracts primary materials from multi-layer stacks
- **Categorical Encoding**: Ordinal encoding for LazyPredict compatibility

### ML Pipeline (`perovskite_ml_pipeline.py`)
- **Data Quality Assessment**: Comprehensive missing data analysis
- **Multi-Model Benchmarking**: Tests 8+ regression algorithms
- **Cross-Validation**: 5-fold CV for model stability assessment
- **Feature Importance Analysis**: Random Forest-based importance ranking
- **Performance Visualization**: Automated plot generation
- **Model Persistence**: Saves trained models and preprocessing objects

### Results Exploration (`explore_results.py`)
- Interactive data exploration
- Composition analysis (A-site, B-site, X-site distributions)
- PCE statistics and percentiles
- Model performance summaries
- Feature importance visualization

### Insight Generation (`generate_insights.py`)
- Comprehensive 9-panel visualization dashboard
- PCE distribution analysis
- Composition-based performance comparisons
- Correlation heatmaps
- Feature engineering insights

## Installation

### Prerequisites

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pickle
- joblib

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd perovskite-ml-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Python Packages

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
simplejson>=3.18.0
```

## Usage

### Step 1: Data Cleaning

Clean the raw perovskite dataset:

```bash
python clean_perovskite_data.py
```

**Input**: `pvmaster.csv` (raw data from Perovskite Database)  
**Outputs**:
- `pvmaster_cleaned.csv` - Cleaned and encoded dataset
- `encoding_map.json` - Categorical encoding mappings

**Configuration** (in script):
```python
# Stoichiometry tolerances
TOLERANCE_A = 0.05  # A-site sum ≈ 1.0
TOLERANCE_B = 0.05  # B-site sum ≈ 1.0
TOLERANCE_C = 0.1   # C-site sum ≈ 3.0
```

### Step 2: Run ML Pipeline

Execute the complete machine learning workflow:

```bash
python perovskite_ml_pipeline.py
```

**Input**: `pvmaster_cleaned.csv`  
**Outputs** (15+ files):
```
01_initial_data.csv                    # Initial validated data
02_data_quality_report.csv             # Missing data analysis
02_data_quality_visualization.png      # Quality charts
03_preprocessing_objects.pkl           # Scalers and encoders
06_model_benchmark_results.csv         # Model comparison
06_model_benchmark_comparison.png      # Performance plots
07_cross_validation_results.csv        # CV metrics
07_cross_validation_comparison.png     # CV visualization
08_feature_importance.csv              # Feature rankings
08_feature_importance_top20.png        # Top features plot
08_best_model.pkl                      # Trained model
08_prediction_analysis.png             # Predictions vs actual
09_final_report.txt                    # Comprehensive report
```

### Step 3: Explore Results

Interactively explore the pipeline outputs:

```bash
python explore_results.py
```

Or use in Jupyter/IPython for interactive analysis:
```python
%run explore_results.py

# Access loaded objects
print(df_engineered.head())
print(feature_importance.head(10))
```

### Step 4: Generate Additional Insights

Create comprehensive visualization dashboard:

```bash
python generate_insights.py
```

**Outputs**:
- `10_comprehensive_insights.png` - 9-panel analysis dashboard
- `11_correlation_matrix.png` - Feature correlation heatmap

## Data Pipeline Details

### Cleaning Stages

1. **Column Selection**: Filters essential ABX₃ and process columns
2. **NaN Removal**: Drops rows with missing ion composition data
3. **B-site Filtering**: Keeps only Pb-based perovskites
4. **A-site Filtering**: Retains Cs/FA/MA compositions
5. **C-site Filtering**: Keeps Cl/Br/I halides
6. **Stoichiometry Validation**: Ensures ABX₃ structure with tolerances
7. **Deposition Deduplication**: Cleans procedure strings (e.g., "Spin | Spin" → "Spin")
8. **Unknown Value Removal**: Regex-based filtering of placeholders
9. **Stack Simplification**: Extracts primary layer from multi-layer stacks
10. **Feature Encoding**: Ordinal encoding for categorical variables
11. **Quality Checks**: Duplicate removal and validation

### Feature Engineering

Engineered features include:
- **Ion Presence Flags**: `has_MA`, `has_FA`, `has_Cs`, `has_Pb`, `has_I`, `has_Br`, `has_Cl`
- **Stack Counts**: Number of layers in ETL/HTL/substrate/backcontact
- **Composition Indicators**: Binary flags for specific ion combinations
- **Deposition Method Encodings**: Numerical representations of procedures

## Model Performance

### Target Metrics
- **RMSE**: < 2.0% (prediction error)
- **R²**: > 0.75 (variance explained)

### Benchmarked Models
1. Random Forest Regressor ⭐ (typically best)
2. Gradient Boosting Regressor
3. Extra Trees Regressor
4. Ridge Regression
5. Lasso Regression
6. ElasticNet
7. K-Nearest Neighbors
8. Decision Tree Regressor

### Typical Results
- **Best Model**: Random Forest
- **Test RMSE**: ~3.7%
- **Test R²**: ~0.50
- **Top Feature**: `has_FA` (Formamidinium presence)

## Output Files Reference

### CSV Files
| File | Description |
|------|-------------|
| `pvmaster_cleaned.csv` | Cleaned dataset after filtering |
| `01_initial_data.csv` | Validated initial data |
| `02_data_quality_report.csv` | Missing data analysis |
| `06_model_benchmark_results.csv` | Model comparison metrics |
| `07_cross_validation_results.csv` | CV performance |
| `08_feature_importance.csv` | Feature importance rankings |

### Pickle Files
| File | Description |
|------|-------------|
| `03_preprocessing_objects.pkl` | Scalers, imputers, encoders |
| `08_best_model.pkl` | Trained best model |

### Visualizations
| File | Description |
|------|-------------|
| `02_data_quality_visualization.png` | Missing data charts |
| `06_model_benchmark_comparison.png` | 4-panel model comparison |
| `07_cross_validation_comparison.png` | CV results bar chart |
| `08_feature_importance_top20.png` | Top 20 features |
| `08_prediction_analysis.png` | Predictions vs actual + residuals |
| `10_comprehensive_insights.png` | 9-panel dashboard |
| `11_correlation_matrix.png` | Feature correlation heatmap |

### Reports
| File | Description |
|------|-------------|
| `09_final_report.txt` | Comprehensive text report |
| `encoding_map.json` | Categorical encoding reference |

## Configuration

### Cleaning Pipeline Parameters

```python
# In clean_perovskite_data.py
TOLERANCE_A = 0.05  # A-site coefficient sum tolerance
TOLERANCE_B = 0.05  # B-site coefficient sum tolerance
TOLERANCE_C = 0.1   # C-site coefficient sum tolerance
```

### ML Pipeline Parameters

```python
# In perovskite_ml_pipeline.py
TARGET = 'JV_default_PCE'          # Target variable
TEST_SIZE = 0.2                    # Train/test split ratio
RANDOM_STATE = 42                  # Reproducibility seed
N_ESTIMATORS = 200                 # Tree-based model trees
MAX_DEPTH = 20                     # Random Forest depth
CV_FOLDS = 5                       # Cross-validation folds
```

## Making Predictions

### Load Trained Model

```python
import pickle
import pandas as pd
import numpy as np

# Load model and preprocessing
with open('08_best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('03_preprocessing_objects.pkl', 'rb') as f:
    preprocessing = pickle.load(f)

# Prepare new data
X_new = pd.DataFrame({...})  # Your new data
X_scaled = preprocessing['scaler'].transform(X_new)

# Predict
predictions = model.predict(X_scaled)
print(f"Predicted PCE: {predictions[0]:.2f}%")
```

## Data Quality Insights

### Typical Dataset Statistics (After Cleaning)
- **Initial Rows**: ~60,000+
- **Cleaned Rows**: ~42,000
- **Removal Rate**: ~30%
- **Features**: 8-15 (depends on encoding)

### Common Data Issues
- **Missing Values**: 20-70% in many columns
- **Unknown Placeholders**: "Unknown", "N/A", "nan"
- **Duplicate Procedures**: "Spin | Spin" patterns
- **Invalid Stoichiometry**: Non-ABX₃ structures

## Troubleshooting

### Issue: ImportError for simplejson
```bash
pip install simplejson
```

### Issue: Memory Error During Cross-Validation
- Reduce `n_estimators` in models (e.g., 100 instead of 200)
- Decrease `n_jobs` parameter (e.g., 4 instead of -1)
- Use smaller dataset subset for testing

### Issue: "pvmaster.csv not found"
- Download the dataset from [Perovskite Database](https://www.perovskitedatabase.com/)
- Place `pvmaster.csv` in the same directory as the scripts
- Or update `DATA_PATH` in the script

### Issue: Encoding Errors in CSV
```python
# In load functions, add:
pd.read_csv('pvmaster.csv', encoding='utf-8', low_memory=False)
```

### Issue: Joblib Parallel Processing Errors
The pipeline uses a custom temporary directory strategy to avoid `/dev/shm` issues:
```python
# This is already implemented in the code
temp_dir = tempfile.mkdtemp(prefix="joblib_cv_", dir=OUTPUT_DIR)
```

## Performance Optimization

### Speed Tips
1. **Use fewer CV folds**: Change from 5 to 3 for faster iteration
2. **Reduce n_estimators**: Use 50-100 for quick testing
3. **Sample data**: Test on 10% subset first
4. **Parallel processing**: Ensure `n_jobs=-1` for multi-core usage

### Memory Tips
1. **Use float32**: Convert features to float32 instead of float64
2. **Drop high-cardinality features**: Remove columns with >1000 unique values
3. **Chunk processing**: Process large datasets in batches

## Domain Knowledge Notes

### ABX₃ Perovskite Structure
- **A-site**: Organic/inorganic cation (Cs⁺, FA⁺, MA⁺)
- **B-site**: Metal cation (Pb²⁺, Sn²⁺)
- **X-site**: Halide anion (I⁻, Br⁻, Cl⁻)

### Typical High-Performance Compositions
- **FA-Pb-I**: Formamidinium lead iodide (high efficiency)
- **MA-Pb-I₃**: Methylammonium lead iodide (classic)
- **Mixed cations**: FA₀.₈₅Cs₀.₁₅Pb(I₀.₉Br₀.₁)₃ (stable + efficient)

### Deposition Methods
- **Spin-coating**: Most common for lab-scale
- **Evaporation**: Vacuum-based deposition
- **Solution casting**: Simple but less uniform
- **Blade coating**: Scalable manufacturing

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{perovskite_ml_pipeline,
  title={Perovskite Solar Cell PCE Prediction ML Pipeline},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
```

## License

[Your License Here - e.g., MIT, Apache 2.0]

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Acknowledgments

- **Perovskite Database Project**: Data source
- **scikit-learn**: ML framework
- **Community**: Feature engineering insights

## Support

For issues or questions:
- Open a GitHub issue
- Check existing documentation
- Review example outputs in `explore_results.py`

## Roadmap

### Future Enhancements
- [ ] Deep learning models (neural networks)
- [ ] SHAP value analysis for interpretability
- [ ] Hyperparameter optimization with Optuna
- [ ] Ensemble methods (stacking, blending)
- [ ] External feature integration (bandgap, stability data)
- [ ] Web-based prediction interface
- [ ] Automated reporting with PDF generation
- [ ] Real-time model monitoring dashboard

## Version History

- **v1.0.0** (2025-11-04): Initial release
  - Data cleaning pipeline
  - Multi-model benchmarking
  - Cross-validation
  - Feature importance analysis
  - Visualization suite

---

**Built with ❤️ for the perovskite solar cell research community**

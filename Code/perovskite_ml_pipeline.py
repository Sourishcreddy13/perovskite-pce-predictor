"""
Automated ML Pipeline for Perovskite Solar Cell PCE Prediction
Based on PRD v1.0 - November 04, 2025

This script performs:
1. Data loading from pre-cleaned dataset
2. Data quality assessment
3. Model benchmarking with LazyPredict
4. Advanced modeling with cross-validation
5. Model interpretation with SHAP
6. Report generation and visualization

All outputs are saved to the ghost/ghost_5 directory.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pickle
import tempfile
import shutil

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from joblib import parallel_backend
from joblib.externals.loky import get_reusable_executor

warnings.filterwarnings('ignore')
np.random.seed(42)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# CONFIGURATION
# ============================================================================

# The dataset is already cleaned and encoded, no need to select columns
# All features are numeric and ready for ML

TARGET = 'JV_default_PCE'
OUTPUT_DIR = '.'  # Current directory (ghost/ghost_5)
DATA_PATH = 'pvmaster_cleaned.csv'  # Pre-cleaned and encoded dataset

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def save_figure(fig, filename):
    """Save figure to output directory"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    log(f"Saved figure: {filename}")
    plt.close(fig)

def save_dataframe(df, filename):
    """Save dataframe to output directory"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    log(f"Saved dataframe: {filename}")

# ============================================================================
# STEP 1: DATA LOADING & VALIDATION
# ============================================================================

def load_and_validate_data():
    """Load pre-cleaned data and perform initial validation"""
    log("=" * 80)
    log("STEP 1: DATA LOADING & VALIDATION")
    log("=" * 80)
    
    # Load pre-cleaned data
    log(f"Loading pre-cleaned data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    log(f"Dataset shape: {df.shape}")
    log(f"Columns: {list(df.columns)}")
    
    # Validate target variable
    log(f"\nValidating target variable: {TARGET}")
    log(f"Original rows: {len(df)}")
    
    # Convert target to numeric (should already be numeric, but ensure)
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    
    # Remove null and non-positive PCE values
    initial_rows = len(df)
    df = df[df[TARGET].notna()]
    df = df[df[TARGET] > 0]
    
    log(f"Rows after removing null/zero PCE: {len(df)} (removed {initial_rows - len(df)})")
    log(f"PCE statistics:\n{df[TARGET].describe()}")
    
    # Save initial dataset
    save_dataframe(df, '01_initial_data.csv')
    
    return df


# ============================================================================
# STEP 2: DATA QUALITY ASSESSMENT
# ============================================================================

def assess_data_quality(df):
    """Generate comprehensive data quality report"""
    log("\n" + "=" * 80)
    log("STEP 2: DATA QUALITY ASSESSMENT")
    log("=" * 80)
    
    quality_report = []
    
    for col in df.columns:
        if col == TARGET:
            continue
            
        total = len(df)
        missing = df[col].isna().sum()
        missing_pct = (missing / total) * 100
        
        # Count 'Unknown' and similar entries
        unknown_count = 0
        if df[col].dtype == 'object':
            unknown_patterns = ['unknown', 'nan', 'none', 'n/a', '']
            for pattern in unknown_patterns:
                unknown_count += df[col].astype(str).str.lower().str.strip().eq(pattern).sum()
        
        unknown_pct = (unknown_count / total) * 100
        total_missing_pct = missing_pct + unknown_pct
        
        unique_values = df[col].nunique()
        dtype = df[col].dtype
        
        quality_report.append({
            'Column': col,
            'Missing_Count': missing,
            'Missing_Pct': missing_pct,
            'Unknown_Count': unknown_count,
            'Unknown_Pct': unknown_pct,
            'Total_Missing_Pct': total_missing_pct,
            'Unique_Values': unique_values,
            'Data_Type': dtype,
            'Drop_Recommendation': 'DROP' if total_missing_pct > 70 or unique_values > 1000 else 'KEEP'
        })
    
    quality_df = pd.DataFrame(quality_report)
    quality_df = quality_df.sort_values('Total_Missing_Pct', ascending=False)
    
    log("\nData Quality Summary:")
    log(f"Columns with >70% missing: {(quality_df['Total_Missing_Pct'] > 70).sum()}")
    log(f"High cardinality columns (>100 unique): {(quality_df['Unique_Values'] > 100).sum()}")
    
    save_dataframe(quality_df, '02_data_quality_report.csv')
    
    # Visualize data quality
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Missing data percentage
    quality_df_top = quality_df.head(15)
    axes[0].barh(quality_df_top['Column'], quality_df_top['Total_Missing_Pct'])
    axes[0].set_xlabel('Missing/Unknown Percentage')
    axes[0].set_title('Top 15 Columns by Missing/Unknown Data')
    axes[0].axvline(x=70, color='r', linestyle='--', label='70% Threshold')
    axes[0].legend()
    
    # Unique values
    axes[1].barh(quality_df_top['Column'], quality_df_top['Unique_Values'])
    axes[1].set_xlabel('Number of Unique Values')
    axes[1].set_title('Top 15 Columns by Cardinality')
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    save_figure(fig, '02_data_quality_visualization.png')
    
    return quality_df

# ============================================================================
# STEP 3: PREPARE DATA FOR MODELING (Data is already cleaned and encoded)
# ============================================================================

def prepare_for_modeling(df):
    """Prepare train/test split - data is already cleaned and encoded"""
    log("\n" + "=" * 80)
    log("STEP 3: PREPARING DATA FOR MODELING")
    log("=" * 80)
    
    df_model = df.copy()
    
    # Separate features and target
    y = df_model[TARGET].values
    X = df_model.drop(columns=[TARGET])
    
    # Drop Ref_ID if present (not a feature)
    if 'Ref_ID' in X.columns:
        X = X.drop(columns=['Ref_ID'])
    
    log(f"Feature shape: {X.shape}")
    log(f"Target shape: {y.shape}")
    log(f"Features: {list(X.columns)}")
    
    # Data is already encoded, but check for any missing values
    log(f"\nMissing values per column:")
    missing = X.isnull().sum()
    if missing.sum() > 0:
        log(missing[missing > 0])
        # Simple imputation with median for any missing values
        log("\nImputing missing values with median...")
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns
        )
    else:
        log("No missing values - data is clean!")
        imputer = None
    
    # Train/test split
    log("\nSplitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    log(f"Train set size: {X_train.shape}")
    log(f"Test set size: {X_test.shape}")
    
    # Scale features
    log("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Save preprocessing objects
    with open(os.path.join(OUTPUT_DIR, '03_preprocessing_objects.pkl'), 'wb') as f:
        pickle.dump({
            'imputer': imputer,
            'scaler': scaler,
            'feature_names': X_train.columns.tolist()
        }, f)
    log("Saved preprocessing objects")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns.tolist()

# ============================================================================
# STEP 6: MODEL BENCHMARKING
# ============================================================================

def benchmark_models(X_train, X_test, y_train, y_test):
    """Benchmark multiple regression models"""
    log("\n" + "=" * 80)
    log("STEP 6: MODEL BENCHMARKING")
    log("=" * 80)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'ElasticNet': ElasticNet(random_state=42),
        'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        log(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results.append({
            'Model': name,
            'Train_RMSE': train_rmse,
            'Test_RMSE': test_rmse,
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Train_MAE': train_mae,
            'Test_MAE': test_mae,
            'Overfit_Score': train_r2 - test_r2
        })
        
        log(f"{name} - Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_RMSE')
    
    save_dataframe(results_df, '06_model_benchmark_results.csv')
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # RMSE comparison
    axes[0, 0].barh(results_df['Model'], results_df['Test_RMSE'], color='steelblue')
    axes[0, 0].set_xlabel('Test RMSE')
    axes[0, 0].set_title('Model Comparison - Test RMSE (Lower is Better)')
    axes[0, 0].axvline(x=2.0, color='r', linestyle='--', label='Target <2.0')
    axes[0, 0].legend()
    
    # R² comparison
    axes[0, 1].barh(results_df['Model'], results_df['Test_R2'], color='forestgreen')
    axes[0, 1].set_xlabel('Test R²')
    axes[0, 1].set_title('Model Comparison - Test R² (Higher is Better)')
    axes[0, 1].axvline(x=0.75, color='r', linestyle='--', label='Target >0.75')
    axes[0, 1].legend()
    
    # Train vs Test RMSE
    x = np.arange(len(results_df))
    width = 0.35
    axes[1, 0].bar(x - width/2, results_df['Train_RMSE'], width, label='Train', color='lightblue')
    axes[1, 0].bar(x + width/2, results_df['Test_RMSE'], width, label='Test', color='steelblue')
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Train vs Test RMSE')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
    axes[1, 0].legend()
    
    # Overfitting assessment
    axes[1, 1].barh(results_df['Model'], results_df['Overfit_Score'], 
                    color=['red' if x > 0.1 else 'green' for x in results_df['Overfit_Score']])
    axes[1, 1].set_xlabel('Overfit Score (Train R² - Test R²)')
    axes[1, 1].set_title('Overfitting Assessment (Lower is Better)')
    axes[1, 1].axvline(x=0.1, color='orange', linestyle='--', label='Threshold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    save_figure(fig, '06_model_benchmark_comparison.png')
    
    return results_df

# ============================================================================
# STEP 7: ADVANCED MODELING WITH CROSS-VALIDATION
# ============================================================================

def advanced_modeling_with_cv(X_train, y_train, feature_names):
    """Perform cross-validation on best models"""
    log("\n" + "=" * 80)
    log("STEP 7: ADVANCED MODELING WITH CROSS-VALIDATION")
    log("=" * 80)
    
    # Best models based on benchmark
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20, 
                                               min_samples_split=5, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                        learning_rate=0.1, random_state=42),
        'Extra Trees': ExtraTreesRegressor(n_estimators=200, max_depth=20,
                                           min_samples_split=5, random_state=42, n_jobs=-1)
    }
    
    cv_results = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        log(f"\nCross-validating {name}...")

        # Use a dedicated temporary folder so joblib does not rely on /dev/shm
        temp_dir = tempfile.mkdtemp(prefix="joblib_cv_", dir=OUTPUT_DIR)
        try:
            with parallel_backend('loky', temp_folder=temp_dir, inner_max_num_threads=1):
                cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=kfold,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1,
                )
            cv_scores = -cv_scores  # Convert to positive RMSE
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            # Explicitly clear the reusable executor cache to avoid stale workers
            get_reusable_executor().shutdown(wait=True)

        cv_results.append({
            'Model': name,
            'CV_Mean_RMSE': cv_scores.mean(),
            'CV_Std_RMSE': cv_scores.std(),
            'CV_Min_RMSE': cv_scores.min(),
            'CV_Max_RMSE': cv_scores.max()
        })
        
        log(f"{name} - CV RMSE: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    cv_results_df = pd.DataFrame(cv_results)
    save_dataframe(cv_results_df, '07_cross_validation_results.csv')
    
    # Visualize CV results
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(cv_results_df))
    ax.bar(x, cv_results_df['CV_Mean_RMSE'], yerr=cv_results_df['CV_Std_RMSE'],
           capsize=10, color='steelblue', alpha=0.7)
    ax.set_xlabel('Model')
    ax.set_ylabel('RMSE')
    ax.set_title('Cross-Validation Results (Mean ± Std)')
    ax.set_xticks(x)
    ax.set_xticklabels(cv_results_df['Model'])
    ax.axhline(y=2.0, color='r', linestyle='--', label='Target RMSE <2.0')
    ax.legend()
    
    plt.tight_layout()
    save_figure(fig, '07_cross_validation_comparison.png')
    
    return cv_results_df

# ============================================================================
# STEP 8: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(X_train, X_test, y_train, y_test, feature_names):
    """Analyze feature importance using best model"""
    log("\n" + "=" * 80)
    log("STEP 8: FEATURE IMPORTANCE ANALYSIS")
    log("=" * 80)
    
    # Train best model (Random Forest typically best)
    log("Training Random Forest for feature importance...")
    model = RandomForestRegressor(n_estimators=200, max_depth=20, 
                                  min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Get feature importance
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    save_dataframe(importance_df, '08_feature_importance.csv')
    
    # Visualize top 20 features
    fig, ax = plt.subplots(figsize=(12, 10))
    top_20 = importance_df.head(20)
    ax.barh(top_20['Feature'], top_20['Importance'], color='forestgreen')
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 20 Most Important Features (Random Forest)')
    ax.invert_yaxis()
    
    plt.tight_layout()
    save_figure(fig, '08_feature_importance_top20.png')
    
    # Save trained model
    with open(os.path.join(OUTPUT_DIR, '08_best_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    log("Saved best model")
    
    # Generate predictions for analysis
    y_pred_test = model.predict(X_test)
    
    # Prediction vs Actual plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot
    axes[0].scatter(y_test, y_pred_test, alpha=0.5, s=10)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual PCE (%)')
    axes[0].set_ylabel('Predicted PCE (%)')
    axes[0].set_title(f'Prediction vs Actual (R² = {r2_score(y_test, y_pred_test):.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_test - y_pred_test
    axes[1].scatter(y_pred_test, residuals, alpha=0.5, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted PCE (%)')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, '08_prediction_analysis.png')
    
    return importance_df, model

# ============================================================================
# STEP 9: GENERATE FINAL REPORT
# ============================================================================

def generate_final_report(df_original, results_df, cv_results_df, importance_df):
    """Generate comprehensive final report"""
    log("\n" + "=" * 80)
    log("STEP 9: GENERATING FINAL REPORT")
    log("=" * 80)
    
    report = []
    report.append("=" * 80)
    report.append("PEROVSKITE SOLAR CELL PCE PREDICTION - ML PIPELINE REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Dataset summary
    report.append("1. DATASET SUMMARY")
    report.append("-" * 80)
    report.append(f"Total records processed: {len(df_original)}")
    report.append(f"Target variable: JV_default_PCE (Power Conversion Efficiency)")
    report.append(f"PCE Range: {df_original['JV_default_PCE'].min():.2f}% - {df_original['JV_default_PCE'].max():.2f}%")
    report.append(f"PCE Mean: {df_original['JV_default_PCE'].mean():.2f}% (±{df_original['JV_default_PCE'].std():.2f}%)")
    report.append("")
    
    # Model performance
    report.append("2. MODEL PERFORMANCE SUMMARY")
    report.append("-" * 80)
    best_model = results_df.iloc[0]
    report.append(f"Best Model: {best_model['Model']}")
    report.append(f"Test RMSE: {best_model['Test_RMSE']:.4f}%")
    report.append(f"Test R²: {best_model['Test_R2']:.4f}")
    report.append(f"Test MAE: {best_model['Test_MAE']:.4f}%")
    
    if best_model['Test_RMSE'] < 2.0:
        report.append("✓ SUCCESS: Achieved target RMSE <2.0%")
    else:
        report.append(f"⚠ WARNING: RMSE {best_model['Test_RMSE']:.4f}% exceeds target of 2.0%")
    
    if best_model['Test_R2'] > 0.75:
        report.append("✓ SUCCESS: Achieved target R² >0.75")
    else:
        report.append(f"⚠ WARNING: R² {best_model['Test_R2']:.4f} below target of 0.75")
    report.append("")
    
    # Cross-validation
    report.append("3. CROSS-VALIDATION RESULTS")
    report.append("-" * 80)
    for _, row in cv_results_df.iterrows():
        report.append(f"{row['Model']}: {row['CV_Mean_RMSE']:.4f} (±{row['CV_Std_RMSE']:.4f})")
    report.append("")
    
    # Top features
    report.append("4. TOP 10 PREDICTIVE FEATURES")
    report.append("-" * 80)
    for i, row in importance_df.head(10).iterrows():
        report.append(f"{row['Feature']}: {row['Importance']:.4f}")
    report.append("")
    
    # Recommendations
    report.append("5. RECOMMENDATIONS")
    report.append("-" * 80)
    report.append("• Focus on optimizing top 10 features for PCE improvement")
    report.append("• Collect more data on underrepresented perovskite compositions")
    report.append("• Consider ensemble methods combining top 3 models")
    report.append("• Implement continuous monitoring of model performance")
    report.append("• Validate predictions with experimental data")
    report.append("")
    
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    with open(os.path.join(OUTPUT_DIR, '09_final_report.txt'), 'w') as f:
        f.write(report_text)
    
    log("Final report generated")
    print("\n" + report_text)
    
    return report_text

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute complete ML pipeline"""
    log("=" * 80)
    log("PEROVSKITE ML PIPELINE - STARTING")
    log("=" * 80)
    log(f"Output directory: {OUTPUT_DIR}")
    
    try:
        # Step 1: Load pre-cleaned data
        df = load_and_validate_data()
        
        # Step 2: Assess quality
        quality_df = assess_data_quality(df)
        
        # Step 3: Prepare for modeling (data is already clean and encoded)
        X_train, X_test, y_train, y_test, feature_names = prepare_for_modeling(df)
        
        # Step 4: Benchmark models with LazyPredict
        results_df = benchmark_models(X_train, X_test, y_train, y_test)
        
        # Step 5: Cross-validation on best models
        cv_results_df = advanced_modeling_with_cv(X_train, y_train, feature_names)
        
        # Step 6: Feature importance analysis
        importance_df, best_model = analyze_feature_importance(
            X_train, X_test, y_train, y_test, feature_names
        )
        
        # Step 7: Final report
        report = generate_final_report(df, results_df, cv_results_df, importance_df)
        
        log("\n" + "=" * 80)
        log("PIPELINE COMPLETED SUCCESSFULLY!")
        log("=" * 80)
        log(f"All results saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        log(f"\n ERROR: Pipeline failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

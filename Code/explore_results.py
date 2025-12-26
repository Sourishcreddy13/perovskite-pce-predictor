"""
Interactive Results Explorer for Perovskite ML Pipeline
Load and explore the generated artifacts from the ML pipeline.
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("PEROVSKITE ML PIPELINE - RESULTS EXPLORER")
print("=" * 80)

# Load datasets
print("\n📊 Loading datasets...")
df_initial = pd.read_csv('01_initial_filtered_data.csv')
df_clean = pd.read_csv('03_cleaned_data.csv')
df_engineered = pd.read_csv('04_feature_engineered_data.csv')
quality_report = pd.read_csv('02_data_quality_report.csv')
benchmark_results = pd.read_csv('06_model_benchmark_results.csv')
cv_results = pd.read_csv('07_cross_validation_results.csv')
feature_importance = pd.read_csv('08_feature_importance.csv')

print(f"✓ Loaded 7 datasets")

# Load model and preprocessing
print("\n🤖 Loading model artifacts...")
with open('08_best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('05_preprocessing_objects.pkl', 'rb') as f:
    preprocessing = pickle.load(f)

print(f"✓ Loaded Random Forest model")
print(f"✓ Loaded preprocessing objects")

print("\n" + "=" * 80)
print("DATASET STATISTICS")
print("=" * 80)

print(f"\nInitial Dataset:")
print(f"  Rows: {len(df_initial):,}")
print(f"  Columns: {len(df_initial.columns)}")

print(f"\nCleaned Dataset:")
print(f"  Rows: {len(df_clean):,}")
print(f"  Columns: {len(df_clean.columns)}")
print(f"  Rows removed: {len(df_initial) - len(df_clean):,}")

print(f"\nEngineered Dataset:")
print(f"  Rows: {len(df_engineered):,}")
print(f"  Columns: {len(df_engineered.columns)}")
print(f"  Features added: {len(df_engineered.columns) - len(df_clean.columns)}")

print("\n" + "=" * 80)
print("PCE DISTRIBUTION ANALYSIS")
print("=" * 80)

pce = df_engineered['JV_default_PCE']
print(f"\nPower Conversion Efficiency Statistics:")
print(f"  Mean:   {pce.mean():.3f}%")
print(f"  Median: {pce.median():.3f}%")
print(f"  Std:    {pce.std():.3f}%")
print(f"  Min:    {pce.min():.3f}%")
print(f"  Max:    {pce.max():.3f}%")
print(f"\nPercentiles:")
print(f"  25th:   {pce.quantile(0.25):.3f}%")
print(f"  50th:   {pce.quantile(0.50):.3f}%")
print(f"  75th:   {pce.quantile(0.75):.3f}%")
print(f"  90th:   {pce.quantile(0.90):.3f}%")
print(f"  95th:   {pce.quantile(0.95):.3f}%")
print(f"  99th:   {pce.quantile(0.99):.3f}%")

print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)

print("\nTop 3 Models by Test RMSE:")
top_3 = benchmark_results.nsmallest(3, 'Test_RMSE')
for idx, row in top_3.iterrows():
    print(f"\n{idx+1}. {row['Model']}")
    print(f"   Test RMSE: {row['Test_RMSE']:.4f}%")
    print(f"   Test R²:   {row['Test_R2']:.4f}")
    print(f"   Test MAE:  {row['Test_MAE']:.4f}%")
    print(f"   Overfit:   {row['Overfit_Score']:.4f}")

print("\n" + "=" * 80)
print("CROSS-VALIDATION ANALYSIS")
print("=" * 80)

for idx, row in cv_results.iterrows():
    print(f"\n{row['Model']}:")
    print(f"  Mean RMSE: {row['CV_Mean_RMSE']:.4f} ± {row['CV_Std_RMSE']:.4f}")
    print(f"  Range: [{row['CV_Min_RMSE']:.4f}, {row['CV_Max_RMSE']:.4f}]")
    print(f"  Stability: {'Excellent' if row['CV_Std_RMSE'] < 0.05 else 'Good' if row['CV_Std_RMSE'] < 0.1 else 'Fair'}")

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

print("\nTop 15 Most Important Features:")
for idx, row in feature_importance.head(15).iterrows():
    bar_length = int(row['Importance'] * 50)
    bar = '█' * bar_length
    print(f"{idx+1:2}. {row['Feature']:45} {row['Importance']:.4f} {bar}")

print("\n" + "=" * 80)
print("COMPOSITION ANALYSIS")
print("=" * 80)

if 'has_MA' in df_engineered.columns:
    print(f"\nA-site Ion Distribution:")
    print(f"  Methylammonium (MA): {df_engineered['has_MA'].sum():,} ({df_engineered['has_MA'].mean()*100:.1f}%)")
    print(f"  Formamidinium (FA):  {df_engineered['has_FA'].sum():,} ({df_engineered['has_FA'].mean()*100:.1f}%)")
    print(f"  Cesium (Cs):         {df_engineered['has_Cs'].sum():,} ({df_engineered['has_Cs'].mean()*100:.1f}%)")

if 'has_Pb' in df_engineered.columns:
    print(f"\nB-site Ion Distribution:")
    print(f"  Lead (Pb):   {df_engineered['has_Pb'].sum():,} ({df_engineered['has_Pb'].mean()*100:.1f}%)")
    print(f"  Tin (Sn):    {df_engineered['has_Sn'].sum():,} ({df_engineered['has_Sn'].mean()*100:.1f}%)")

if 'has_I' in df_engineered.columns:
    print(f"\nX-site Ion Distribution:")
    print(f"  Iodide (I):    {df_engineered['has_I'].sum():,} ({df_engineered['has_I'].mean()*100:.1f}%)")
    print(f"  Bromide (Br):  {df_engineered['has_Br'].sum():,} ({df_engineered['has_Br'].mean()*100:.1f}%)")
    print(f"  Chloride (Cl): {df_engineered['has_Cl'].sum():,} ({df_engineered['has_Cl'].mean()*100:.1f}%)")

print("\n" + "=" * 80)
print("PCE BY COMPOSITION")
print("=" * 80)

if 'has_FA' in df_engineered.columns and 'has_Pb' in df_engineered.columns:
    print("\nMean PCE by Key Compositions:")
    
    compositions = [
        ('FA-Pb', (df_engineered['has_FA'] == 1) & (df_engineered['has_Pb'] == 1)),
        ('MA-Pb', (df_engineered['has_MA'] == 1) & (df_engineered['has_Pb'] == 1)),
        ('Cs-Pb', (df_engineered['has_Cs'] == 1) & (df_engineered['has_Pb'] == 1)),
        ('FA-I', (df_engineered['has_FA'] == 1) & (df_engineered['has_I'] == 1)),
        ('MA-I', (df_engineered['has_MA'] == 1) & (df_engineered['has_I'] == 1)),
    ]
    
    for name, mask in compositions:
        if mask.sum() > 0:
            mean_pce = df_engineered.loc[mask, 'JV_default_PCE'].mean()
            count = mask.sum()
            print(f"  {name:10} {mean_pce:6.2f}%  (n={count:,})")

print("\n" + "=" * 80)
print("DATA QUALITY INSIGHTS")
print("=" * 80)

print("\nColumns Dropped (>70% missing):")
dropped = quality_report[quality_report['Drop_Recommendation'] == 'DROP']
for idx, row in dropped.iterrows():
    print(f"  - {row['Column']}: {row['Total_Missing_Pct']:.1f}% missing")

print(f"\nColumns Kept: {len(quality_report[quality_report['Drop_Recommendation'] == 'KEEP'])}")

print("\n" + "=" * 80)
print("MODEL INSIGHTS")
print("=" * 80)

best_model_name = benchmark_results.iloc[0]['Model']
best_rmse = benchmark_results.iloc[0]['Test_RMSE']
best_r2 = benchmark_results.iloc[0]['Test_R2']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"\nKey Metrics:")
print(f"  Test RMSE: {best_rmse:.4f}%")
print(f"  Test R²:   {best_r2:.4f}")

if best_rmse < 2.0:
    print(f"  ✓ RMSE meets target (<2.0%)")
else:
    print(f"  ⚠ RMSE above target (target: <2.0%)")

if best_r2 > 0.75:
    print(f"  ✓ R² meets target (>0.75)")
else:
    print(f"  ⚠ R² below target (target: >0.75)")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("\n1. Feature Engineering:")
print("   - Focus on FA, Pb compositions (highest importance)")
print("   - Optimize ETL/HTL deposition procedures")
print("   - Create interaction terms between A/B/X site ions")

print("\n2. Model Improvements:")
print("   - Try XGBoost/LightGBM with hyperparameter tuning")
print("   - Ensemble top 3 models (RF, Extra Trees, GB)")
print("   - Implement SHAP analysis for deeper insights")

print("\n3. Data Collection:")
print("   - Increase samples for rare compositions")
print("   - Add external features (bandgap, stability)")
print("   - Normalize by device architecture type")

print("\n" + "=" * 80)
print("AVAILABLE FUNCTIONS")
print("=" * 80)

print("\nLoaded objects (accessible in Python):")
print("  - df_initial: Initial filtered dataset")
print("  - df_clean: Cleaned dataset")
print("  - df_engineered: Feature engineered dataset")
print("  - quality_report: Data quality metrics")
print("  - benchmark_results: Model comparison results")
print("  - cv_results: Cross-validation results")
print("  - feature_importance: Feature importance rankings")
print("  - best_model: Trained Random Forest model")
print("  - preprocessing: Preprocessing objects (encoders, scaler, etc.)")

print("\nExample Usage:")
print("  # Make predictions on new data")
print("  # predictions = best_model.predict(X_scaled)")
print("  ")
print("  # Explore feature importance")
print("  # print(feature_importance.head(20))")
print("  ")
print("  # Analyze PCE distribution")
print("  # df_engineered['JV_default_PCE'].hist(bins=50)")

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
print("\nAll artifacts loaded and ready for interactive analysis!")
print("Use this script as a template or run interactively in Jupyter/IPython.")
print("\n")

#!/usr/bin/env python3
"""
Perovskite Data Cleaning Pipeline
==================================
This script cleans the perovskite solar cell dataset by:
1. Filtering ion compositions (A-site: Cs/FA/MA, B-site: Pb, C-site: Cl/Br/I)
2. Validating stoichiometry (ABX3 structure with proper coefficient sums)
3. Deduplicating deposition procedure values (e.g., 'Spin-coating | Spin-coating' -> 'Spin-coating')
4. Removing 'Unknown' and empty values using regex
5. Extracting primary materials from stack sequences (e.g., 'TiO2-c | TiO2-mp' -> 'TiO2-c')
6. Encoding categorical features for machine learning
7. Selecting only relevant columns for LazyPredict modeling

Author: Data Cleaning Pipeline
Date: November 2025
"""

import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path
import json
from sklearn.preprocessing import LabelEncoder



# ============================================================================
# Configuration
# ============================================================================

# Columns needed for filtering (ABX3 validation)
COLUMNS_FOR_FILTERING = [
    'Ref_ID',
    'Perovskite_composition_a_ions',
    'Perovskite_composition_a_ions_coefficients',
    'Perovskite_composition_b_ions',
    'Perovskite_composition_b_ions_coefficients',
    'Perovskite_composition_c_ions',
    'Perovskite_composition_c_ions_coefficients',
    'Perovskite_composition_short_form',
    'Perovskite_composition_long_form',
    'Perovskite_deposition_procedure',
    'ETL_stack_sequence',
    'ETL_deposition_procedure',
    'HTL_stack_sequence',
    'HTL_deposition_procedure',
    'Backcontact_stack_sequence',
    'Backcontact_deposition_procedure',
    'Substrate_stack_sequence',
    'JV_default_PCE'
]

# Columns to keep in final cleaned dataset (after filtering)
# ABX3 ion columns and composition forms are dropped since they're only needed for filtering
COLUMNS_TO_KEEP_FINAL = [
    'Perovskite_deposition_procedure',
    'ETL_stack_sequence',
    'ETL_deposition_procedure',
    'HTL_stack_sequence',
    'HTL_deposition_procedure',
    'Backcontact_stack_sequence',
    'Backcontact_deposition_procedure',
    'Substrate_stack_sequence',
    'JV_default_PCE'
]

# Stoichiometry tolerances
TOLERANCE_A = 0.05  # A-site should sum to ~1.0
TOLERANCE_B = 0.05  # B-site should sum to ~1.0
TOLERANCE_C = 0.1   # C-site should sum to ~3.0


# ============================================================================
# Helper Functions
# ============================================================================

def sum_coefficients(coeff_string):
    """
    Sum coefficient strings like '0.1; 0.9' or '3'.
    Handles NaNs and non-numeric values gracefully.
    
    Args:
        coeff_string: String containing coefficients separated by semicolons
        
    Returns:
        float: Sum of coefficients, or np.nan if invalid
    """
    if pd.isna(coeff_string):
        return np.nan
    
    try:
        # Split by semicolon and sum numeric parts
        parts = str(coeff_string).split(';')
        numbers = [pd.to_numeric(part.strip(), errors='coerce') for part in parts]
        total = np.nansum(numbers)
        
        # Return NaN if all parts were invalid
        return total if not np.isnan(total) else np.nan
    except Exception:
        return np.nan


def is_unknown_or_empty(value):
    """
    Check if a value is 'Unknown', empty, or effectively meaningless.
    Uses regex to catch variations like 'unknown', 'UNKNOWN', 'N/A', etc.
    
    Args:
        value: The value to check
        
    Returns:
        bool: True if the value should be filtered out
    """
    if pd.isna(value):
        return True
    
    # Convert to string and strip whitespace
    str_value = str(value).strip()
    
    # Empty string
    if not str_value or str_value == '':
        return True
    
    # Match common placeholder patterns (case-insensitive)
    # Matches: unknown, Unknown, UNKNOWN, n/a, N/A, none, None, -, --, etc.
    unknown_pattern = r'^(unknown|n/?a|none|null|nan|-{1,}|\.{2,}|\?+)$'
    if re.match(unknown_pattern, str_value, re.IGNORECASE):
        return True
    
    return False


def extract_first_stack_material(value):
    """
    Extract ONLY the first material/element from stack sequences.
    
    For ETL and HTL stack sequences that contain multiple layers separated by '|',
    we only keep the first one (the primary/bulk layer).
    
    Examples:
        'TiO2-c | TiO2-mp' -> 'TiO2-c'
        'Nb2O5 | PCBM-60 | Bphen' -> 'Nb2O5'
        'Spiro-MeOTAD' -> 'Spiro-MeOTAD' (no pipe, return as-is)
        'PCBM' -> 'PCBM'
    
    Args:
        value: String with pipe-separated stack materials
        
    Returns:
        str: First material only, stripped of whitespace
    """
    if pd.isna(value):
        return value
    
    str_value = str(value).strip()
    
    if not str_value:
        return value
    
    # Split by pipe and take only the first element
    first_material = str_value.split('|')[0].strip()
    
    return first_material


def deduplicate_deposition_values(value):
    """
    Remove duplicate deposition methods in strings separated by '|' or '>>'.
    Examples:
        'Spin-coating | Spin-coating' -> 'Spin-coating'
        'Spin-coating | Spin-coating | Spin-coating | Evaporation' -> 'Spin-coating | Evaporation'
        'Evaporation | Evaporation' -> 'Evaporation'
        'Spin-coating >> Co-evaporation >> Evaporation >> Spin-coating' ->
            'Spin-coating >> Co-evaporation >> Evaporation'
    
    Args:
        value: String with pipe or arrow separated deposition methods
        
    Returns:
        str: Deduplicated string with each method appearing once in order
    """
    if pd.isna(value):
        return value

    str_value = str(value).strip()
    if not str_value:
        return value

    # Detect preferred joiner based on original formatting
    prefers_arrow = '>>' in str_value and '|' not in str_value

    # Split by either '|' or '>>', trimming whitespace around parts
    parts = [part.strip() for part in re.split(r'\s*(?:\|\s*|>>\s*)', str_value) if part.strip()]

    # Remove duplicates while preserving first occurrence order
    deduplicated = []
    seen = set()
    for part in parts:
        if part not in seen:
            deduplicated.append(part)
            seen.add(part)

    if not deduplicated:
        return ''

    joiner = ' >> ' if prefers_arrow else ' | '
    return joiner.join(deduplicated)


def filter_unknown_values(df, columns_to_check):
    """
    Filter out rows with 'Unknown' or empty values in specified columns.
    
    Args:
        df: DataFrame to filter
        columns_to_check: List of column names to check
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    initial_count = len(df)
    df_filtered = df.copy()
    
    for col in columns_to_check:
        if col in df_filtered.columns:
            # Create mask for valid (non-unknown) values
            valid_mask = ~df_filtered[col].apply(is_unknown_or_empty)
            rows_before = len(df_filtered)
            df_filtered = df_filtered[valid_mask]
            rows_removed = rows_before - len(df_filtered)
            
            if rows_removed > 0:
                print(f"  ↳ Removed {rows_removed} rows with Unknown/empty values in '{col}'")
    
    total_removed = initial_count - len(df_filtered)
    print(f"  Total rows removed due to Unknown/empty values: {total_removed}")
    
    return df_filtered

def encode_categorical_features(df, verbose=True):
    """
    Encode categorical features using OrdinalEncoder for LazyPredict compatibility.
    
    LazyPredict requires all features to be numeric, so we encode:
    - Composition forms (short and long)
    - Deposition procedures
    - Stack sequences
    
    Note: ABX3 ion columns have been dropped before this step as they were
    only needed for filtering.
    
    OrdinalEncoder is preferred over LabelEncoder because:
    - Handles multiple columns at once efficiently
    - Better handling of unknown values during prediction
    - Maintains consistent encoding across train/test splits
    
    Args:
        df: DataFrame with cleaned categorical data
        verbose: Print encoding details
        
    Returns:
        tuple: (encoded_df, encoding_map) - DataFrame with encoded features and encoding dictionary
    """
    df_encoded = df.copy()
    
    # Define categorical columns to encode
    categorical_columns = [
        'Perovskite_composition_short_form',
        'Perovskite_composition_long_form',
        'Perovskite_deposition_procedure',
        'ETL_stack_sequence',
        'ETL_deposition_procedure',
        'HTL_stack_sequence',
        'HTL_deposition_procedure',
        'Backcontact_stack_sequence',
        'Backcontact_deposition_procedure',
        'Substrate_stack_sequence'
    ]
    
    # Only encode columns that exist in the DataFrame
    columns_to_encode = [col for col in categorical_columns if col in df_encoded.columns]
    
    if verbose:
        print(f"  → Encoding {len(columns_to_encode)} categorical columns with OrdinalEncoder...")
    
    from sklearn.preprocessing import OrdinalEncoder
    
    # Convert to string to handle any mixed types
    for col in columns_to_encode:
        df_encoded[col] = df_encoded[col].astype(str)
    
    # Create OrdinalEncoder with unknown value handling
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    
    # Fit and transform all categorical columns at once
    df_encoded[columns_to_encode] = encoder.fit_transform(df_encoded[columns_to_encode])
    
    # Build encoding map for reference
    encoding_map = {}
    for idx, col in enumerate(columns_to_encode):
        categories = encoder.categories_[idx].tolist()
        encoding_map[col] = {
            'encoder': encoder,  # Reference to the full encoder
            'classes': categories,
            'n_classes': len(categories)
        }
        
        if verbose:
            print(f"    ✓ {col}: {len(categories)} unique values → encoded to [0-{len(categories)-1}]")
    
    if verbose:
        print(f"  ✓ Ordinal encoding complete - all features now numeric")
        print(f"  ✓ Unknown values will be encoded as -1 during prediction")
        print(f"  ✓ Final feature count: {len(df_encoded.columns)} columns")
    
    return df_encoded, encoding_map



def clean_perovskite_dataset(df, verbose=True):
    """
    Complete cleaning pipeline for perovskite solar cell data.
    
    Args:
        df: Raw DataFrame from pvmaster.csv
        verbose: Print detailed progress messages
        
    Returns:
        pd.DataFrame: Cleaned DataFrame ready for modeling
    """
    if verbose:
        print("=" * 70)
        print("PEROVSKITE DATA CLEANING PIPELINE")
        print("=" * 70)
        print(f"\n📊 Initial dataset: {len(df)} rows × {len(df.columns)} columns")
    
    df_clean = df.copy()
    
    # ========================================================================
    # Step 1: Select columns needed for filtering and cleaning
    # ========================================================================
    if verbose:
        print(f"\n🔧 Step 1: Selecting {len(COLUMNS_FOR_FILTERING)} columns for filtering...")
    
    missing_cols = [col for col in COLUMNS_FOR_FILTERING if col not in df_clean.columns]
    if missing_cols:
        print(f"  ⚠️  Warning: Missing columns in dataset: {missing_cols}")
        available_cols = [col for col in COLUMNS_FOR_FILTERING if col in df_clean.columns]
        df_clean = df_clean[available_cols]
    else:
        df_clean = df_clean[COLUMNS_FOR_FILTERING]
    
    if verbose:
        print(f"  ✓ Dataset shape after column selection: {df_clean.shape}")
    
    # ========================================================================
    # Step 2: Drop rows with NaN in critical ion composition columns
    # ========================================================================
    if verbose:
        print(f"\n🔧 Step 2: Removing rows with missing ion composition data...")
    
    ion_columns = [
        'Perovskite_composition_a_ions',
        'Perovskite_composition_a_ions_coefficients',
        'Perovskite_composition_b_ions',
        'Perovskite_composition_b_ions_coefficients',
        'Perovskite_composition_c_ions',
        'Perovskite_composition_c_ions_coefficients'
    ]
    
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=ion_columns)
    rows_dropped = initial_rows - len(df_clean)
    
    if verbose:
        print(f"  ↳ Dropped {rows_dropped} rows with NaN in ion columns")
        print(f"  ✓ Remaining rows: {len(df_clean)}")
    
    # ========================================================================
    # Step 3: Filter B-site ions (must be exactly 'Pb')
    # ========================================================================
    if verbose:
        print(f"\n🔧 Step 3: Filtering B-site ions (keeping only 'Pb')...")
    
    initial_rows = len(df_clean)
    df_clean['Perovskite_composition_b_ions'] = df_clean['Perovskite_composition_b_ions'].astype(str)
    df_clean = df_clean[df_clean['Perovskite_composition_b_ions'] == 'Pb']
    rows_filtered = initial_rows - len(df_clean)
    
    if verbose:
        print(f"  ↳ Filtered out {rows_filtered} rows with non-Pb B-site")
        print(f"  ✓ Remaining rows: {len(df_clean)}")
    
    # ========================================================================
    # Step 4: Filter A-site ions (must contain Cs, FA, or MA)
    # ========================================================================
    if verbose:
        print(f"\n🔧 Step 4: Filtering A-site ions (keeping Cs/FA/MA)...")
    
    initial_rows = len(df_clean)
    a_ion_pattern = r'(Cs|FA|MA)'
    df_clean = df_clean[
        df_clean['Perovskite_composition_a_ions'].astype(str).str.contains(
            a_ion_pattern, na=False, regex=True
        )
    ]
    rows_filtered = initial_rows - len(df_clean)
    
    if verbose:
        print(f"  ↳ Filtered out {rows_filtered} rows without Cs/FA/MA A-sites")
        print(f"  ✓ Remaining rows: {len(df_clean)}")
    
    # ========================================================================
    # Step 5: Filter C-site ions (must contain Cl, Br, or I)
    # ========================================================================
    if verbose:
        print(f"\n🔧 Step 5: Filtering C-site ions (keeping Cl/Br/I)...")
    
    initial_rows = len(df_clean)
    c_ion_pattern = r'(Cl|Br|I)'
    df_clean = df_clean[
        df_clean['Perovskite_composition_c_ions'].astype(str).str.contains(
            c_ion_pattern, na=False, regex=True
        )
    ]
    rows_filtered = initial_rows - len(df_clean)
    
    if verbose:
        print(f"  ↳ Filtered out {rows_filtered} rows without Cl/Br/I C-sites")
        print(f"  ✓ Remaining rows: {len(df_clean)}")
    
    # ========================================================================
    # Step 6: Validate stoichiometry (ABX3 structure)
    # ========================================================================
    if verbose:
        print(f"\n🔧 Step 6: Validating stoichiometry (ABX₃ structure)...")
    
    if not df_clean.empty:
        # Calculate coefficient sums
        df_clean['_A_sum'] = df_clean['Perovskite_composition_a_ions_coefficients'].apply(sum_coefficients)
        df_clean['_B_sum'] = df_clean['Perovskite_composition_b_ions_coefficients'].apply(sum_coefficients)
        df_clean['_C_sum'] = df_clean['Perovskite_composition_c_ions_coefficients'].apply(sum_coefficients)
        
        # Filter A-site sum ≈ 1.0
        initial_rows = len(df_clean)
        df_clean = df_clean[np.isclose(df_clean['_A_sum'], 1.0, atol=TOLERANCE_A)]
        if verbose:
            print(f"  ↳ A-site sum check: removed {initial_rows - len(df_clean)} rows (sum ≉ 1.0 ± {TOLERANCE_A})")
        
        # Filter B-site sum ≈ 1.0
        initial_rows = len(df_clean)
        df_clean = df_clean[np.isclose(df_clean['_B_sum'], 1.0, atol=TOLERANCE_B)]
        if verbose:
            print(f"  ↳ B-site sum check: removed {initial_rows - len(df_clean)} rows (sum ≉ 1.0 ± {TOLERANCE_B})")
        
        # Filter C-site sum ≈ 3.0
        initial_rows = len(df_clean)
        df_clean = df_clean[np.isclose(df_clean['_C_sum'], 3.0, atol=TOLERANCE_C)]
        if verbose:
            print(f"  ↳ C-site sum check: removed {initial_rows - len(df_clean)} rows (sum ≉ 3.0 ± {TOLERANCE_C})")
        
        # Drop temporary sum columns
        df_clean = df_clean.drop(columns=['_A_sum', '_B_sum', '_C_sum'])
        
        if verbose:
            print(f"  ✓ Remaining rows after stoichiometry validation: {len(df_clean)}")
    else:
        if verbose:
            print("  ⚠️  DataFrame empty after ion filtering, skipping stoichiometry check")
    
    # ========================================================================
    # Step 7: Deduplicate deposition procedure values
    # ========================================================================
    if verbose:
        print(f"\n🔧 Step 7: Deduplicating deposition procedure values...")
    
    deposition_columns = [
        'Perovskite_deposition_procedure',
        'ETL_deposition_procedure',
        'HTL_deposition_procedure',
        'Backcontact_deposition_procedure'
    ]
    
    # Apply deduplication to each deposition column
    for col in deposition_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(deduplicate_deposition_values)
            if verbose:
                print(f"  ✓ Deduplicated: {col}")
    
    if verbose:
        print(f"  ✓ Deposition procedures cleaned")
    
    # ========================================================================
    # Step 8: Remove 'Unknown' and empty values
    # ========================================================================
    if verbose:
        print(f"\n🔧 Step 8: Removing 'Unknown' and empty values (regex-based)...")
    
    # Check all string/categorical columns for Unknown values
    columns_to_check = [
        'Perovskite_deposition_procedure',
        'ETL_stack_sequence',
        'ETL_deposition_procedure',
        'HTL_stack_sequence',
        'HTL_deposition_procedure',
        'Backcontact_stack_sequence',
        'Backcontact_deposition_procedure',
        'Substrate_stack_sequence',
        'Perovskite_composition_short_form',
        'Perovskite_composition_long_form'
    ]
    
    # Only check columns that exist
    columns_to_check = [col for col in columns_to_check if col in df_clean.columns]
    
    df_clean = filter_unknown_values(df_clean, columns_to_check)
    
    if verbose:
        print(f"  ✓ Remaining rows: {len(df_clean)}")

    # ========================================================================
    # Step 9: Extract first material from ETL and HTL stack sequences
    # ========================================================================
    if verbose:
        print(f"\n🔧 Step 9: Extracting primary material from stack sequences...")
    
    stack_columns = ['ETL_stack_sequence', 'HTL_stack_sequence']
    
    for col in stack_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(extract_first_stack_material)
            
            if verbose:
                print(f"  ✓ {col}: Extracted first material from pipe-separated layers")
    
    if verbose:
        print(f"  ✓ Stack sequences simplified to primary materials only")

    # ========================================================================
    # Step 10: Drop ABX3 filtering columns and composition forms (no longer needed)
    # ========================================================================
    if verbose:
        print(f"\n🔧 Step 10: Dropping ABX3 filtering columns and composition forms...")
    
    columns_to_drop = [
        'Perovskite_composition_a_ions',
        'Perovskite_composition_a_ions_coefficients',
        'Perovskite_composition_b_ions',
        'Perovskite_composition_b_ions_coefficients',
        'Perovskite_composition_c_ions',
        'Perovskite_composition_c_ions_coefficients',
        'Perovskite_composition_short_form',
        'Perovskite_composition_long_form'
    ]
    
    cols_dropped = []
    for col in columns_to_drop:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])
            cols_dropped.append(col)
    
    if verbose:
        print(f"  ✓ Dropped {len(cols_dropped)} columns used only for filtering:")
        for col in cols_dropped:
            print(f"    - {col}")
        print(f"  ✓ Remaining columns: {len(df_clean.columns)}")
    
    # ========================================================================
    # Step 11: Encode categorical features for LazyPredict
    # ========================================================================
    if verbose:
        print(f"\n🔧 Step 11: Encoding categorical features...")
    
    df_clean, encoding_map = encode_categorical_features(df_clean, verbose=verbose)
    
    if verbose:
        print(f"  ✓ All features now numeric and LazyPredict-ready")
    
    # ========================================================================
    # Step 12: Final quality checks
    # ========================================================================
    if verbose:
        print(f"\n🔧 Step 12: Final quality checks...")

    
    # Check for duplicate Ref_IDs (if Ref_ID exists)
    if 'Ref_ID' in df_clean.columns:
        duplicates = df_clean['Ref_ID'].duplicated().sum()
        if duplicates > 0:
            if verbose:
                print(f"  ⚠️  Found {duplicates} duplicate Ref_IDs (keeping first occurrence)")
            df_clean = df_clean.drop_duplicates(subset='Ref_ID', keep='first')
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    if verbose:
        print(f"  ✓ Final dataset shape: {df_clean.shape}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("✅ CLEANING COMPLETE")
        print("=" * 70)
        print(f"Original dataset:  {len(df)} rows")
        print(f"Cleaned dataset:   {len(df_clean)} rows")
        print(f"Rows removed:      {len(df) - len(df_clean)} ({100 * (len(df) - len(df_clean)) / len(df):.1f}%)")
        print(f"Columns kept:      {len(df_clean.columns)}")
        print("=" * 70 + "\n")
    
    return df_clean, encoding_map


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main function to run the cleaning pipeline.
    """
    # File paths
    input_file = "pvmaster.csv"
    output_file = "pvmaster_cleaned.csv"
    
    print(f"\n🔍 Looking for input file: {input_file}")
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Error: Input file '{input_file}' not found!")
        print(f"   Current directory: {Path.cwd()}")
        sys.exit(1)
    
    # Load data
    print(f"📂 Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Successfully loaded {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        sys.exit(1)
    
    # Clean data
    df_cleaned, encoding_map = clean_perovskite_dataset(df, verbose=True)
    
    # Save cleaned data
    if len(df_cleaned) > 0:
        print(f"💾 Saving cleaned data to {output_file}...")
        try:
            df_cleaned.to_csv(output_file, index=False)
            print(f"✅ Successfully saved {len(df_cleaned)} rows to {output_file}")
            
            # Save encoding map for reference
            encoding_file = "encoding_map.json"
            # Convert LabelEncoders to serializable format
            serializable_map = {}
            for col, info in encoding_map.items():
                serializable_map[col] = {
                    'classes': info['classes'],
                    'n_classes': info['n_classes']
                }
            
            with open(encoding_file, 'w') as f:
                json.dump(serializable_map, f, indent=2)
            print(f"✅ Saved encoding map to {encoding_file}")
            
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            sys.exit(1)
    else:
        print("⚠️  Warning: Cleaned dataset is empty! No file saved.")
        sys.exit(1)
    
    # Display sample of cleaned data
    print("\n📊 Sample of cleaned data (first 5 rows):")
    print(df_cleaned.head().to_string())
    
    # Display basic statistics
    print("\n📈 Data quality report:")
    print(f"   - Unique Ref_IDs: {df_cleaned['Ref_ID'].nunique() if 'Ref_ID' in df_cleaned.columns else 'N/A'}")
    print(f"   - Total features: {len(df_cleaned.columns)}")
    print(f"   - Missing values per column:")
    missing_summary = df_cleaned.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    if len(missing_summary) > 0:
        for col, count in missing_summary.items():
            print(f"     • {col}: {count} ({100*count/len(df_cleaned):.1f}%)")
    else:
        print("     • No missing values! 🎉")
    
    print("\n✅ All done!\n")


if __name__ == "__main__":
    main()

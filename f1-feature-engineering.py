"""
F1 Feature Engineering Script - WITH VERIFICATION
Transforms raw race data into predictive features with comprehensive validation

LOGIC:
1. Load and validate raw data
2. Create features step-by-step
3. Verify each feature set before proceeding
4. Save verified features
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURE_VERIFICATION_CONFIG = {
    'max_missing_percentage': 0.15,  # Max 15% missing values per feature
    'min_feature_variance': 0.01,    # Features must have some variance
    'max_correlation': 0.95,         # Max correlation between features
}

# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================

def load_and_validate_raw_data(filepath: str = 'f1_historical_data_2018_2025_MASTER.csv') -> Tuple[pd.DataFrame, Dict]:
    """Load raw data and perform initial validation"""
    
    print("="*80)
    print("📂 STEP 1: LOADING RAW DATA")
    print("="*80)
    
    validation = {
        'passed': False,
        'issues': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df):,} records from {filepath}")
    except FileNotFoundError:
        validation['issues'].append(f"File not found: {filepath}")
        print(f"❌ File not found: {filepath}")
        return None, validation
    except Exception as e:
        validation['issues'].append(f"Error loading file: {str(e)}")
        print(f"❌ Error loading file: {str(e)}")
        return None, validation
    
    # Required columns
    required_columns = [
        'Year', 'Round', 'Driver', 'Team', 'Circuit', 'CircuitID',
        'GridPosition', 'QualiPosition', 'FinishPosition', 'Points', 'DNF'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation['issues'].append(f"Missing required columns: {missing_columns}")
        print(f"❌ Missing columns: {missing_columns}")
        return None, validation
    
    # Basic statistics
    validation['stats'] = {
        'total_records': len(df),
        'years': df['Year'].nunique(),
        'drivers': df['Driver'].nunique(),
        'teams': df['Team'].nunique(),
        'circuits': df['Circuit'].nunique(),
        'races': df.groupby(['Year', 'Round']).ngroups
    }
    
    print(f"\n📊 Data Overview:")
    print(f"   Years: {df['Year'].min()} - {df['Year'].max()}")
    print(f"   Total races: {validation['stats']['races']}")
    print(f"   Unique drivers: {validation['stats']['drivers']}")
    print(f"   Unique teams: {validation['stats']['teams']}")
    
    # Check for data quality issues
    if validation['stats']['total_records'] < 1000:
        validation['warnings'].append("Very few records - results may be unreliable")
    
    if validation['stats']['races'] < 50:
        validation['warnings'].append("Limited race history - model may underperform")
    
    # Check for missing values in critical columns
    critical_missing = {}
    for col in ['Year', 'Round', 'Driver', 'Team', 'FinishPosition']:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            critical_missing[col] = missing_count
    
    if critical_missing:
        validation['warnings'].append(f"Missing values in critical columns: {critical_missing}")
        print(f"⚠️  Missing values detected: {critical_missing}")
    
    validation['passed'] = len(validation['issues']) == 0
    
    if validation['passed']:
        print(f"✅ Data validation PASSED\n")
    else:
        print(f"❌ Data validation FAILED\n")
        for issue in validation['issues']:
            print(f"   ❌ {issue}")
    
    return df, validation

# ============================================================================
# FEATURE CREATION FUNCTIONS
# ============================================================================

def create_momentum_features(df: pd.DataFrame, window: int = 5) -> Tuple[pd.DataFrame, Dict]:
    """Create driver momentum features with verification - PREVENTS DATA LEAKAGE"""
    
    print("="*80)
    print("📈 STEP 2: CREATING MOMENTUM FEATURES")
    print("="*80)
    
    verification = {
        'passed': False,
        'features_created': [],
        'issues': [],
        'warnings': []
    }
    
    try:
        df = df.sort_values(['Driver', 'Year', 'Round']).reset_index(drop=True)
        
        # Driver momentum - USE SHIFT TO PREVENT DATA LEAKAGE!
        print(f"   Creating driver momentum (last {window} races)...")
        print(f"   🔒 Using .shift(1) to prevent data leakage...")
        
        # Calculate rolling averages BEFORE the current race (shift by 1)
        df['Driver_AvgPosition_Last5'] = df.groupby('Driver')['FinishPosition'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        
        df['Driver_AvgPoints_Last5'] = df.groupby('Driver')['Points'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        
        df['Driver_DNF_Rate_Last5'] = df.groupby('Driver')['DNF'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        
        # Fill first race NaN values with reasonable defaults
        df['Driver_AvgPosition_Last5'].fillna(10, inplace=True)
        df['Driver_AvgPoints_Last5'].fillna(2, inplace=True)
        df['Driver_DNF_Rate_Last5'].fillna(0.1, inplace=True)
        
        # Races since last DNF
        print(f"   Creating races since DNF...")
        df['Races_Since_DNF'] = 0
        for driver in df['Driver'].unique():
            mask = df['Driver'] == driver
            driver_data = df.loc[mask].copy()
            
            races_since = 0
            races_since_list = []
            for dnf in driver_data['DNF'].values:
                if dnf == 1:
                    races_since = 0
                else:
                    races_since += 1
                races_since_list.append(races_since)
            
            df.loc[mask, 'Races_Since_DNF'] = races_since_list
        
        verification['features_created'] = [
            'Driver_AvgPosition_Last5',
            'Driver_AvgPoints_Last5', 
            'Driver_DNF_Rate_Last5',
            'Races_Since_DNF'
        ]
        
        # Verify features
        for feature in verification['features_created']:
            missing_pct = df[feature].isna().sum() / len(df)
            if missing_pct > FEATURE_VERIFICATION_CONFIG['max_missing_percentage']:
                verification['issues'].append(
                    f"{feature}: {missing_pct*100:.1f}% missing (threshold: {FEATURE_VERIFICATION_CONFIG['max_missing_percentage']*100:.1f}%)"
                )
            
            if df[feature].var() < FEATURE_VERIFICATION_CONFIG['min_feature_variance']:
                verification['warnings'].append(f"{feature}: Very low variance")
        
        # CRITICAL: Check for data leakage
        print(f"   🔒 Checking for data leakage...")
        for feature in ['Driver_AvgPosition_Last5', 'Driver_AvgPoints_Last5']:
            # For first race of each driver, these should be default values, not actual race data
            first_races = df.groupby('Driver').head(1)
            if feature == 'Driver_AvgPosition_Last5':
                if (first_races[feature] != 10).any():
                    verification['warnings'].append(f"{feature}: Potential data leakage detected in first races")
        
        verification['passed'] = len(verification['issues']) == 0
        
        print(f"   ✅ Created {len(verification['features_created'])} momentum features")
        if verification['warnings']:
            print(f"   ⚠️  {len(verification['warnings'])} warnings")
        if verification['issues']:
            print(f"   ❌ {len(verification['issues'])} issues")
            for issue in verification['issues']:
                print(f"      {issue}")
        
    except Exception as e:
        verification['issues'].append(f"Error creating momentum features: {str(e)}")
        print(f"   ❌ Error: {str(e)}")
    
    return df, verification

def create_team_features(df: pd.DataFrame, window: int = 5) -> Tuple[pd.DataFrame, Dict]:
    """Create team performance features with verification - PREVENTS DATA LEAKAGE"""
    
    print("\n="*80)
    print("🏢 STEP 3: CREATING TEAM PERFORMANCE FEATURES")
    print("="*80)
    
    verification = {
        'passed': False,
        'features_created': [],
        'issues': [],
        'warnings': []
    }
    
    try:
        df = df.sort_values(['Team', 'Year', 'Round']).reset_index(drop=True)
        
        print(f"   Creating team momentum features...")
        print(f"   🔒 Using .shift(1) to prevent data leakage...")
        
        # Team average position in last N races - SHIFT TO PREVENT LEAKAGE
        df['Team_AvgPosition_Last5'] = df.groupby(['Team', 'Year'])['FinishPosition'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        
        df['Team_AvgPoints_Last5'] = df.groupby(['Team', 'Year'])['Points'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        
        df['Team_DNF_Rate_Last5'] = df.groupby(['Team', 'Year'])['DNF'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        
        # Fill NaN with reasonable defaults
        df['Team_AvgPosition_Last5'].fillna(10, inplace=True)
        df['Team_AvgPoints_Last5'].fillna(3, inplace=True)
        df['Team_DNF_Rate_Last5'].fillna(0.1, inplace=True)
        
        verification['features_created'] = [
            'Team_AvgPosition_Last5',
            'Team_AvgPoints_Last5',
            'Team_DNF_Rate_Last5'
        ]
        
        # Verify features
        for feature in verification['features_created']:
            missing_pct = df[feature].isna().sum() / len(df)
            if missing_pct > FEATURE_VERIFICATION_CONFIG['max_missing_percentage']:
                verification['issues'].append(
                    f"{feature}: {missing_pct*100:.1f}% missing"
                )
        
        verification['passed'] = len(verification['issues']) == 0
        
        print(f"   ✅ Created {len(verification['features_created'])} team features")
        if verification['issues']:
            print(f"   ❌ {len(verification['issues'])} issues")
            for issue in verification['issues']:
                print(f"      {issue}")
        
    except Exception as e:
        verification['issues'].append(f"Error creating team features: {str(e)}")
        print(f"   ❌ Error: {str(e)}")
    
    return df, verification

def create_circuit_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Create circuit-specific features with verification"""
    
    print("\n="*80)
    print("🏁 STEP 4: CREATING CIRCUIT-SPECIFIC FEATURES")
    print("="*80)
    
    verification = {
        'passed': False,
        'features_created': [],
        'issues': [],
        'warnings': []
    }
    
    try:
        print(f"   Creating circuit history features...")
        
        # Driver circuit history
        df = df.sort_values(['Driver', 'CircuitID', 'Year', 'Round']).reset_index(drop=True)
        
        df['Driver_Circuit_AvgPosition'] = df.groupby(['Driver', 'CircuitID'])['FinishPosition'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        df['Driver_Circuit_AvgPoints'] = df.groupby(['Driver', 'CircuitID'])['Points'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        df['Driver_Circuit_Appearances'] = df.groupby(['Driver', 'CircuitID']).cumcount()
        
        # Fill NaN for first appearances
        df['Driver_Circuit_AvgPosition'].fillna(10, inplace=True)
        df['Driver_Circuit_AvgPoints'].fillna(0, inplace=True)
        
        # Circuit type categorization
        print(f"   Creating circuit type categories...")
        street_circuits = ['Monaco', 'Singapore', 'Baku', 'Jeddah', 'Miami', 'Las Vegas', 'Saudi']
        high_speed_circuits = ['Monza', 'Spa', 'Silverstone', 'Suzuka', 'Jeddah', 'Lusail']
        
        df['Circuit_Type_Street'] = df['CircuitID'].apply(
            lambda x: 1 if any(street in str(x) for street in street_circuits) else 0
        )
        df['Circuit_Type_HighSpeed'] = df['CircuitID'].apply(
            lambda x: 1 if any(high in str(x) for high in high_speed_circuits) else 0
        )
        df['Circuit_Type_Traditional'] = ((df['Circuit_Type_Street'] == 0) & 
                                          (df['Circuit_Type_HighSpeed'] == 0)).astype(int)
        
        verification['features_created'] = [
            'Driver_Circuit_AvgPosition',
            'Driver_Circuit_AvgPoints',
            'Driver_Circuit_Appearances',
            'Circuit_Type_Street',
            'Circuit_Type_HighSpeed',
            'Circuit_Type_Traditional'
        ]
        
        # Verify features
        for feature in verification['features_created']:
            missing_pct = df[feature].isna().sum() / len(df)
            if missing_pct > FEATURE_VERIFICATION_CONFIG['max_missing_percentage']:
                verification['issues'].append(
                    f"{feature}: {missing_pct*100:.1f}% missing"
                )
        
        verification['passed'] = len(verification['issues']) == 0
        
        print(f"   ✅ Created {len(verification['features_created'])} circuit features")
        if verification['issues']:
            print(f"   ❌ {len(verification['issues'])} issues")
            for issue in verification['issues']:
                print(f"      {issue}")
        
    except Exception as e:
        verification['issues'].append(f"Error creating circuit features: {str(e)}")
        print(f"   ❌ Error: {str(e)}")
    
    return df, verification

def create_season_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Create season progress features with verification - PREVENTS DATA LEAKAGE"""
    
    print("\n="*80)
    print("📅 STEP 5: CREATING SEASON PROGRESS FEATURES")
    print("="*80)
    
    verification = {
        'passed': False,
        'features_created': [],
        'issues': [],
        'warnings': []
    }
    
    try:
        print(f"   Creating season cumulative stats...")
        print(f"   🔒 Using cumulative stats BEFORE current race...")
        
        df = df.sort_values(['Driver', 'Year', 'Round']).reset_index(drop=True)
        
        # Season cumulative stats - Use values BEFORE current race
        df['Season_Points_Total'] = df.groupby(['Driver', 'Year'])['Points'].transform(
            lambda x: x.shift(1).cumsum().fillna(0)
        )
        
        df['Season_Avg_Position'] = df.groupby(['Driver', 'Year'])['FinishPosition'].transform(
            lambda x: x.shift(1).expanding().mean().fillna(10)
        )
        
        df['Season_Races_Completed'] = df.groupby(['Driver', 'Year']).cumcount()  # This is correct (0-indexed)
        
        # Podiums and wins - count BEFORE current race
        df['Season_Podiums'] = df.groupby(['Driver', 'Year']).apply(
            lambda x: (x['FinishPosition'] <= 3).shift(1).cumsum().fillna(0)
        ).reset_index(level=[0,1], drop=True)
        
        df['Season_Wins'] = df.groupby(['Driver', 'Year']).apply(
            lambda x: (x['FinishPosition'] == 1).shift(1).cumsum().fillna(0)
        ).reset_index(level=[0,1], drop=True)
        
        verification['features_created'] = [
            'Season_Points_Total',
            'Season_Avg_Position',
            'Season_Races_Completed',
            'Season_Podiums',
            'Season_Wins'
        ]
        
        # Verify features
        for feature in verification['features_created']:
            missing_pct = df[feature].isna().sum() / len(df)
            if missing_pct > FEATURE_VERIFICATION_CONFIG['max_missing_percentage']:
                verification['issues'].append(
                    f"{feature}: {missing_pct*100:.1f}% missing"
                )
        
        verification['passed'] = len(verification['issues']) == 0
        
        print(f"   ✅ Created {len(verification['features_created'])} season features")
        if verification['issues']:
            print(f"   ❌ {len(verification['issues'])} issues")
            for issue in verification['issues']:
                print(f"      {issue}")
        
    except Exception as e:
        verification['issues'].append(f"Error creating season features: {str(e)}")
        print(f"   ❌ Error: {str(e)}")
    
    return df, verification

def create_qualifying_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Create qualifying-related features with verification"""
    
    print("\n="*80)
    print("⏱️  STEP 6: CREATING QUALIFYING FEATURES")
    print("="*80)
    
    verification = {
        'passed': False,
        'features_created': [],
        'issues': [],
        'warnings': []
    }
    
    try:
        print(f"   Creating qualifying delta features...")
        
        # Delta from pole position
        df['Quali_Delta_From_Pole'] = df.groupby(['Year', 'Round'])['QualiPosition'].transform(
            lambda x: x - x.min()
        )
        
        # Grid penalty (difference between quali and grid)
        df['Grid_Penalty'] = df['GridPosition'] - df['QualiPosition']
        df['Grid_Penalty'] = df['Grid_Penalty'].fillna(0)
        
        verification['features_created'] = [
            'Quali_Delta_From_Pole',
            'Grid_Penalty'
        ]
        
        # Verify features
        for feature in verification['features_created']:
            missing_pct = df[feature].isna().sum() / len(df)
            if missing_pct > FEATURE_VERIFICATION_CONFIG['max_missing_percentage']:
                verification['issues'].append(
                    f"{feature}: {missing_pct*100:.1f}% missing"
                )
        
        verification['passed'] = len(verification['issues']) == 0
        
        print(f"   ✅ Created {len(verification['features_created'])} qualifying features")
        if verification['issues']:
            print(f"   ❌ {len(verification['issues'])} issues")
            for issue in verification['issues']:
                print(f"      {issue}")
        
    except Exception as e:
        verification['issues'].append(f"Error creating qualifying features: {str(e)}")
        print(f"   ❌ Error: {str(e)}")
    
    return df, verification

def add_championship_position(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Add championship standing before each race with verification"""
    
    print("\n="*80)
    print("🏆 STEP 7: ADDING CHAMPIONSHIP POSITIONS")
    print("="*80)
    
    verification = {
        'passed': False,
        'features_created': [],
        'issues': [],
        'warnings': []
    }
    
    try:
        print(f"   Calculating championship positions...")
        
        df = df.sort_values(['Year', 'Round', 'Driver']).reset_index(drop=True)
        
        df['Championship_Position'] = 20  # Default
        
        for year in df['Year'].unique():
            year_data = df[df['Year'] == year]
            
            for round_num in sorted(year_data['Round'].unique()):
                # Get standings before this race
                previous_rounds = year_data[year_data['Round'] < round_num]
                
                if len(previous_rounds) > 0:
                    standings = previous_rounds.groupby('Driver')['Points'].sum().sort_values(ascending=False)
                    standings_rank = {driver: rank + 1 for rank, driver in enumerate(standings.index)}
                    
                    # Update championship position for this round
                    mask = (df['Year'] == year) & (df['Round'] == round_num)
                    df.loc[mask, 'Championship_Position'] = df.loc[mask, 'Driver'].map(standings_rank).fillna(20)
        
        verification['features_created'] = ['Championship_Position']
        
        # Verify
        missing_pct = df['Championship_Position'].isna().sum() / len(df)
        if missing_pct > FEATURE_VERIFICATION_CONFIG['max_missing_percentage']:
            verification['issues'].append(
                f"Championship_Position: {missing_pct*100:.1f}% missing"
            )
        
        verification['passed'] = len(verification['issues']) == 0
        
        print(f"   ✅ Created championship position feature")
        if verification['issues']:
            print(f"   ❌ {len(verification['issues'])} issues")
            for issue in verification['issues']:
                print(f"      {issue}")
        
    except Exception as e:
        verification['issues'].append(f"Error creating championship positions: {str(e)}")
        print(f"   ❌ Error: {str(e)}")
    
    return df, verification

# ============================================================================
# FINAL VERIFICATION
# ============================================================================

def perform_final_verification(df: pd.DataFrame, all_feature_names: list) -> Dict:
    """Perform comprehensive final verification"""
    
    print("\n="*80)
    print("🔍 STEP 8: FINAL COMPREHENSIVE VERIFICATION")
    print("="*80)
    
    verification = {
        'passed': False,
        'total_features': len(all_feature_names),
        'issues': [],
        'warnings': [],
        'statistics': {}
    }
    
    print(f"   Verifying {len(all_feature_names)} features...")
    
    # Check 1: Missing values
    print(f"\n   1️⃣  Checking missing values...")
    missing_report = {}
    for feature in all_feature_names:
        if feature in df.columns:
            missing_pct = df[feature].isna().sum() / len(df) * 100
            if missing_pct > FEATURE_VERIFICATION_CONFIG['max_missing_percentage'] * 100:
                missing_report[feature] = missing_pct
    
    if missing_report:
        verification['issues'].append(f"{len(missing_report)} features exceed missing value threshold")
        print(f"      ❌ {len(missing_report)} features with excessive missing values")
        for feat, pct in list(missing_report.items())[:5]:
            print(f"         • {feat}: {pct:.1f}%")
    else:
        print(f"      ✅ All features within missing value threshold")
    
    # Check 2: Feature variance
    print(f"\n   2️⃣  Checking feature variance...")
    low_variance = []
    for feature in all_feature_names:
        if feature in df.columns:
            if df[feature].var() < FEATURE_VERIFICATION_CONFIG['min_feature_variance']:
                low_variance.append(feature)
    
    if low_variance:
        verification['warnings'].append(f"{len(low_variance)} features have low variance")
        print(f"      ⚠️  {len(low_variance)} features with low variance")
    else:
        print(f"      ✅ All features have sufficient variance")
    
    # Check 3: Correlation
    print(f"\n   3️⃣  Checking feature correlations...")
    numeric_features = [f for f in all_feature_names if f in df.columns and df[f].dtype in ['int64', 'float64']]
    if len(numeric_features) > 1:
        corr_matrix = df[numeric_features].corr().abs()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > FEATURE_VERIFICATION_CONFIG['max_correlation']:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            verification['warnings'].append(f"{len(high_corr_pairs)} highly correlated feature pairs found")
            print(f"      ⚠️  {len(high_corr_pairs)} highly correlated pairs (may want to remove one)")
        else:
            print(f"      ✅ No highly correlated feature pairs")
    
    # Statistics
    verification['statistics'] = {
        'total_records': len(df),
        'total_features': len(all_feature_names),
        'numeric_features': len(numeric_features),
        'missing_issues': len(missing_report),
        'low_variance': len(low_variance),
        'high_correlation_pairs': len(high_corr_pairs) if 'high_corr_pairs' in locals() else 0
    }
    
    # Final decision
    verification['passed'] = len(verification['issues']) == 0
    
    print(f"\n   {'='*76}")
    print(f"   📊 Final Statistics:")
    print(f"      Total features: {verification['statistics']['total_features']}")
    print(f"      Numeric features: {verification['statistics']['numeric_features']}")
    print(f"      Issues: {len(verification['issues'])}")
    print(f"      Warnings: {len(verification['warnings'])}")
    print(f"   {'='*76}")
    
    if verification['passed']:
        print(f"   ✅ FINAL VERIFICATION PASSED")
    else:
        print(f"   ❌ FINAL VERIFICATION FAILED")
        for issue in verification['issues']:
            print(f"      ❌ {issue}")
    
    return verification

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main feature engineering pipeline with verification"""
    
    print("\n" + "="*80)
    print("🏎️  F1 FEATURE ENGINEERING WITH VERIFICATION")
    print("="*80)
    
    all_verifications = []
    
    # Step 1: Load data
    df, validation = load_and_validate_raw_data()
    if not validation['passed']:
        print("\n❌ Cannot proceed - data loading failed")
        return None
    all_verifications.append(('Data Loading', validation))
    
    # Step 2: Momentum features
    df, verification = create_momentum_features(df)
    all_verifications.append(('Momentum Features', verification))
    if not verification['passed']:
        print("\n⚠️  Momentum features have issues, but continuing...")
    
    # Step 3: Team features
    df, verification = create_team_features(df)
    all_verifications.append(('Team Features', verification))
    if not verification['passed']:
        print("\n⚠️  Team features have issues, but continuing...")
    
    # Step 4: Circuit features
    df, verification = create_circuit_features(df)
    all_verifications.append(('Circuit Features', verification))
    if not verification['passed']:
        print("\n⚠️  Circuit features have issues, but continuing...")
    
    # Step 5: Season features
    df, verification = create_season_features(df)
    all_verifications.append(('Season Features', verification))
    if not verification['passed']:
        print("\n⚠️  Season features have issues, but continuing...")
    
    # Step 6: Qualifying features
    df, verification = create_qualifying_features(df)
    all_verifications.append(('Qualifying Features', verification))
    if not verification['passed']:
        print("\n⚠️  Qualifying features have issues, but continuing...")
    
    # Step 7: Championship position
    df, verification = add_championship_position(df)
    all_verifications.append(('Championship Position', verification))
    if not verification['passed']:
        print("\n⚠️  Championship position has issues, but continuing...")
    
    # Collect all feature names
    all_feature_names = []
    for name, verif in all_verifications[1:]:  # Skip data loading
        if 'features_created' in verif:
            all_feature_names.extend(verif['features_created'])
    
    # Step 8: Final verification
    final_verification = perform_final_verification(df, all_feature_names)
    all_verifications.append(('Final Verification', final_verification))
    
    # Save engineered features
    output_file = 'f1_engineered_features_2018_2025_VERIFIED.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"💾 SAVED ENGINEERED FEATURES")
    print(f"{'='*80}")
    print(f"   File: {output_file}")
    print(f"   Records: {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Feature columns: {len(all_feature_names)}")
    
    # Summary report
    print(f"\n{'='*80}")
    print(f"📋 FEATURE ENGINEERING SUMMARY")
    print(f"{'='*80}")
    
    for step_name, verif in all_verifications:
        status = "✅" if verif['passed'] else "❌"
        issues = len(verif.get('issues', []))
        warnings = len(verif.get('warnings', []))
        print(f"   {status} {step_name:30s} Issues: {issues:2d} | Warnings: {warnings:2d}")
    
    print(f"\n{'='*80}")
    print(f"🎯 NEXT STEP: Run model training")
    print(f"   python f1_model_training.py")
    print(f"{'='*80}\n")
    
    return df

if __name__ == "__main__":
    engineered_data = main()
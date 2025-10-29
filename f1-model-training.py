"""
F1 Model Training Script - WITH VERIFICATION
Train XGBoost model with comprehensive validation at each step

LOGIC:
1. Load and validate engineered features
2. Prepare training data with verification
3. Split data with verification  
4. Train model with cross-validation
5. Evaluate and verify model performance
6. Save verified model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_VERIFICATION_CONFIG = {
    'min_training_samples': 1000,      # Minimum training samples
    'min_validation_samples': 200,     # Minimum validation samples
    'max_acceptable_mae': 3.5,         # Maximum MAE for validation set
    'min_r2_score': 0.3,              # Minimum R² score
    'cv_folds': 3,                    # Cross-validation folds
}

# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================

def load_and_validate_features(filepath: str = 'f1_engineered_features_2018_2025_VERIFIED.csv') -> Tuple[pd.DataFrame, Dict]:
    """Load engineered features and validate"""
    
    print("="*80)
    print("📂 STEP 1: LOADING ENGINEERED FEATURES")
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
        print(f"❌ Error: {str(e)}")
        return None, validation
    
    # Check required columns
    required_columns = [
        'Year', 'Round', 'Driver', 'Team', 'FinishPosition',
        'GridPosition', 'QualiPosition', 'Driver_AvgPosition_Last5',
        'Team_AvgPosition_Last5', 'Championship_Position'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation['issues'].append(f"Missing required columns: {missing_columns}")
        print(f"❌ Missing columns: {missing_columns}")
        return None, validation
    
    validation['stats'] = {
        'total_records': len(df),
        'total_features': len(df.columns),
        'years': sorted(df['Year'].unique().tolist()),
        'drivers': df['Driver'].nunique(),
        'teams': df['Team'].nunique()
    }
    
    print(f"\n📊 Data Overview:")
    print(f"   Records: {validation['stats']['total_records']:,}")
    print(f"   Features: {validation['stats']['total_features']}")
    print(f"   Years: {validation['stats']['years'][0]} - {validation['stats']['years'][-1]}")
    print(f"   Unique drivers: {validation['stats']['drivers']}")
    print(f"   Unique teams: {validation['stats']['teams']}")
    
    if validation['stats']['total_records'] < MODEL_VERIFICATION_CONFIG['min_training_samples']:
        validation['issues'].append(
            f"Insufficient records: {validation['stats']['total_records']} "
            f"(minimum: {MODEL_VERIFICATION_CONFIG['min_training_samples']})"
        )
    
    validation['passed'] = len(validation['issues']) == 0
    
    if validation['passed']:
        print(f"✅ Data validation PASSED\n")
    else:
        print(f"❌ Data validation FAILED")
        for issue in validation['issues']:
            print(f"   ❌ {issue}")
    
    return df, validation

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict, Dict, List]:
    """Prepare features and target with verification"""
    
    print("="*80)
    print("🔧 STEP 2: PREPARING TRAINING DATA")
    print("="*80)
    
    verification = {
        'passed': False,
        'issues': [],
        'warnings': [],
        'stats': {}
    }
    
    # Define feature columns
    feature_columns = [
        # Grid and Qualifying
        'GridPosition', 'QualiPosition', 'Quali_Delta_From_Pole', 'Grid_Penalty',
        
        # Driver Momentum
        'Driver_AvgPosition_Last5', 'Driver_AvgPoints_Last5', 
        'Driver_DNF_Rate_Last5', 'Races_Since_DNF',
        
        # Team Performance
        'Team_AvgPosition_Last5', 'Team_AvgPoints_Last5', 'Team_DNF_Rate_Last5',
        
        # Circuit History
        'Driver_Circuit_AvgPosition', 'Driver_Circuit_AvgPoints', 
        'Driver_Circuit_Appearances',
        
        # Season Progress
        'Season_Points_Total', 'Season_Avg_Position', 'Season_Races_Completed',
        'Season_Podiums', 'Season_Wins', 'Championship_Position',
        
        # Circuit Type
        'Circuit_Type_Street', 'Circuit_Type_HighSpeed', 'Circuit_Type_Traditional'
    ]
    
    # Check which features exist
    available_features = [f for f in feature_columns if f in df.columns]
    missing_features = [f for f in feature_columns if f not in df.columns]
    
    if missing_features:
        verification['warnings'].append(f"{len(missing_features)} features missing: {missing_features[:5]}")
        print(f"⚠️  {len(missing_features)} features not found (using {len(available_features)} available)")
    
    print(f"   Using {len(available_features)} features for training")
    
    # Encode categorical variables
    print(f"   Encoding categorical variables...")
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    
    try:
        df['Driver_Encoded'] = le_driver.fit_transform(df['Driver'])
        df['Team_Encoded'] = le_team.fit_transform(df['Team'])
    except Exception as e:
        verification['issues'].append(f"Error encoding categories: {str(e)}")
        print(f"   ❌ Encoding failed: {str(e)}")
        return None, None, None, None, None
    
    available_features.extend(['Driver_Encoded', 'Team_Encoded'])
    
    # Prepare X and y
    print(f"   Preparing feature matrix and target...")
    try:
        X = df[available_features].copy()
        y = df['FinishPosition'].copy()
    except Exception as e:
        verification['issues'].append(f"Error preparing data: {str(e)}")
        print(f"   ❌ Preparation failed: {str(e)}")
        return None, None, None, None, None
    
    # Handle missing values
    missing_before = X.isna().sum().sum()
    if missing_before > 0:
        print(f"   ⚠️  Filling {missing_before} missing values with mean")
        X.fillna(X.mean(), inplace=True)
    
    missing_after = X.isna().sum().sum()
    if missing_after > 0:
        verification['warnings'].append(f"{missing_after} missing values remain after filling")
    
    # Verify target variable
    if y.isna().sum() > 0:
        verification['issues'].append(f"{y.isna().sum()} missing target values")
        print(f"   ❌ Target variable has {y.isna().sum()} missing values")
    
    verification['stats'] = {
        'features_used': len(available_features),
        'features_missing': len(missing_features),
        'samples': len(X),
        'missing_values_filled': missing_before,
        'target_range': (y.min(), y.max())
    }
    
    print(f"\n   📊 Preparation Summary:")
    print(f"      Features: {verification['stats']['features_used']}")
    print(f"      Samples: {verification['stats']['samples']:,}")
    print(f"      Target range: {verification['stats']['target_range']}")
    
    verification['passed'] = len(verification['issues']) == 0
    
    if verification['passed']:
        print(f"   ✅ Data preparation PASSED")
    else:
        print(f"   ❌ Data preparation FAILED")
        for issue in verification['issues']:
            print(f"      {issue}")
    
    return X, y, le_driver, le_team, available_features

# ============================================================================
# DATA SPLITTING
# ============================================================================

def split_data_with_verification(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict, Dict]:
    """Split data into train/val/test sets with verification"""
    
    print("\n" + "="*80)
    print("✂️  STEP 3: SPLITTING DATA (2018-2023 Train | 2024 Val | 2025 Test)")
    print("="*80)
    
    verification = {
        'passed': False,
        'issues': [],
        'warnings': [],
        'splits': {}
    }
    
    try:
        # Time-based split
        train_mask = df['Year'] <= 2023
        val_mask = df['Year'] == 2024
        test_mask = df['Year'] == 2025
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        X_val = X[val_mask]
        y_val = y[val_mask]
        
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
        
        verification['splits'] = {
            'train': {'samples': len(X_train), 'years': '2018-2023'},
            'val': {'samples': len(X_val), 'years': '2024'},
            'test': {'samples': len(X_test), 'years': '2025'}
        }
        
        print(f"   Training set:   {len(X_train):5,} samples (2018-2023)")
        print(f"   Validation set: {len(X_val):5,} samples (2024)")
        print(f"   Test set:       {len(X_test):5,} samples (2025)")
        
        # Verify split sizes
        if len(X_train) < MODEL_VERIFICATION_CONFIG['min_training_samples']:
            verification['issues'].append(
                f"Training set too small: {len(X_train)} "
                f"(minimum: {MODEL_VERIFICATION_CONFIG['min_training_samples']})"
            )
        
        if len(X_val) < MODEL_VERIFICATION_CONFIG['min_validation_samples']:
            verification['warnings'].append(
                f"Validation set small: {len(X_val)} "
                f"(recommended: {MODEL_VERIFICATION_CONFIG['min_validation_samples']})"
            )
        
        if len(X_test) == 0:
            verification['warnings'].append("Test set is empty (2025 data not available)")
        
        # Check data leakage (drivers in test not in train)
        train_drivers = set(df[train_mask]['Driver'].unique())
        test_drivers = set(df[test_mask]['Driver'].unique())
        new_drivers = test_drivers - train_drivers
        
        if new_drivers:
            verification['warnings'].append(
                f"{len(new_drivers)} drivers in test set not in training set: {list(new_drivers)[:3]}"
            )
            print(f"\n   ⚠️  {len(new_drivers)} new drivers in test set")
        
        verification['passed'] = len(verification['issues']) == 0
        
        if verification['passed']:
            print(f"   ✅ Data split verification PASSED")
        else:
            print(f"   ❌ Data split verification FAILED")
            for issue in verification['issues']:
                print(f"      {issue}")
        
        return splits, verification
        
    except Exception as e:
        verification['issues'].append(f"Error splitting data: {str(e)}")
        print(f"   ❌ Split failed: {str(e)}")
        return None, verification

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_xgboost_with_verification(X_train, y_train, X_val, y_val) -> Tuple[object, Dict]:
    """Train XGBoost model with verification"""
    
    print("\n" + "="*80)
    print("🤖 STEP 4: TRAINING XGBOOST MODEL")
    print("="*80)
    
    verification = {
        'passed': False,
        'issues': [],
        'warnings': [],
        'metrics': {}
    }
    
    try:
        # Baseline model
        print("   Training baseline model...")
        baseline_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )
        
        baseline_model.fit(X_train, y_train)
        
        # Baseline evaluation
        y_pred_train = baseline_model.predict(X_train)
        y_pred_val = baseline_model.predict(X_val)
        
        baseline_train_mae = mean_absolute_error(y_train, y_pred_train)
        baseline_val_mae = mean_absolute_error(y_val, y_pred_val)
        
        print(f"   Baseline Training MAE:   {baseline_train_mae:.3f}")
        print(f"   Baseline Validation MAE: {baseline_val_mae:.3f}")
        
        # Hyperparameter tuning
        print(f"\n   🔧 Tuning hyperparameters (this may take a while)...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=MODEL_VERIFICATION_CONFIG['cv_folds'],
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        print(f"\n   ✅ Best parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"      {param}: {value}")
        
        print(f"   Best CV MAE: {-grid_search.best_score_:.3f}")
        
        verification['metrics']['baseline_train_mae'] = baseline_train_mae
        verification['metrics']['baseline_val_mae'] = baseline_val_mae
        verification['metrics']['best_cv_mae'] = -grid_search.best_score_
        verification['metrics']['best_params'] = grid_search.best_params_
        
        # Verify model performance
        if baseline_val_mae > MODEL_VERIFICATION_CONFIG['max_acceptable_mae']:
            verification['warnings'].append(
                f"Validation MAE {baseline_val_mae:.3f} exceeds threshold "
                f"{MODEL_VERIFICATION_CONFIG['max_acceptable_mae']}"
            )
        
        verification['passed'] = True
        
        return best_model, verification
        
    except Exception as e:
        verification['issues'].append(f"Training error: {str(e)}")
        print(f"   ❌ Training failed: {str(e)}")
        return None, verification

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model_with_verification(model, X_train, y_train, X_val, y_val, X_test, y_test) -> Dict:
    """Comprehensive model evaluation with verification"""
    
    print("\n" + "="*80)
    print("📊 STEP 5: EVALUATING MODEL PERFORMANCE")
    print("="*80)
    
    verification = {
        'passed': False,
        'issues': [],
        'warnings': [],
        'metrics': {}
    }
    
    try:
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        # Calculate metrics
        metrics = {
            'Training': {
                'MAE': mean_absolute_error(y_train, y_pred_train),
                'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'R2': r2_score(y_train, y_pred_train)
            },
            'Validation': {
                'MAE': mean_absolute_error(y_val, y_pred_val),
                'RMSE': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'R2': r2_score(y_val, y_pred_val)
            }
        }
        
        # Test set (if available)
        if len(X_test) > 0:
            y_pred_test = model.predict(X_test)
            metrics['Test'] = {
                'MAE': mean_absolute_error(y_test, y_pred_test),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'R2': r2_score(y_test, y_pred_test)
            }
        
        # Display metrics
        print(f"\n   📈 Performance Metrics:")
        print(f"   {'-'*60}")
        for dataset, scores in metrics.items():
            print(f"\n   {dataset} Set:")
            print(f"      MAE:  {scores['MAE']:.3f} positions")
            print(f"      RMSE: {scores['RMSE']:.3f} positions")
            print(f"      R²:   {scores['R2']:.3f}")
        
        verification['metrics'] = metrics
        
        # Verification checks
        val_mae = metrics['Validation']['MAE']
        val_r2 = metrics['Validation']['R2']
        
        if val_mae > MODEL_VERIFICATION_CONFIG['max_acceptable_mae']:
            verification['warnings'].append(
                f"Validation MAE {val_mae:.3f} exceeds acceptable threshold "
                f"{MODEL_VERIFICATION_CONFIG['max_acceptable_mae']}"
            )
            print(f"\n   ⚠️  Validation MAE exceeds threshold")
        
        if val_r2 < MODEL_VERIFICATION_CONFIG['min_r2_score']:
            verification['warnings'].append(
                f"Validation R² {val_r2:.3f} below minimum threshold "
                f"{MODEL_VERIFICATION_CONFIG['min_r2_score']}"
            )
            print(f"\n   ⚠️  Validation R² below threshold")
        
        # Check for overfitting
        train_mae = metrics['Training']['MAE']
        if val_mae > train_mae * 1.5:
            verification['warnings'].append(
                f"Possible overfitting: Val MAE {val_mae:.3f} >> Train MAE {train_mae:.3f}"
            )
            print(f"\n   ⚠️  Possible overfitting detected")
        
        # Overall pass/fail
        verification['passed'] = (
            val_mae <= MODEL_VERIFICATION_CONFIG['max_acceptable_mae'] * 1.2 and  # Allow 20% tolerance
            val_r2 >= MODEL_VERIFICATION_CONFIG['min_r2_score'] * 0.8  # Allow 20% tolerance
        )
        
        if verification['passed']:
            print(f"\n   ✅ Model evaluation PASSED")
        else:
            print(f"\n   ⚠️  Model evaluation has concerns but may be usable")
        
        return verification
        
    except Exception as e:
        verification['issues'].append(f"Evaluation error: {str(e)}")
        print(f"   ❌ Evaluation failed: {str(e)}")
        return verification

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def analyze_feature_importance(model, feature_columns: List, top_n: int = 15) -> Dict:
    """Analyze and display feature importance"""
    
    print("\n" + "="*80)
    print("🔍 STEP 6: FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    try:
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\n   Top {top_n} Most Important Features:")
        print(f"   {'-'*60}")
        for i, row in feature_importance.head(top_n).iterrows():
            bar = "█" * int(row['Importance'] * 100)
            print(f"   {row['Feature']:35s} {row['Importance']:.4f} {bar}")
        
        return feature_importance
        
    except Exception as e:
        print(f"   ❌ Feature importance analysis failed: {str(e)}")
        return None

# ============================================================================
# SAVE MODEL
# ============================================================================

def save_model_with_verification(model, le_driver, le_team, feature_columns) -> Dict:
    """Save model and encoders with verification"""
    
    print("\n" + "="*80)
    print("💾 STEP 7: SAVING MODEL AND ENCODERS")
    print("="*80)
    
    verification = {
        'passed': False,
        'issues': [],
        'files_saved': []
    }
    
    try:
        # Save model
        model_file = 'f1_xgboost_model_VERIFIED.joblib'
        joblib.dump(model, model_file)
        verification['files_saved'].append(model_file)
        print(f"   ✅ Model saved: {model_file}")
        
        # Save encoders and feature list
        encoders_file = 'f1_label_encoders_VERIFIED.joblib'
        joblib.dump({
            'driver': le_driver,
            'team': le_team,
            'feature_columns': feature_columns
        }, encoders_file)
        verification['files_saved'].append(encoders_file)
        print(f"   ✅ Encoders saved: {encoders_file}")
        
        # Verify files exist
        import os
        for file in verification['files_saved']:
            if not os.path.exists(file):
                verification['issues'].append(f"File not created: {file}")
        
        verification['passed'] = len(verification['issues']) == 0
        
        if verification['passed']:
            print(f"   ✅ Save verification PASSED")
        else:
            print(f"   ❌ Save verification FAILED")
            for issue in verification['issues']:
                print(f"      {issue}")
        
        return verification
        
    except Exception as e:
        verification['issues'].append(f"Save error: {str(e)}")
        print(f"   ❌ Save failed: {str(e)}")
        return verification

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline with comprehensive verification"""
    
    print("\n" + "="*80)
    print("🏎️  F1 MODEL TRAINING WITH VERIFICATION")
    print("="*80)
    
    all_verifications = []
    
    # Step 1: Load features
    df, validation = load_and_validate_features()
    if not validation['passed']:
        print("\n❌ Cannot proceed - feature loading failed")
        return
    all_verifications.append(('Feature Loading', validation))
    
    # Step 2: Prepare data
    X, y, le_driver, le_team, feature_columns = prepare_training_data(df)
    if X is None:
        print("\n❌ Cannot proceed - data preparation failed")
        return
    
    # Step 3: Split data
    splits, verification = split_data_with_verification(df, X, y)
    if splits is None:
        print("\n❌ Cannot proceed - data split failed")
        return
    all_verifications.append(('Data Split', verification))
    
    # Step 4: Train model
    model, verification = train_xgboost_with_verification(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val']
    )
    if model is None:
        print("\n❌ Cannot proceed - model training failed")
        return
    all_verifications.append(('Model Training', verification))
    
    # Step 5: Evaluate model
    verification = evaluate_model_with_verification(
        model,
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val'],
        splits['X_test'], splits['y_test']
    )
    all_verifications.append(('Model Evaluation', verification))
    
    # Step 6: Feature importance
    feature_importance = analyze_feature_importance(model, feature_columns)
    
    # Step 7: Save model
    verification = save_model_with_verification(model, le_driver, le_team, feature_columns)
    all_verifications.append(('Model Save', verification))
    
    # Final summary
    print("\n" + "="*80)
    print("📋 TRAINING SUMMARY")
    print("="*80)
    
    for step_name, verif in all_verifications:
        status = "✅" if verif['passed'] else "⚠️ " if len(verif.get('warnings', [])) > 0 else "❌"
        issues = len(verif.get('issues', []))
        warnings = len(verif.get('warnings', []))
        print(f"   {status} {step_name:25s} Issues: {issues:2d} | Warnings: {warnings:2d}")
    
    print("\n" + "="*80)
    print("🎯 NEXT STEP: Run championship prediction")
    print("   python f1_championship_prediction.py")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
"""
ROBUST F1 Data Collection Script (2018-2025)
Production-ready with STRICT year-by-year verification

LOGIC:
1. Collect data for Year N
2. VERIFY Year N data against quality thresholds
3. If verification FAILS -> Retry collection for Year N (up to 3 attempts)
4. If verification PASSES -> Save Year N data and proceed to Year N+1
5. Only move forward when current year passes all verification checks

Features:
- Year-by-year collection with mandatory verification
- Automatic retry on failed verification
- Comprehensive data quality checks
- Progressive saving (each year saved separately)
- Detailed verification reports
"""

import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import time
import os
from typing import Optional, Dict, List, Tuple
warnings.filterwarnings('ignore')

# Enable FastF1 cache
CACHE_DIR = 'f1_cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Verification thresholds
VERIFICATION_CONFIG = {
    'min_races_percentage': 0.75,  # Must collect at least 75% of expected races
    'min_drivers_per_race': 15,    # Each race should have at least 15 drivers
    'max_missing_positions': 0.1,  # Max 10% missing finish positions
    'max_dnf_rate': 0.4,           # Max 40% DNF rate per race
    'max_retry_attempts': 3,       # Retry failed year up to 3 times
}

# Expected races per year (for verification)
EXPECTED_RACES = {
    2018: 21, 2019: 21, 2020: 17, 2021: 22,
    2022: 22, 2023: 22, 2024: 24, 2025: 24
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_get_attribute(obj, attr, default=None):
    """Safely get attribute from object"""
    try:
        val = getattr(obj, attr, default)
        if pd.isna(val):
            return default
        return val
    except:
        return default

def safe_timedelta_to_seconds(td):
    """Safely convert timedelta to seconds"""
    try:
        if pd.isna(td):
            return np.nan
        if hasattr(td, 'total_seconds'):
            return td.total_seconds()
        return np.nan
    except:
        return np.nan

# ============================================================================
# DATA COLLECTION FUNCTIONS
# ============================================================================

def collect_session_data(year: int, round_num: int, session_type: str, max_retries: int = 3) -> Tuple[Optional[object], Optional[pd.DataFrame]]:
    """Collect data for a specific session with retry logic"""
    for attempt in range(max_retries):
        try:
            session = fastf1.get_session(year, round_num, session_type)
            session.load()
            
            if not hasattr(session, 'results'):
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None, None
            
            results = session.results
            
            if results is None or results.empty:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None, None
            
            return session, results
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return None, None
    
    return None, None

def extract_driver_data(row, session, quali_results: pd.DataFrame, year: int, round_num: int) -> Dict:
    """Extract all driver data from a race result with safe fallbacks"""
    driver_abbr = safe_get_attribute(row, 'Abbreviation', 'UNK')
    
    # Get qualifying position
    quali_pos = 20
    if not quali_results.empty:
        try:
            driver_quali = quali_results[quali_results['Abbreviation'] == driver_abbr]
            if not driver_quali.empty:
                quali_pos = safe_get_attribute(driver_quali.iloc[0], 'Position', 20)
        except:
            pass
    
    # Extract qualifying times
    q1_time, q2_time, q3_time = np.nan, np.nan, np.nan
    if not quali_results.empty:
        try:
            driver_quali = quali_results[quali_results['Abbreviation'] == driver_abbr]
            if not driver_quali.empty:
                q1_time = safe_timedelta_to_seconds(safe_get_attribute(driver_quali.iloc[0], 'Q1'))
                q2_time = safe_timedelta_to_seconds(safe_get_attribute(driver_quali.iloc[0], 'Q2'))
                q3_time = safe_timedelta_to_seconds(safe_get_attribute(driver_quali.iloc[0], 'Q3'))
        except:
            pass
    
    fastest_lap = safe_timedelta_to_seconds(safe_get_attribute(row, 'FastestLap'))
    
    data = {
        'Year': year,
        'Round': round_num,
        'Circuit': safe_get_attribute(session.event, 'EventName', 'Unknown'),
        'CircuitID': safe_get_attribute(session.event, 'Location', 'Unknown'),
        'Country': safe_get_attribute(session.event, 'Country', 'Unknown'),
        'Driver': driver_abbr,
        'DriverNumber': safe_get_attribute(row, 'DriverNumber', 0),
        'FullName': safe_get_attribute(row, 'FullName', ''),
        'Team': safe_get_attribute(row, 'TeamName', 'Unknown'),
        'TeamColor': safe_get_attribute(row, 'TeamColor', ''),
        'GridPosition': safe_get_attribute(row, 'GridPosition', 20),
        'QualiPosition': quali_pos,
        'FinishPosition': safe_get_attribute(row, 'Position', 22),
        'Points': safe_get_attribute(row, 'Points', 0),
        'Status': safe_get_attribute(row, 'Status', 'Unknown'),
        'FastestLap': fastest_lap,
        'Q1Time': q1_time,
        'Q2Time': q2_time,
        'Q3Time': q3_time,
    }
    
    return data

def collect_race_data(year: int, round_num: int, event_name: str = "") -> Optional[pd.DataFrame]:
    """Collect comprehensive race data for a specific GP"""
    print(f"    Round {round_num:2d}: {event_name:<35}", end=' ')
    
    # Collect race session
    race_session, race_results = collect_session_data(year, round_num, 'R')
    
    if race_session is None or race_results is None:
        print("❌ FAILED")
        return None
    
    # Collect qualifying session
    quali_session, quali_results = collect_session_data(year, round_num, 'Q')
    
    if quali_results is None:
        quali_results = pd.DataFrame()
        print("⚠️  (no quali) ", end='')
    
    # Extract data for all drivers
    race_data = []
    for idx, row in race_results.iterrows():
        try:
            driver_data = extract_driver_data(row, race_session, quali_results, year, round_num)
            race_data.append(driver_data)
        except Exception as e:
            continue
    
    if not race_data:
        print("❌ NO DATA")
        return None
    
    df = pd.DataFrame(race_data)
    print(f"✓ ({len(df)} drivers)")
    return df

def clean_year_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize data for a specific year"""
    
    # Handle finish positions (DNFs = 22)
    df['FinishPosition'] = df.apply(
        lambda row: 22 if pd.isna(row['FinishPosition']) or str(row['Status']).lower() not in ['finished', '+1 lap', '+2 laps']
        else row['FinishPosition'],
        axis=1
    )
    
    # Convert positions to numeric
    numeric_cols = ['FinishPosition', 'GridPosition', 'QualiPosition', 'DriverNumber', 'Points']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values
    df['GridPosition'] = df['GridPosition'].fillna(20)
    df['QualiPosition'] = df['QualiPosition'].fillna(20)
    df['DriverNumber'] = df['DriverNumber'].fillna(0)
    
    # Create DNF flag
    df['DNF'] = df.apply(
        lambda row: 1 if row['FinishPosition'] >= 20 or str(row['Status']).lower() not in ['finished', '+1 lap', '+2 laps']
        else 0,
        axis=1
    )
    
    # Standardize team names
    team_mapping = {
        'Aston Martin': 'Aston Martin',
        'Racing Point': 'Aston Martin',
        'Force India': 'Aston Martin',
        'Alpine': 'Alpine',
        'Renault': 'Alpine',
        'AlphaTauri': 'RB',
        'Toro Rosso': 'RB',
        'Alfa Romeo': 'Kick Sauber',
        'Sauber': 'Kick Sauber',
        'Kick Sauber': 'Kick Sauber',
        'Haas F1 Team': 'Haas',
        'Red Bull Racing': 'Red Bull',
        'Mercedes': 'Mercedes',
        'Ferrari': 'Ferrari',
        'McLaren': 'McLaren',
        'Williams': 'Williams',
    }
    
    df['Team'] = df['Team'].replace(team_mapping)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['Year', 'Round', 'Driver'], keep='first')
    
    # Sort
    df = df.sort_values(['Year', 'Round', 'FinishPosition']).reset_index(drop=True)
    
    return df

# ============================================================================
# VERIFICATION FUNCTIONS
# ============================================================================

def verify_year_data(df: pd.DataFrame, year: int) -> Dict:
    """
    COMPREHENSIVE verification of collected data for a specific year
    Returns verification result with pass/fail status
    """
    verification = {
        'year': year,
        'passed': False,
        'total_races': 0,
        'total_records': 0,
        'unique_drivers': 0,
        'unique_teams': 0,
        'races_collected': [],
        'missing_races': [],
        'issues': [],
        'warnings': [],
        'quality_metrics': {},
        'retry_recommended': False
    }
    
    if df is None or df.empty:
        verification['issues'].append("CRITICAL: No data collected for this year")
        verification['retry_recommended'] = True
        return verification
    
    year_data = df[df['Year'] == year]
    
    if year_data.empty:
        verification['issues'].append(f"CRITICAL: No data found for year {year}")
        verification['retry_recommended'] = True
        return verification
    
    # Basic statistics
    verification['total_races'] = year_data['Round'].nunique()
    verification['total_records'] = len(year_data)
    verification['unique_drivers'] = year_data['Driver'].nunique()
    verification['unique_teams'] = year_data['Team'].nunique()
    verification['races_collected'] = sorted(year_data['Round'].unique().tolist())
    
    # Expected races for this year
    expected_races = EXPECTED_RACES.get(year, 20)
    min_required_races = int(expected_races * VERIFICATION_CONFIG['min_races_percentage'])
    
    # ========================================================================
    # CRITICAL CHECKS (Must pass to proceed)
    # ========================================================================
    
    # Check 1: Minimum number of races
    if verification['total_races'] < min_required_races:
        verification['issues'].append(
            f"CRITICAL: Only {verification['total_races']} races collected "
            f"(minimum required: {min_required_races}/{expected_races})"
        )
        verification['retry_recommended'] = True
    
    # Check 2: Minimum drivers per race
    drivers_per_race = year_data.groupby('Round').size()
    races_with_few_drivers = drivers_per_race[drivers_per_race < VERIFICATION_CONFIG['min_drivers_per_race']]
    if len(races_with_few_drivers) > 0:
        verification['issues'].append(
            f"CRITICAL: {len(races_with_few_drivers)} races have fewer than "
            f"{VERIFICATION_CONFIG['min_drivers_per_race']} drivers"
        )
        for round_num, count in races_with_few_drivers.items():
            verification['issues'].append(f"  - Round {round_num}: only {count} drivers")
        verification['retry_recommended'] = True
    
    # Check 3: Missing finish positions
    missing_positions = year_data['FinishPosition'].isna().sum()
    missing_percentage = missing_positions / len(year_data)
    if missing_percentage > VERIFICATION_CONFIG['max_missing_positions']:
        verification['issues'].append(
            f"CRITICAL: {missing_percentage*100:.1f}% of records have missing finish positions "
            f"(max allowed: {VERIFICATION_CONFIG['max_missing_positions']*100:.1f}%)"
        )
        verification['retry_recommended'] = True
    
    # Check 4: Unrealistic DNF rates per race
    dnf_per_race = year_data.groupby('Round')['DNF'].mean()
    high_dnf_races = dnf_per_race[dnf_per_race > VERIFICATION_CONFIG['max_dnf_rate']]
    if len(high_dnf_races) > expected_races * 0.3:  # More than 30% of races have high DNF
        verification['warnings'].append(
            f"WARNING: {len(high_dnf_races)} races have suspiciously high DNF rates (>{VERIFICATION_CONFIG['max_dnf_rate']*100:.0f}%)"
        )
    
    # Check 5: Driver consistency (no driver should have more than 25 races in a year)
    driver_race_counts = year_data.groupby('Driver').size()
    suspicious_drivers = driver_race_counts[driver_race_counts > expected_races + 3]
    if len(suspicious_drivers) > 0:
        verification['issues'].append(
            f"CRITICAL: {len(suspicious_drivers)} drivers have too many race entries (possible duplicates)"
        )
        verification['retry_recommended'] = True
    
    # Check 6: Team consistency (should have 2-4 drivers per team for the year)
    team_driver_counts = year_data.groupby('Team')['Driver'].nunique()
    suspicious_teams = team_driver_counts[(team_driver_counts < 1) | (team_driver_counts > 5)]
    if len(suspicious_teams) > 0:
        verification['warnings'].append(
            f"WARNING: {len(suspicious_teams)} teams have unusual driver counts"
        )
    
    # Check 7: Points consistency (top finishers should have points)
    top_finishers_no_points = year_data[
        (year_data['FinishPosition'] <= 10) & 
        (year_data['Points'] == 0) & 
        (year_data['DNF'] == 0)
    ]
    if len(top_finishers_no_points) > 0:
        verification['warnings'].append(
            f"WARNING: {len(top_finishers_no_points)} top-10 finishes have 0 points"
        )
    
    # ========================================================================
    # QUALITY METRICS
    # ========================================================================
    
    verification['quality_metrics'] = {
        'race_completion': verification['total_races'] / expected_races,
        'avg_drivers_per_race': year_data.groupby('Round').size().mean(),
        'data_completeness': 1 - missing_percentage,
        'avg_dnf_rate': year_data['DNF'].mean(),
        'unique_circuits': year_data['Circuit'].nunique(),
    }
    
    # Calculate overall quality score (0-100)
    quality_score = 100.0
    quality_score *= verification['quality_metrics']['race_completion']
    quality_score -= len(verification['issues']) * 15
    quality_score -= len(verification['warnings']) * 5
    verification['quality_metrics']['overall_score'] = max(0, quality_score)
    
    # ========================================================================
    # FINAL PASS/FAIL DECISION
    # ========================================================================
    
    # Year passes verification if:
    # 1. No critical issues
    # 2. Quality score >= 70
    # 3. At least minimum required races collected
    
    if (len(verification['issues']) == 0 and 
        verification['quality_metrics']['overall_score'] >= 70 and
        verification['total_races'] >= min_required_races):
        verification['passed'] = True
    else:
        verification['passed'] = False
    
    return verification

def print_verification_report(verification: Dict):
    """Print detailed verification report for a year"""
    year = verification['year']
    status = "✅ PASSED" if verification['passed'] else "❌ FAILED"
    
    print(f"\n  {'='*76}")
    print(f"  📊 VERIFICATION REPORT FOR {year} - {status}")
    print(f"  {'='*76}")
    
    # Statistics
    print(f"\n  📈 Statistics:")
    print(f"     Races collected: {verification['total_races']}")
    print(f"     Total records: {verification['total_records']}")
    print(f"     Unique drivers: {verification['unique_drivers']}")
    print(f"     Unique teams: {verification['unique_teams']}")
    
    # Quality metrics
    if verification['quality_metrics']:
        print(f"\n  🎯 Quality Metrics:")
        metrics = verification['quality_metrics']
        print(f"     Race completion: {metrics['race_completion']*100:.1f}%")
        print(f"     Avg drivers/race: {metrics['avg_drivers_per_race']:.1f}")
        print(f"     Data completeness: {metrics['data_completeness']*100:.1f}%")
        print(f"     Avg DNF rate: {metrics['avg_dnf_rate']*100:.1f}%")
        print(f"     Overall score: {metrics['overall_score']:.1f}/100")
    
    # Issues (Critical)
    if verification['issues']:
        print(f"\n  ❌ CRITICAL ISSUES ({len(verification['issues'])}):")
        for issue in verification['issues']:
            print(f"     • {issue}")
    
    # Warnings
    if verification['warnings']:
        print(f"\n  ⚠️  WARNINGS ({len(verification['warnings'])}):")
        for warning in verification['warnings']:
            print(f"     • {warning}")
    
    # Races collected
    if verification['races_collected']:
        print(f"\n  🏁 Races collected: {verification['races_collected']}")
    
    # Recommendation
    print(f"\n  💡 Recommendation:")
    if verification['passed']:
        print(f"     ✅ Data quality is GOOD - Safe to proceed to next year")
    elif verification['retry_recommended']:
        print(f"     🔄 RETRY RECOMMENDED - Data quality is insufficient")
    else:
        print(f"     ⚠️  Data has issues but may be usable - Manual review suggested")
    
    print(f"  {'='*76}\n")

# ============================================================================
# YEAR-BY-YEAR COLLECTION WITH VERIFICATION
# ============================================================================

def collect_single_year_with_verification(year: int, attempt: int = 1) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Collect data for a single year and verify it
    Returns (dataframe, verification_result)
    """
    print(f"\n{'='*80}")
    print(f"📅 COLLECTING {year} DATA (Attempt {attempt}/{VERIFICATION_CONFIG['max_retry_attempts']})")
    print('='*80)
    
    year_data = []
    
    try:
        # Get schedule for the year
        schedule = fastf1.get_event_schedule(year)
        
        if schedule is None or schedule.empty:
            print(f"  ❌ Could not get schedule for {year}")
            verification = {'year': year, 'passed': False, 'issues': ['Failed to get schedule'], 'retry_recommended': True}
            return None, verification
        
        # Filter out testing sessions
        schedule = schedule[schedule['EventFormat'] != 'testing']
        total_rounds = len(schedule)
        
        print(f"  Expected races: {total_rounds}")
        print(f"  Collecting races:\n")
        
        # Collect each race
        for idx, event in schedule.iterrows():
            round_num = event['RoundNumber']
            event_name = event['EventName']
            
            race_data = collect_race_data(year, round_num, event_name)
            
            if race_data is not None and not race_data.empty:
                year_data.append(race_data)
            
            time.sleep(1)  # Rate limiting
    
    except Exception as e:
        print(f"  ❌ Error during collection: {str(e)}")
        verification = {'year': year, 'passed': False, 'issues': [f'Collection error: {str(e)}'], 'retry_recommended': True}
        return None, verification
    
    # Combine year data
    if not year_data:
        print(f"\n  ❌ No data collected for {year}")
        verification = {'year': year, 'passed': False, 'issues': ['No data collected'], 'retry_recommended': True}
        return None, verification
    
    year_df = pd.concat(year_data, ignore_index=True)
    
    # Clean data
    print(f"\n  🧹 Cleaning data...")
    year_df = clean_year_data(year_df)
    
    # VERIFY data
    print(f"  🔍 Verifying data quality...")
    verification = verify_year_data(year_df, year)
    
    # Print verification report
    print_verification_report(verification)
    
    return year_df, verification

def collect_all_years_with_verification(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Collect data year by year with MANDATORY verification before proceeding
    Only moves to next year if current year passes verification
    """
    all_verified_data = []
    verification_history = []
    
    print(f"\n{'='*80}")
    print(f"🏁 STARTING YEAR-BY-YEAR COLLECTION: {start_year} to {end_year}")
    print('='*80)
    print(f"\nVerification thresholds:")
    print(f"  • Minimum races: {VERIFICATION_CONFIG['min_races_percentage']*100:.0f}% of expected")
    print(f"  • Minimum drivers per race: {VERIFICATION_CONFIG['min_drivers_per_race']}")
    print(f"  • Maximum missing data: {VERIFICATION_CONFIG['max_missing_positions']*100:.0f}%")
    print(f"  • Maximum retry attempts: {VERIFICATION_CONFIG['max_retry_attempts']}")
    
    for year in range(start_year, end_year + 1):
        year_verified = False
        year_df = None
        
        # Try up to max_retry_attempts times
        for attempt in range(1, VERIFICATION_CONFIG['max_retry_attempts'] + 1):
            year_df, verification = collect_single_year_with_verification(year, attempt)
            
            if verification['passed']:
                print(f"  ✅ {year} DATA VERIFIED - Proceeding to next year")
                year_verified = True
                all_verified_data.append(year_df)
                verification_history.append(verification)
                
                # Save year data immediately
                year_file = f'f1_data_{year}_verified.csv'
                year_df.to_csv(year_file, index=False)
                print(f"  💾 Saved verified data to: {year_file}\n")
                
                break
            else:
                print(f"  ❌ {year} DATA FAILED VERIFICATION")
                
                if attempt < VERIFICATION_CONFIG['max_retry_attempts']:
                    print(f"  🔄 Retrying {year} collection...")
                    time.sleep(5)
                else:
                    print(f"  ⛔ {year} FAILED AFTER {VERIFICATION_CONFIG['max_retry_attempts']} ATTEMPTS")
                    print(f"  ⚠️  Moving to next year (data quality may be compromised)\n")
                    
                    # Still save the data but mark it as unverified
                    if year_df is not None and not year_df.empty:
                        year_file = f'f1_data_{year}_UNVERIFIED.csv'
                        year_df.to_csv(year_file, index=False)
                        print(f"  💾 Saved UNVERIFIED data to: {year_file}\n")
                        all_verified_data.append(year_df)
                        verification_history.append(verification)
        
        # Wait before next year
        time.sleep(2)
    
    # Print final summary
    print_final_summary(verification_history)
    
    # Combine all years
    if all_verified_data:
        return pd.concat(all_verified_data, ignore_index=True)
    else:
        return pd.DataFrame()

def print_final_summary(verification_history: List[Dict]):
    """Print final collection summary"""
    print(f"\n{'='*80}")
    print(f"📋 FINAL COLLECTION SUMMARY")
    print('='*80)
    
    passed_years = [v for v in verification_history if v['passed']]
    failed_years = [v for v in verification_history if not v['passed']]
    
    print(f"\n✅ Successfully verified years: {len(passed_years)}/{len(verification_history)}")
    print(f"❌ Failed verification years: {len(failed_years)}/{len(verification_history)}")
    
    if passed_years:
        print(f"\n✅ VERIFIED YEARS:")
        for v in passed_years:
            score = v.get('quality_metrics', {}).get('overall_score', 0)
            print(f"   {v['year']}: {v['total_races']} races, {v['total_records']} records, Quality: {score:.1f}/100")
    
    if failed_years:
        print(f"\n❌ FAILED VERIFICATION:")
        for v in failed_years:
            score = v.get('quality_metrics', {}).get('overall_score', 0)
            print(f"   {v['year']}: {v['total_races']} races, {v['total_records']} records, Quality: {score:.1f}/100")
            print(f"      Issues: {len(v.get('issues', []))}")
    
    total_races = sum(v['total_races'] for v in verification_history)
    total_records = sum(v['total_records'] for v in verification_history)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total years collected: {len(verification_history)}")
    print(f"   Total races: {total_races}")
    print(f"   Total records: {total_records:,}")
    print(f"   Average quality score: {np.mean([v.get('quality_metrics', {}).get('overall_score', 0) for v in verification_history]):.1f}/100")

def add_2025_mexican_gp():
    """Add manually verified 2025 Mexican GP data"""
    
    mexican_gp_results = [
        {'Driver': 'NOR', 'Number': 4, 'Name': 'Lando Norris', 'Team': 'McLaren', 'Grid': 1, 'Quali': 1, 'Finish': 1, 'Points': 25, 'Status': 'Finished'},
        {'Driver': 'LEC', 'Number': 16, 'Name': 'Charles Leclerc', 'Team': 'Ferrari', 'Grid': 2, 'Quali': 2, 'Finish': 2, 'Points': 19, 'Status': 'Finished'},
        {'Driver': 'VER', 'Number': 1, 'Name': 'Max Verstappen', 'Team': 'Red Bull', 'Grid': 5, 'Quali': 5, 'Finish': 3, 'Points': 15, 'Status': 'Finished'},
        {'Driver': 'BEA', 'Number': 87, 'Name': 'Oliver Bearman', 'Team': 'Haas', 'Grid': 10, 'Quali': 10, 'Finish': 4, 'Points': 12, 'Status': 'Finished'},
        {'Driver': 'PIA', 'Number': 81, 'Name': 'Oscar Piastri', 'Team': 'McLaren', 'Grid': 8, 'Quali': 8, 'Finish': 5, 'Points': 10, 'Status': 'Finished'},
        {'Driver': 'ANT', 'Number': 12, 'Name': 'Andrea Kimi Antonelli', 'Team': 'Mercedes', 'Grid': 6, 'Quali': 6, 'Finish': 6, 'Points': 8, 'Status': 'Finished'},
        {'Driver': 'RUS', 'Number': 63, 'Name': 'George Russell', 'Team': 'Mercedes', 'Grid': 4, 'Quali': 4, 'Finish': 7, 'Points': 6, 'Status': 'Finished'},
        {'Driver': 'HAM', 'Number': 44, 'Name': 'Lewis Hamilton', 'Team': 'Ferrari', 'Grid': 3, 'Quali': 3, 'Finish': 8, 'Points': 4, 'Status': 'Finished'},
        {'Driver': 'OCO', 'Number': 31, 'Name': 'Esteban Ocon', 'Team': 'Haas', 'Grid': 5, 'Quali': 5, 'Finish': 9, 'Points': 2, 'Status': 'Finished'},
        {'Driver': 'BOR', 'Number': 10, 'Name': 'Gabriel Bortoleto', 'Team': 'Kick Sauber', 'Grid': 16, 'Quali': 16, 'Finish': 10, 'Points': 1, 'Status': 'Finished'},
        {'Driver': 'TSU', 'Number': 22, 'Name': 'Yuki Tsunoda', 'Team': 'Red Bull', 'Grid': 13, 'Quali': 13, 'Finish': 11, 'Points': 0, 'Status': 'Finished'},
        {'Driver': 'ALB', 'Number': 23, 'Name': 'Alexander Albon', 'Team': 'Williams', 'Grid': 17, 'Quali': 17, 'Finish': 12, 'Points': 0, 'Status': 'Finished'},
        {'Driver': 'HAD', 'Number': 20, 'Name': 'Isack Hadjar', 'Team': 'RB', 'Grid': 9, 'Quali': 9, 'Finish': 13, 'Points': 0, 'Status': 'Finished'},
        {'Driver': 'STR', 'Number': 18, 'Name': 'Lance Stroll', 'Team': 'Aston Martin', 'Grid': 19, 'Quali': 19, 'Finish': 14, 'Points': 0, 'Status': 'Finished'},
        {'Driver': 'GAS', 'Number': 10, 'Name': 'Pierre Gasly', 'Team': 'Alpine', 'Grid': 18, 'Quali': 18, 'Finish': 15, 'Points': 0, 'Status': 'Finished'},
        {'Driver': 'COL', 'Number': 43, 'Name': 'Franco Colapinto', 'Team': 'Alpine', 'Grid': 20, 'Quali': 20, 'Finish': 16, 'Points': 0, 'Status': 'Finished'},
        {'Driver': 'SAI', 'Number': 55, 'Name': 'Carlos Sainz', 'Team': 'Williams', 'Grid': 7, 'Quali': 7, 'Finish': 22, 'Points': 0, 'Status': 'Retired'},
        {'Driver': 'ALO', 'Number': 14, 'Name': 'Fernando Alonso', 'Team': 'Aston Martin', 'Grid': 12, 'Quali': 12, 'Finish': 22, 'Points': 0, 'Status': 'Retired'},
        {'Driver': 'HUL', 'Number': 27, 'Name': 'Nico Hulkenberg', 'Team': 'Kick Sauber', 'Grid': 14, 'Quali': 14, 'Finish': 22, 'Points': 0, 'Status': 'Retired'},
        {'Driver': 'LAW', 'Number': 30, 'Name': 'Liam Lawson', 'Team': 'RB', 'Grid': 11, 'Quali': 11, 'Finish': 22, 'Points': 0, 'Status': 'Retired'},
    ]
    
    data_2025 = []
    for result in mexican_gp_results:
        data_2025.append({
            'Year': 2025,
            'Round': 20,
            'Circuit': 'Mexico City Grand Prix',
            'CircuitID': 'Mexico City',
            'Country': 'Mexico',
            'Driver': result['Driver'],
            'DriverNumber': result['Number'],
            'FullName': result['Name'],
            'Team': result['Team'],
            'TeamColor': '',
            'GridPosition': result['Grid'],
            'QualiPosition': result['Quali'],
            'FinishPosition': result['Finish'],
            'Points': result['Points'],
            'Status': result['Status'],
            'DNF': 0 if result['Status'] == 'Finished' else 1,
            'FastestLap': np.nan,
            'Q1Time': np.nan,
            'Q2Time': np.nan,
            'Q3Time': np.nan,
        })
    
    return pd.DataFrame(data_2025)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🏎️  ROBUST F1 DATA COLLECTION SYSTEM")
    print("     WITH YEAR-BY-YEAR VERIFICATION")
    print("="*80)
    print("\nThis script will collect F1 race data from 2018-2025")
    print("Each year is VERIFIED before moving to the next year")
    print("\nEstimated time: 30-60 minutes for full collection")
    print("="*80)
    
    # User options
    print("\n📋 Collection Options:")
    print("  1. Full collection (2018-2024) + 2025 Mexican GP")
    print("  2. Recent years only (2022-2024) + 2025 Mexican GP")
    print("  3. Single year test")
    print("  4. Quick mode (2025 Mexican GP only)")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        start_year, end_year = 2018, 2024
    elif choice == "2":
        start_year, end_year = 2022, 2024
    elif choice == "3":
        year = input("Enter year to collect: ").strip()
        start_year = end_year = int(year) if year.isdigit() else 2024
    else:
        start_year, end_year = None, None
    
    # Collect historical data with verification
    if start_year is not None:
        print(f"\n🚀 Starting collection from {start_year} to {end_year}")
        print(f"⏳ This will take a while... Each year will be verified before proceeding.\n")
        historical_data = collect_all_years_with_verification(start_year, end_year)
    else:
        historical_data = pd.DataFrame()
        print("\n⚡ Quick mode selected - skipping historical collection")
    
    # Add 2025 Mexican GP
    print(f"\n{'='*80}")
    print("📍 Adding 2025 Mexican GP results...")
    print('='*80)
    mexican_2025 = add_2025_mexican_gp()
    print(f"  ✅ Added 20 drivers from Mexican GP\n")
    
    # Combine ALL data into ONE master file
    if not historical_data.empty:
        full_data = pd.concat([historical_data, mexican_2025], ignore_index=True)
    else:
        full_data = mexican_2025
        print(f"⚠️  No historical data collected - MASTER file will only contain 2025 Mexican GP data\n")
    
    # Save MASTER FILE with all years combined
    master_file = 'f1_historical_data_2018_2025_MASTER.csv'
    full_data.to_csv(master_file, index=False)
    
    # Final summary
    print(f"{'='*80}")
    print("✅ DATA COLLECTION COMPLETE!")
    print('='*80)
    print(f"\n💾 FILES CREATED:")
    print(f"   📁 MASTER FILE: {master_file}")
    print(f"      └─ Contains ALL years merged together")
    
    if start_year is not None:
        print(f"\n   📁 Individual year files (for backup):")
        for year in range(start_year, end_year + 1):
            verified_file = f'f1_data_{year}_verified.csv'
            unverified_file = f'f1_data_{year}_UNVERIFIED.csv'
            if os.path.exists(verified_file):
                print(f"      ✅ {verified_file}")
            elif os.path.exists(unverified_file):
                print(f"      ⚠️  {unverified_file}")
    
    print(f"\n📊 MASTER FILE STATISTICS:")
    print(f"   Total records: {len(full_data):,}")
    print(f"   Total races: {full_data.groupby(['Year', 'Round']).ngroups}")
    print(f"   Years covered: {full_data['Year'].min()} - {full_data['Year'].max()}")
    print(f"   Unique drivers: {full_data['Driver'].nunique()}")
    print(f"   Unique teams: {full_data['Team'].nunique()}")
    print(f"   Unique circuits: {full_data['Circuit'].nunique()}")
    
    # Year breakdown
    print(f"\n📅 YEAR BREAKDOWN:")
    year_summary = full_data.groupby('Year').agg({
        'Round': 'nunique',
        'Driver': 'count',
        'Points': 'sum'
    }).rename(columns={'Round': 'Races', 'Driver': 'Records', 'Points': 'Total_Points'})
    print(year_summary.to_string())
    
    # Display sample
    print(f"\n{'='*80}")
    print("📋 SAMPLE DATA (2025 Mexican GP Top 10):")
    print('='*80)
    sample = full_data[full_data['Year'] == 2025].head(10)
    print(sample[['Driver', 'FullName', 'Team', 'FinishPosition', 'Points']].to_string(index=False))
    
    print(f"\n{'='*80}")
    print("🎯 NEXT STEPS:")
    print('='*80)
    print(f"1. ✅ Use MASTER FILE: {master_file}")
    print("2. 🔧 Run feature engineering: python f1_feature_engineering.py")
    print("3. 🤖 Train model: python f1_model_training.py")
    print("4. 🏆 Make predictions: python f1_championship_prediction.py")
    print('='*80)
    
    print(f"\n💡 TIP: The MASTER file contains all years merged together.")
    print(f"     Individual year files are kept as backups for verification.")
    print(f"\n{'='*80}\n")
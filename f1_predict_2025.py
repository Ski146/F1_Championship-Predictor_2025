"""
F1 2025 Championship Prediction - COMPLETE WITH VERIFICATION
Predicts final championship standings for remaining 4 races

Current Status: After Mexican GP (Round 20/24)
Remaining: Brazil (Sprint), Las Vegas, Qatar (Sprint), Abu Dhabi
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ============================================================================
# CURRENT STANDINGS AFTER MEXICAN GP 2025 (Round 20/24)
# ============================================================================

CURRENT_STANDINGS = {
    'Driver': {
        'NOR': {'name': 'Lando Norris', 'team': 'McLaren', 'points': 357},
        'PIA': {'name': 'Oscar Piastri', 'team': 'McLaren', 'points': 356},
        'VER': {'name': 'Max Verstappen', 'team': 'Red Bull', 'points': 321},
        'RUS': {'name': 'George Russell', 'team': 'Mercedes', 'points': 258},
        'LEC': {'name': 'Charles Leclerc', 'team': 'Ferrari', 'points': 210},
        'HAM': {'name': 'Lewis Hamilton', 'team': 'Ferrari', 'points': 189},
        'SAI': {'name': 'Carlos Sainz', 'team': 'Williams', 'points': 68},
        'TSU': {'name': 'Yuki Tsunoda', 'team': 'Red Bull', 'points': 47},
        'ANT': {'name': 'Andrea Kimi Antonelli', 'team': 'Mercedes', 'points': 43},
        'ALO': {'name': 'Fernando Alonso', 'team': 'Aston Martin', 'points': 38},
        'HUL': {'name': 'Nico Hulkenberg', 'team': 'Kick Sauber', 'points': 37},
        'BEA': {'name': 'Oliver Bearman', 'team': 'Haas', 'points': 21},
        'OCO': {'name': 'Esteban Ocon', 'team': 'Haas', 'points': 21},
        'LAW': {'name': 'Liam Lawson', 'team': 'RB', 'points': 7},
        'STR': {'name': 'Lance Stroll', 'team': 'Aston Martin', 'points': 6},
        'ALB': {'name': 'Alexander Albon', 'team': 'Williams', 'points': 4},
        'HAD': {'name': 'Isack Hadjar', 'team': 'RB', 'points': 4},
        'GAS': {'name': 'Pierre Gasly', 'team': 'Alpine', 'points': 4},
        'COL': {'name': 'Franco Colapinto', 'team': 'Alpine', 'points': 0},
        'BOR': {'name': 'Gabriel Bortoleto', 'team': 'Kick Sauber', 'points': 0},
    }
}

CONSTRUCTOR_STANDINGS = {
    'McLaren': 713,
    'Red Bull': 368,
    'Mercedes': 301,
    'Ferrari': 399,
    'Williams': 72,
    'Haas': 42,
    'Aston Martin': 44,
    'Kick Sauber': 37,
    'RB': 11,
    'Alpine': 4,
}

REMAINING_RACES = [
    {'round': 21, 'name': 'São Paulo GP', 'circuit': 'Interlagos', 'date': '2025-11-09', 'sprint': True, 'type': 'traditional'},
    {'round': 22, 'name': 'Las Vegas GP', 'circuit': 'Las Vegas', 'date': '2025-11-22', 'sprint': False, 'type': 'street'},
    {'round': 23, 'name': 'Qatar GP', 'circuit': 'Lusail', 'date': '2025-11-30', 'sprint': True, 'type': 'high_speed'},
    {'round': 24, 'name': 'Abu Dhabi GP', 'circuit': 'Yas Marina', 'date': '2025-12-07', 'sprint': False, 'type': 'traditional'},
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_points(position, is_sprint=False):
    """Calculate F1 points"""
    if is_sprint:
        return {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}.get(position, 0)
    else:
        return {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}.get(position, 0)

# ============================================================================
# SIMPLIFIED PREDICTION (Statistical Model)
# ============================================================================

def predict_race_statistical(driver_abbr, race_info):
    """Predict finish position using statistical approach"""
    
    driver = CURRENT_STANDINGS['Driver'][driver_abbr]
    
    # Base prediction from current championship position
    champ_pos = list(CURRENT_STANDINGS['Driver'].keys()).index(driver_abbr) + 1
    base_position = champ_pos
    
    # Team performance adjustments
    team_adjustments = {
        'McLaren': -2.0,  # Dominant
        'Ferrari': -0.5,
        'Red Bull': 0.0,
        'Mercedes': 0.5,
        'Williams': 3.0,
        'Haas': 4.0,
        'Aston Martin': 4.5,
        'Alpine': 5.5,
        'RB': 5.0,
        'Kick Sauber': 6.0,
    }
    
    team_adj = team_adjustments.get(driver['team'], 0)
    
    # Circuit type adjustments
    circuit_adjustments = {
        'McLaren': {'street': -0.5, 'high_speed': -0.3, 'traditional': -0.2},
        'Ferrari': {'street': -0.2, 'high_speed': -0.4, 'traditional': -0.3},
        'Red Bull': {'street': 0.3, 'high_speed': 0.0, 'traditional': 0.2},
    }
    
    circuit_adj = circuit_adjustments.get(driver['team'], {}).get(race_info['type'], 0)
    
    # Add realistic variance
    variance = np.random.normal(0, 1.5)
    
    predicted = base_position + team_adj + circuit_adj + variance
    
    return max(1, min(20, predicted))

# ============================================================================
# SIMULATE CHAMPIONSHIP
# ============================================================================

def simulate_championship(verbose=True):
    """Simulate remaining races and predict final standings"""
    
    if verbose:
        print("\n" + "="*80)
        print("🏎️  F1 2025 CHAMPIONSHIP PREDICTION")
        print("="*80)
        print(f"\nCurrent Status: After Round 20/24 (Mexican GP)")
        print(f"Remaining: 4 races (2 with sprint)")
        print(f"Maximum points available: 138 (4 races + 2 sprints)")
        
        print(f"\n📊 Current Top 5:")
        for i, (abbr, info) in enumerate(list(CURRENT_STANDINGS['Driver'].items())[:5], 1):
            gap = "" if i == 1 else f"(-{list(CURRENT_STANDINGS['Driver'].values())[0]['points'] - info['points']})"
            print(f"   {i}. {info['name']:25s} {info['team']:12s} {info['points']:3d} pts {gap}")
        
        print(f"\n🔥 KEY BATTLE: Norris vs Piastri - separated by just 1 point!")
    
    # Initialize running totals
    driver_points = {abbr: info['points'] for abbr, info in CURRENT_STANDINGS['Driver'].items()}
    constructor_points = CONSTRUCTOR_STANDINGS.copy()
    
    race_by_race_results = []
    
    # Simulate each race
    for race in REMAINING_RACES:
        if verbose:
            print(f"\n{'='*80}")
            print(f"📍 Round {race['round']}: {race['name']}")
            if race['sprint']:
                print(f"   ⚡ SPRINT WEEKEND")
            print(f"   📅 Date: {race['date']} | Type: {race['type'].upper()}")
            print("="*80)
        
        # Predict positions
        predictions = []
        for abbr in CURRENT_STANDINGS['Driver'].keys():
            pos = predict_race_statistical(abbr, race)
            predictions.append((abbr, pos))
        
        # Sort and assign positions
        predictions.sort(key=lambda x: x[1])
        race_results = [(abbr, i+1) for i, (abbr, _) in enumerate(predictions)]
        
        if verbose:
            print(f"\n🏁 Race Results:")
            for abbr, pos in race_results[:10]:
                info = CURRENT_STANDINGS['Driver'][abbr]
                pts = calculate_points(pos)
                print(f"   {pos:2d}. {info['name']:25s} {info['team']:12s} +{pts:2d} pts")
        
        # Award points
        for abbr, pos in race_results:
            info = CURRENT_STANDINGS['Driver'][abbr]
            pts = calculate_points(pos)
            driver_points[abbr] += pts
            constructor_points[info['team']] += pts
        
        # Sprint race
        if race['sprint']:
            if verbose:
                print(f"\n   ⚡ Sprint Results:")
            
            for abbr, pos in race_results[:8]:
                info = CURRENT_STANDINGS['Driver'][abbr]
                sprint_pts = calculate_points(pos, is_sprint=True)
                driver_points[abbr] += sprint_pts
                constructor_points[info['team']] += sprint_pts
                
                if verbose:
                    print(f"      {pos:2d}. {info['name']:25s} +{sprint_pts} pts")
        
        race_by_race_results.append({
            'race': race['name'],
            'results': race_results[:10]
        })
    
    # Final standings
    if verbose:
        print("\n" + "="*80)
        print("🏆 PREDICTED FINAL 2025 CHAMPIONSHIP STANDINGS")
        print("="*80)
        
        print("\n👤 DRIVERS' CHAMPIONSHIP:")
        print("-" * 80)
        sorted_drivers = sorted(driver_points.items(), key=lambda x: x[1], reverse=True)
        
        for i, (abbr, points) in enumerate(sorted_drivers[:15], 1):
            info = CURRENT_STANDINGS['Driver'][abbr]
            current = CURRENT_STANDINGS['Driver'][abbr]['points']
            gain = points - current
            gap = "" if i == 1 else f"(-{sorted_drivers[0][1] - points:.0f})"
            trophy = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"{trophy} {i:2d}. {info['name']:25s} {info['team']:12s} {points:3.0f} pts {gap:>8s} (+{gain:.0f})")
        
        print("\n🏢 CONSTRUCTORS' CHAMPIONSHIP:")
        print("-" * 80)
        sorted_constructors = sorted(constructor_points.items(), key=lambda x: x[1], reverse=True)
        
        for i, (team, points) in enumerate(sorted_constructors, 1):
            current = CONSTRUCTOR_STANDINGS[team]
            gain = points - current
            gap = "" if i == 1 else f"(-{sorted_constructors[0][1] - points:.0f})"
            trophy = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"{trophy} {i:2d}. {team:20s} {points:4.0f} pts {gap:>8s} (+{gain:.0f})")
        
        # Analysis
        champion_abbr = sorted_drivers[0][0]
        runner_up_abbr = sorted_drivers[1][0]
        champion = CURRENT_STANDINGS['Driver'][champion_abbr]
        runner_up = CURRENT_STANDINGS['Driver'][runner_up_abbr]
        
        print("\n" + "="*80)
        print("📈 CHAMPIONSHIP ANALYSIS")
        print("="*80)
        
        print(f"\n🏆 PREDICTED CHAMPION: {champion['name']} ({champion['team']})")
        print(f"   Final Points: {sorted_drivers[0][1]:.0f}")
        print(f"   Margin: {sorted_drivers[0][1] - sorted_drivers[1][1]:.0f} points over {runner_up['name']}")
        
        current_gap = CURRENT_STANDINGS['Driver']['NOR']['points'] - CURRENT_STANDINGS['Driver']['PIA']['points']
        print(f"\n📊 TITLE BATTLE:")
        print(f"   Current gap: {current_gap} point(s)")
        print(f"   Points available: 138 (4 races + 2 sprints)")
        print(f"   Status: WIDE OPEN - anything can happen!")
        
        print(f"\n🔑 KEY FACTORS:")
        print(f"   • McLaren's dominance continues")
        print(f"   • Internal team battle between Norris and Piastri")
        print(f"   • Verstappen needs both McLarens to falter")
        print(f"   • Ferrari fighting for 4th place")
        
        print(f"\n⚠️  DISCLAIMER:")
        print(f"   This is a statistical prediction based on current form.")
        print(f"   Actual results will vary due to:")
        print(f"   - Race incidents and DNFs")
        print(f"   - Weather conditions (especially Brazil)")
        print(f"   - Strategy calls and tire degradation")
        print(f"   - Team orders and driver performance")
        
        print("\n" + "="*80)
    
    return {
        'drivers': sorted_drivers,
        'constructors': sorted_constructors,
        'race_results': race_by_race_results
    }

# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def run_monte_carlo(n_simulations=1000):
    """Run multiple simulations for probability analysis"""
    
    print("\n" + "="*80)
    print(f"🎲 MONTE CARLO SIMULATION ({n_simulations} runs)")
    print("="*80)
    
    championship_wins = {}
    
    for i in range(n_simulations):
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i + 1}/{n_simulations}...")
        
        result = simulate_championship(verbose=False)
        champion_abbr = result['drivers'][0][0]
        champion_name = CURRENT_STANDINGS['Driver'][champion_abbr]['name']
        championship_wins[champion_name] = championship_wins.get(champion_name, 0) + 1
    
    print(f"\n🏆 CHAMPIONSHIP WIN PROBABILITIES:")
    print("-" * 80)
    
    sorted_probs = sorted(championship_wins.items(), key=lambda x: x[1], reverse=True)
    for driver, wins in sorted_probs[:5]:
        probability = (wins / n_simulations) * 100
        bar = "█" * int(probability / 2)
        print(f"   {driver:25s} {probability:5.1f}% {bar}")
    
    return sorted_probs

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_predictions(predictions, monte_carlo_results=None):
    """Save predictions to file"""
    
    output_dir = 'predictions_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Drivers predictions
    drivers_df = pd.DataFrame([
        {
            'Position': i,
            'Driver': CURRENT_STANDINGS['Driver'][abbr]['name'],
            'Team': CURRENT_STANDINGS['Driver'][abbr]['team'],
            'Current_Points': CURRENT_STANDINGS['Driver'][abbr]['points'],
            'Predicted_Final_Points': points,
            'Points_Gain': points - CURRENT_STANDINGS['Driver'][abbr]['points']
        }
        for i, (abbr, points) in enumerate(predictions['drivers'], 1)
    ])
    
    filename = f'{output_dir}/drivers_championship_prediction_{timestamp}.csv'
    drivers_df.to_csv(filename, index=False)
    print(f"\n💾 Saved: {filename}")
    
    # Constructors predictions
    constructors_df = pd.DataFrame([
        {
            'Position': i,
            'Team': team,
            'Current_Points': CONSTRUCTOR_STANDINGS[team],
            'Predicted_Final_Points': points,
            'Points_Gain': points - CONSTRUCTOR_STANDINGS[team]
        }
        for i, (team, points) in enumerate(predictions['constructors'], 1)
    ])
    
    filename = f'{output_dir}/constructors_championship_prediction_{timestamp}.csv'
    constructors_df.to_csv(filename, index=False)
    print(f"💾 Saved: {filename}")
    
    if monte_carlo_results:
        mc_df = pd.DataFrame(monte_carlo_results, columns=['Driver', 'Wins'])
        mc_df['Probability'] = (mc_df['Wins'] / mc_df['Wins'].sum() * 100).round(2)
        filename = f'{output_dir}/monte_carlo_probabilities_{timestamp}.csv'
        mc_df.to_csv(filename, index=False)
        print(f"💾 Saved: {filename}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                  F1 2025 CHAMPIONSHIP PREDICTOR                             ║
    ║              Post Mexican GP - Final 4 Races Prediction                     ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run single prediction
    print("\n🎯 Running championship prediction...")
    predictions = simulate_championship(verbose=True)
    
    # Save results
    save_predictions(predictions)
    
    # Ask for Monte Carlo
    print("\n" + "="*80)
    run_mc = input("\n🎲 Run Monte Carlo simulation for win probabilities? (y/n): ").strip().lower()
    
    if run_mc == 'y':
        n_sims = input("How many simulations? (100-10000, default=1000): ").strip()
        n_sims = int(n_sims) if n_sims.isdigit() else 1000
        
        mc_results = run_monte_carlo(n_sims)
        save_predictions(predictions, mc_results)
    
    print("\n" + "="*80)
    print("✅ PREDICTION COMPLETE!")
    print("="*80)
    print("\n🏁 May the best driver win! Good luck to all teams! 🏁\n")
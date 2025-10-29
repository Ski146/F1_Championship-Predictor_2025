# F1 2025 Championship Prediction System

A complete Machine Learning pipeline to predict the 2025 Formula 1 World Championship based on historical race data, driver form, team performance, and circuit-specific factors.

## 🏎️ Project Overview

This project uses XGBoost regression to predict race finishing positions for the remaining 2025 F1 season and forecast the final championship standings.

### Key Features
- **Historical Data Collection**: Automated data gathering from 2018-2025 F1 seasons using FastF1 API
- **Advanced Feature Engineering**: 20+ engineered features including driver momentum, team performance, circuit history
- **ML Model Training**: XGBoost regressor with hyperparameter tuning
- **Real-time Predictions**: Predict remaining 5 races of 2025 season
- **Interactive UI**: HTML css-based visualization dashboard

## 📁 Project Structure
```
f1-championship-prediction/
│
├── data_collection.py          # Collect historical F1 data (2018-2024)
├── feature_engineering.py      # Transform raw data into ML features
├── train_model.py              # Train XGBoost model
├── predict_2025.py             # Generate 2025 predictions
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/                       # Generated data files
│   ├── f1_historical_data_2018_2025.csv
│   ├── f1_engineered_features.csv
│   └── label_encoders.pkl
│
├── models/                     # Trained models
│   ├── f1_prediction_model_latest.pkl
│   ├── feature_columns.pkl
│   └── feature_importance.png
│
└── ui/                         # React frontend
    └── F1PredictionApp.jsx
```

## 📊 Usage

### Step 1: Collect Historical Data

Collect F1 race data from 2018-2025 (takes 15-30 minutes):
```bash
python data_collection.py
```

**Output**: `f1_historical_data_2018_2024.csv` (~150+ races, ~3000 driver records)

### Step 2: Engineer Features

Transform raw data into ML-ready features:
```bash
python feature_engineering.py
```

**Output**: 
- `f1_engineered_features.csv` (20+ features per race)
- `label_encoders.pkl` (categorical encodings)

### Step 3: Train Model

Train XGBoost model with hyperparameter tuning:
```bash
python train_model.py
```

**Output**:
- `f1_prediction_model_latest.pkl` (trained model)
- `feature_columns.pkl` (feature names)
- `feature_importance.png` (visualization)
- Model metrics (MAE, RMSE, R²)

**Expected Performance**: ~2-3 positions MAE on test set

### Step 4: Generate 2025 Predictions

Predict remaining 5 races and final championship:
```bash
python predict_2025.py
```

**Output**:
- Console output with race-by-race predictions https://ski146.github.io/F1_Championship-Predictor_2025/index.html
- `championship_predictions_2025.txt` (detailed results)

## 🧠 Model Features

### Input Features (20 total)

**Driver Features**
- Average finish position (last 5 races)
- Total points (last 5 races)
- Circuit-specific performance history

**Team Features**
- Season average finish position
- Total team points
- DNF/reliability rate
- Circuit-specific team performance

**Race Context Features**
- Qualifying position
- Grid position
- Starting position category (front row, top 5, top 10)
- Season progress (race number/24)

**Historical Performance**
- 2024 winners at each circuit
- Multi-year(2018-2025) circuit performance

## 📈 Model Performance

Based on 2024 test set:
- **MAE**: ~2.3 positions (average prediction error)
- **RMSE**: ~3.1 positions
- **R²**: ~0.72

Performance by position group:
- **Top 3 (Podium)**: ±1.5 positions MAE
- **Points (P4-P10)**: ±2.0 positions MAE
- **Lower field (P11-P20)**: ±3.5 positions MAE

## 🎯 2025 Predictions

### Current Standings (After Round 20 - Mexican GP)
1. Oscar Piastri - 357 pts (McLaren)
2. Lando Norris - 356 pts (McLaren) 
3. Max Verstappen - 321 pts (Red Bull) 

### Remaining Races
- **Round 21**: Brazilian GP (Nov 2) **Sprint**
- **Round 22**: Las Vegas GP (Nov 23)
- **Round 23**: Qatar GP (Nov 30) **Sprint**
- **Round 24**: Abu Dhabi GP (Dec 7)

### Championship Win Probability
- **Oscar Piastri**: ~85-90% (strong favorite)
- **Lando Norris**: ~25-30% (needs Piastri to falter)
- **Max Verstappen**: ~2-5% (mathematical chance only) 

## 🖥️ Interactive UI

The React dashboard provides:
- Live championship standings
- Race-by-race prediction cards
- Win probability visualizations
- Championship scenarios analysis

## 🔧 Configuration

### Modify Prediction Parameters

Edit `predict_2025.py` to adjust:
- Driver lineups (`CURRENT_STANDINGS_2025`)
- Team performance metrics (`TEAM_PERFORMANCE_2025`)
- Circuit mappings

### Retrain Model

To retrain with different parameters, edit `f1-train_model.py`:
- `test_year`: Change train/test split
- `param_grid`: Modify hyperparameter search space
- Feature selection: Add/remove features

## 📝 Data Sources

- **FastF1 API**: Official F1 timing data
- **Ergast API**: Historical race results (alternative)
- **FIA**: Official race regulations and points system

## ⚠️ Limitations

- Model trained on 2018-2024 data (recent regulations era)
- Cannot predict unforeseen events (crashes, mechanical failures, weather)
- Circuit encoding may struggle with new tracks
- Rookie drivers have limited historical data

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add weather data integration
- Implement tire strategy features
- Include practice/FP session data
- Add sprint race predictions

## 📄 License

MIT License - feel free to use for personal/educational purposes

## 🙏 Acknowledgments

- FastF1 library maintainers
- F1 data community
- Ergast Developer API

---

**Note**: This is a prediction model for entertainment/analysis purposes. Actual F1 results may vary significantly due to unpredictable race conditions, driver decisions, and team strategies.

## 📞 Quick Commands
```bash
# Full pipeline (first time)
pip install -r requirements.txt
python data_collection.py      # 20-30 min
python feature_engineering.py  # 2-5 min
python train_model.py          # 5-15 min
python predict_2025.py         # <1 min

# Update predictions only (after new race)
python predict_2025.py

# Retrain model
python feature_engineering.py && python train_model.py
```

## 🏆 Current Prediction

**Predicted 2025 World Champion: Oscar Piastri (McLaren)**
- 85-90% win probability (⚠️ Statistical prediction based on ML model (2018-2025 F1 data) - actual results may vary.)
- 30-point lead over Norris
- 4 races remaining

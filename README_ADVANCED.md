# 🏎️ Advanced F1 Race Prediction System

A sophisticated Formula 1 prediction system that combines **2025 Season Race Prediction** and **Complete Race Prediction** using advanced machine learning techniques.

## 🚀 Features

### 1. 🏆 2025 Season Race Prediction
- **Team Switch Analysis**: Predict driver performance when switching teams
- **Multi-Team Comparison**: Compare driver performance across different teams
- **Impact Analysis**: Analyze the impact of team switches on driver performance
- **Historical Data Integration**: Uses comprehensive historical F1 data

### 2. 🏁 Complete Race Prediction
- **Qualifying + Race Prediction**: Predict both qualifying and race positions
- **All 2024 Drivers**: Predict for all drivers who raced in 2024
- **Any Circuit**: Predict for any F1 circuit in the database
- **Deep Learning Models**: Uses advanced ML models with GridSearchCV
- **Comprehensive Features**: 50+ engineered features per prediction

### 3. 📊 Advanced Analytics
- **Model Performance Analysis**: Compare multiple ML algorithms
- **Feature Importance**: Understand what drives predictions
- **Grid Search CV**: Optimized hyperparameters for best performance
- **Multiple Algorithms**: Random Forest, XGBoost, LightGBM, Neural Networks

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd F1_datasets
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Ensure you have the F1 datasets**:
- `results.csv`
- `drivers.csv`
- `constructors.csv`
- `races.csv`
- `qualifying.csv`
- `lap_times.csv`

## 🎯 Usage

### Interactive Mode (Recommended)
```bash
python interactive_race_predictor.py
```

This launches a comprehensive interactive menu system with:
- 🏆 2025 Season Race Prediction
- 🏁 Complete Race Prediction
- 📊 Model Performance Analysis
- 🔍 Data Exploration
- 🎯 Quick Predictions

### Direct Script Usage

#### 1. 2025 Season Prediction
```python
from f1_team_switch_predictor import F1TeamSwitchPredictor

predictor = F1TeamSwitchPredictor()
predictor.load_data()

# Predict driver performance with team switch
prediction = predictor.predict_team_switch_performance(driver_id, team_id)
```

#### 2. Complete Race Prediction
```python
from advanced_race_predictor import AdvancedRacePredictor

predictor = AdvancedRacePredictor()
predictor.load_data()

# Train models (required for predictions)
predictor.train_qualifying_model()
predictor.train_race_model()

# Predict complete race
race_prediction = predictor.predict_complete_race(circuit_id)
```

## 🏗️ System Architecture

### Advanced Race Predictor
```
┌─────────────────────────────────────────────────────────────┐
│                    AdvancedRacePredictor                    │
├─────────────────────────────────────────────────────────────┤
│  Data Loading & Processing                                  │
│  ├── results_df, drivers_df, constructors_df               │
│  ├── races_df, qualifying_df, lap_times_df                 │
│  └── 2024 drivers identification                           │
├─────────────────────────────────────────────────────────────┤
│  Feature Engineering                                        │
│  ├── Driver Features (20+ metrics)                         │
│  ├── Team Features (20+ metrics)                           │
│  ├── Circuit Features (5+ metrics)                         │
│  └── Interaction Features (10+ metrics)                    │
├─────────────────────────────────────────────────────────────┤
│  Model Training                                             │
│  ├── Qualifying Model (GridSearchCV)                       │
│  ├── Race Model (GridSearchCV)                             │
│  ├── Multiple Algorithms                                   │
│  └── Performance Comparison                                │
├─────────────────────────────────────────────────────────────┤
│  Prediction Engine                                          │
│  ├── Complete Race Prediction                              │
│  ├── Qualifying + Race Positions                           │
│  └── Points Calculation                                    │
└─────────────────────────────────────────────────────────────┘
```

### Feature Engineering Pipeline
```
Driver Performance Metrics:
├── Basic Metrics (avg_position, avg_points, finish_rate)
├── Advanced Metrics (position_std, podium_rate, top5_rate)
├── Trend Analysis (position_trend, points_trend)
├── Qualifying Performance (avg_qualifying, pole_rate)
├── Lap Time Analysis (consistency, fastest_lap_rate)
└── Experience Metrics (driver_experience, years_active)

Team Performance Metrics:
├── Team Performance (team_avg_position, team_podium_rate)
├── Team Consistency (team_position_std, team_points_std)
├── Team Trends (team_position_trend, team_points_trend)
├── Team Qualifying (team_avg_qualifying, team_pole_rate)
└── Team Experience (team_experience, team_years_active)

Interaction Features:
├── Driver-Team Synergy (experience_ratio, performance_gap)
├── Qualifying-Race Sync (qualifying_sync, consistency_sync)
└── Circuit-Specific (driver_circuit_exp, team_circuit_exp)
```

## 🤖 Machine Learning Models

### Algorithms Used
1. **Random Forest Regressor**
   - Hyperparameters: n_estimators, max_depth, min_samples_split
   - Advantages: Robust, handles non-linear relationships

2. **Gradient Boosting Regressor**
   - Hyperparameters: n_estimators, learning_rate, max_depth
   - Advantages: High accuracy, handles complex patterns

3. **XGBoost Regressor**
   - Hyperparameters: n_estimators, max_depth, learning_rate, subsample
   - Advantages: Fast, handles missing values, regularization

4. **LightGBM Regressor**
   - Hyperparameters: n_estimators, max_depth, learning_rate, num_leaves
   - Advantages: Memory efficient, fast training

5. **Neural Network (MLPRegressor)**
   - Hyperparameters: hidden_layer_sizes, activation, alpha
   - Advantages: Captures complex non-linear relationships

### Model Selection Process
1. **GridSearchCV**: 5-fold cross-validation
2. **Hyperparameter Optimization**: Exhaustive search
3. **Performance Metrics**: R², MAE, RMSE
4. **Best Model Selection**: Highest R² score
5. **Feature Importance**: For tree-based models

## 📊 Performance Metrics

### Model Evaluation
- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **Cross-Validation**: 5-fold CV for robust evaluation

### Feature Importance
- **Tree-based Models**: Feature importance scores
- **Top 15 Features**: Most influential features
- **Feature Categories**: Driver, Team, Circuit, Interaction

## 🎮 Interactive Features

### Main Menu Options
1. **🏆 2025 Season Race Prediction**
   - Predict team switch performance
   - Compare across multiple teams
   - Analyze team switch impact
   - View available data

2. **🏁 Complete Race Prediction**
   - Train advanced models
   - Predict for any circuit
   - Popular circuits prediction
   - View available circuits

3. **📊 Model Performance Analysis**
   - View model metrics
   - Compare algorithms
   - Feature importance
   - Best model identification

4. **🔍 Data Exploration**
   - Dataset overview
   - 2024 drivers list
   - Available circuits
   - Historical data exploration

5. **🎯 Quick Predictions**
   - Popular driver comparisons
   - Famous circuit predictions
   - Pre-configured scenarios

### Quick Prediction Scenarios
- Max Verstappen: Red Bull vs Mercedes
- Lewis Hamilton: Mercedes vs Ferrari
- Charles Leclerc: Ferrari vs Red Bull
- Lando Norris: McLaren vs Red Bull
- Monaco Grand Prix prediction
- British Grand Prix prediction

## 📈 Example Outputs

### Complete Race Prediction
```
🏁 MONACO GRAND PRIX PREDICTION
================================================================================
Pos Driver               Team           Qual Race Points
--------------------------------------------------------------------------------
1   Max Verstappen       Red Bull       1     1     25
2   Charles Leclerc      Ferrari        2     2     18
3   Lewis Hamilton       Mercedes       3     3     15
4   Lando Norris         McLaren        4     4     12
5   Carlos Sainz         Ferrari        5     5     10
...
```

### Team Switch Analysis
```
🏆 MAX VERSTAPPEN - RED BULL vs MERCEDES
--------------------------------------------------
📊 CURRENT PERFORMANCE:
Average position: 1.2
Average points: 22.1
Podium rate: 95.2%

🏎️  RED BULL PREDICTION:
Average position: 1.1
Average points: 23.5
Podium rate: 97.8%

🏎️  MERCEDES PREDICTION:
Average position: 2.8
Average points: 15.2
Podium rate: 65.4%
```

## 🔧 Technical Details

### Data Sources
- **Results**: Race results and positions
- **Drivers**: Driver information and statistics
- **Constructors**: Team information and performance
- **Races**: Circuit and race information
- **Qualifying**: Qualifying session results
- **Lap Times**: Detailed lap time data (sample)

### Data Processing
- **Feature Engineering**: 50+ comprehensive features
- **Data Cleaning**: Missing value handling
- **Scaling**: RobustScaler for numerical features
- **Validation**: Train-test split with stratification

### Model Training
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Hyperparameter Tuning**: GridSearchCV optimization
- **Model Comparison**: Multiple algorithms tested
- **Performance Tracking**: Comprehensive metrics

## 🚀 Performance Optimization

### Training Optimization
- **Parallel Processing**: Multi-core GridSearchCV
- **Memory Management**: Efficient data handling
- **Early Stopping**: For gradient boosting models
- **Feature Selection**: Automatic feature importance

### Prediction Optimization
- **Caching**: Pre-computed features
- **Batch Processing**: Efficient predictions
- **Memory Efficient**: Optimized data structures

## 🔮 Future Enhancements

### Planned Features
1. **Real-time Predictions**: Live race predictions
2. **Weather Integration**: Weather impact analysis
3. **Tire Strategy**: Tire wear and strategy prediction
4. **Driver Form**: Recent form and momentum
5. **Team Development**: Car development trajectory
6. **Circuit Characteristics**: Track-specific features
7. **Seasonal Trends**: Year-over-year performance
8. **Interactive Visualizations**: Charts and graphs

### Model Improvements
1. **Ensemble Methods**: Stacking multiple models
2. **Deep Learning**: More sophisticated neural networks
3. **Time Series**: Sequential prediction models
4. **Bayesian Optimization**: Advanced hyperparameter tuning
5. **Feature Selection**: Automated feature engineering

## 📝 Usage Examples

### Example 1: Complete Race Prediction
```python
# Initialize predictor
predictor = AdvancedRacePredictor()
predictor.load_data()

# Train models
predictor.train_qualifying_model()
predictor.train_race_model()

# Predict for Monaco
monaco_circuit = predictor.races_df[
    predictor.races_df['name'].str.contains('Monaco')
]['circuitId'].iloc[0]

prediction = predictor.predict_complete_race(monaco_circuit)
```

### Example 2: Team Switch Analysis
```python
# Initialize predictor
predictor = F1TeamSwitchPredictor()
predictor.load_data()

# Get driver and team IDs
driver_id = 1  # Max Verstappen
team_id = 1    # Red Bull

# Predict performance
prediction = predictor.predict_team_switch_performance(driver_id, team_id)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Formula 1 for the historical data
- Scikit-learn for the machine learning framework
- XGBoost and LightGBM for advanced algorithms
- The F1 community for inspiration and feedback

---

**🏎️ Ready to predict the future of Formula 1! 🏎️** 
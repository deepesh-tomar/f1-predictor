#!/usr/bin/env python3
"""
Advanced F1 Race Predictor with Deep Learning Models
Predicts qualifying and race positions for all 2024 drivers at any circuit for 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
import joblib
import os

# Suppress LightGBM warnings
import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)

class AdvancedRacePredictor:
    def __init__(self, force_retrain=False):
        self.results_df = None
        self.drivers_df = None
        self.constructors_df = None
        self.races_df = None
        self.qualifying_df = None
        self.lap_times_df = None
        
        # Models
        self.qualifying_model = None
        self.race_model = None
        self.qualifying_scaler = RobustScaler()
        self.race_scaler = RobustScaler()
        
        # Feature columns
        self.qualifying_features = None
        self.race_features = None
        
        # Model performance
        self.qualifying_performance = {}
        self.race_performance = {}
        
        # 2024 drivers
        self.drivers_2024 = None
        
        # Model file paths
        self.qualifying_model_path = "qualifying_model.joblib"
        self.race_model_path = "race_model.joblib"
        self.qualifying_scaler_path = "qualifying_scaler.joblib"
        self.race_scaler_path = "race_scaler.joblib"
        self.qualifying_features_path = "qualifying_features.joblib"
        self.race_features_path = "race_features.joblib"
        
        self.force_retrain = force_retrain
        
    def load_data(self):
        """Load all relevant F1 datasets and models if available"""
        print("Loading F1 datasets...")
        self.results_df = pd.read_csv('results.csv')
        self.drivers_df = pd.read_csv('drivers.csv')
        self.constructors_df = pd.read_csv('constructors.csv')
        self.races_df = pd.read_csv('races.csv')
        self.qualifying_df = pd.read_csv('qualifying.csv')
        self.lap_times_df = pd.read_csv('lap_times.csv', nrows=100000)
        print("Data loaded successfully!")
        self.get_2024_drivers()
        self.load_or_train_models()

    def load_or_train_models(self):
        """Load models from disk if available, else train and save them."""
        all_exist = all([
            os.path.exists(self.qualifying_model_path),
            os.path.exists(self.race_model_path),
            os.path.exists(self.qualifying_scaler_path),
            os.path.exists(self.race_scaler_path),
            os.path.exists(self.qualifying_features_path),
            os.path.exists(self.race_features_path)
        ])
        if all_exist and not self.force_retrain:
            print("\n🔄 Loading pre-trained models for instant predictions...")
            self.qualifying_model = joblib.load(self.qualifying_model_path)
            self.race_model = joblib.load(self.race_model_path)
            self.qualifying_scaler = joblib.load(self.qualifying_scaler_path)
            self.race_scaler = joblib.load(self.race_scaler_path)
            self.qualifying_features = joblib.load(self.qualifying_features_path)
            self.race_features = joblib.load(self.race_features_path)
            print("✅ Models loaded successfully!")
        else:
            print("\n⚡ Training models (this may take a while)...")
            self.train_qualifying_model()
            self.train_race_model()
            print("\n💾 Saving trained models for future fast startup...")
            joblib.dump(self.qualifying_model, self.qualifying_model_path)
            joblib.dump(self.race_model, self.race_model_path)
            joblib.dump(self.qualifying_scaler, self.qualifying_scaler_path)
            joblib.dump(self.race_scaler, self.race_scaler_path)
            joblib.dump(self.qualifying_features, self.qualifying_features_path)
            joblib.dump(self.race_features, self.race_features_path)
            print("✅ Models saved!")
        
    def get_2024_drivers(self):
        """Get all drivers who raced in 2024"""
        print("Identifying 2024 drivers...")
        
        # Get 2024 races
        races_2024 = self.races_df[self.races_df['year'] == 2024]
        
        # Get all results from 2024
        results_2024 = self.results_df[self.results_df['raceId'].isin(races_2024['raceId'])]
        
        # Get unique drivers from 2024
        drivers_2024_ids = results_2024['driverId'].unique()
        
        # Get driver details
        self.drivers_2024 = self.drivers_df[self.drivers_df['driverId'].isin(drivers_2024_ids)].copy()
        self.drivers_2024 = self.drivers_2024.sort_values('surname')
        
        print(f"Found {len(self.drivers_2024)} drivers who raced in 2024")
        
    def create_comprehensive_driver_features(self, driver_id, races_back=20):
        """Create comprehensive features for a specific driver"""
        driver_results = self.results_df[self.results_df['driverId'] == driver_id].copy()
        
        if driver_results.empty:
            return None
            
        # Merge with races
        driver_results = driver_results.merge(self.races_df[['raceId', 'year', 'circuitId', 'name']], 
                                            on='raceId', how='left')
        
        # Sort chronologically
        driver_results = driver_results.sort_values(['year', 'raceId'])
        
        features = {}
        
        # Recent performance metrics (last N races)
        recent_results = driver_results.tail(races_back)
        
        if not recent_results.empty:
            # Basic performance metrics
            features['avg_position'] = recent_results['positionOrder'].mean()
            features['avg_points'] = recent_results['points'].mean()
            features['finish_rate'] = (recent_results['positionOrder'].notna().sum() / len(recent_results))
            features['podium_rate'] = (recent_results['positionOrder'] <= 3).sum() / len(recent_results)
            features['points_finish_rate'] = (recent_results['points'] > 0).sum() / len(recent_results)
            
            # Advanced performance metrics
            features['position_std'] = recent_results['positionOrder'].std()
            features['points_std'] = recent_results['points'].std()
            features['best_position'] = recent_results['positionOrder'].min()
            features['worst_position'] = recent_results['positionOrder'].max()
            features['top5_rate'] = (recent_results['positionOrder'] <= 5).sum() / len(recent_results)
            features['top10_rate'] = (recent_results['positionOrder'] <= 10).sum() / len(recent_results)
            
            # Trend analysis
            if len(recent_results) >= 10:
                recent_10 = recent_results.tail(10)
                older_10 = recent_results.head(10)
                features['position_trend'] = older_10['positionOrder'].mean() - recent_10['positionOrder'].mean()
                features['points_trend'] = recent_10['points'].mean() - older_10['points'].mean()
            
            # Qualifying performance
            driver_qualifying = self.qualifying_df[self.qualifying_df['driverId'] == driver_id].tail(races_back)
            if not driver_qualifying.empty:
                features['avg_qualifying_position'] = driver_qualifying['position'].mean()
                features['qualifying_consistency'] = driver_qualifying['position'].std()
                features['best_qualifying'] = driver_qualifying['position'].min()
                features['qualifying_to_race_gap'] = features['avg_position'] - features['avg_qualifying_position']
                features['qualifying_pole_rate'] = (driver_qualifying['position'] == 1).sum() / len(driver_qualifying)
                features['qualifying_top3_rate'] = (driver_qualifying['position'] <= 3).sum() / len(driver_qualifying)
            
            # Lap time consistency
            driver_laps = self.lap_times_df[self.lap_times_df['driverId'] == driver_id].tail(3000)
            if not driver_laps.empty:
                features['avg_lap_time'] = driver_laps['milliseconds'].mean()
                features['lap_time_consistency'] = driver_laps['milliseconds'].std()
                features['fastest_lap_rate'] = (driver_laps['position'] == 1).sum() / len(driver_laps)
        
        # Driver experience and longevity
        features['driver_experience'] = len(driver_results)
        features['years_active'] = driver_results['year'].max() - driver_results['year'].min() + 1
        features['races_per_year'] = features['driver_experience'] / features['years_active']
        
        # Current season performance (2024)
        current_season = driver_results[driver_results['year'] == 2024]
        if not current_season.empty:
            features['current_season_avg_position'] = current_season['positionOrder'].mean()
            features['current_season_points'] = current_season['points'].sum()
            features['current_season_races'] = len(current_season)
            features['current_season_podiums'] = (current_season['positionOrder'] <= 3).sum()
        
        return features
    
    def create_comprehensive_team_features(self, constructor_id, races_back=20):
        """Create comprehensive features for a specific team"""
        team_results = self.results_df[self.results_df['constructorId'] == constructor_id].copy()
        
        if team_results.empty:
            return None
            
        # Merge with races
        team_results = team_results.merge(self.races_df[['raceId', 'year', 'circuitId', 'name']], 
                                        on='raceId', how='left')
        
        # Sort chronologically
        team_results = team_results.sort_values(['year', 'raceId'])
        
        features = {}
        
        # Recent team performance
        recent_team_results = team_results.tail(races_back * 2)  # More data for teams (2 drivers)
        
        if not recent_team_results.empty:
            # Basic team metrics
            features['team_avg_position'] = recent_team_results['positionOrder'].mean()
            features['team_avg_points'] = recent_team_results['points'].mean()
            features['team_finish_rate'] = (recent_team_results['positionOrder'].notna().sum() / len(recent_team_results))
            features['team_podium_rate'] = (recent_team_results['positionOrder'] <= 3).sum() / len(recent_team_results)
            features['team_points_finish_rate'] = (recent_team_results['points'] > 0).sum() / len(recent_team_results)
            
            # Advanced team metrics
            features['team_position_std'] = recent_team_results['positionOrder'].std()
            features['team_points_std'] = recent_team_results['points'].std()
            features['team_best_position'] = recent_team_results['positionOrder'].min()
            features['team_worst_position'] = recent_team_results['positionOrder'].max()
            features['team_top5_rate'] = (recent_team_results['positionOrder'] <= 5).sum() / len(recent_team_results)
            features['team_top10_rate'] = (recent_team_results['positionOrder'] <= 10).sum() / len(recent_team_results)
            
            # Team trend analysis
            if len(recent_team_results) >= 20:
                recent_20 = recent_team_results.tail(20)
                older_20 = recent_team_results.head(20)
                features['team_position_trend'] = older_20['positionOrder'].mean() - recent_20['positionOrder'].mean()
                features['team_points_trend'] = recent_20['points'].mean() - older_20['points'].mean()
            
            # Team qualifying performance
            team_qualifying = self.qualifying_df[self.qualifying_df['constructorId'] == constructor_id].tail(races_back * 2)
            if not team_qualifying.empty:
                features['team_avg_qualifying'] = team_qualifying['position'].mean()
                features['team_qualifying_consistency'] = team_qualifying['position'].std()
                features['team_best_qualifying'] = team_qualifying['position'].min()
                features['team_qualifying_to_race_gap'] = features['team_avg_position'] - features['team_avg_qualifying']
                features['team_qualifying_pole_rate'] = (team_qualifying['position'] == 1).sum() / len(team_qualifying)
                features['team_qualifying_top3_rate'] = (team_qualifying['position'] <= 3).sum() / len(team_qualifying)
        
        # Team experience and history
        features['team_experience'] = len(team_results)
        features['team_years_active'] = team_results['year'].max() - team_results['year'].min() + 1
        features['team_races_per_year'] = features['team_experience'] / features['team_years_active']
        
        # Current season team performance (2024)
        current_season = team_results[team_results['year'] == 2024]
        if not current_season.empty:
            features['team_current_season_avg_position'] = current_season['positionOrder'].mean()
            features['team_current_season_points'] = current_season['points'].sum()
            features['team_current_season_races'] = len(current_season)
            features['team_current_season_podiums'] = (current_season['positionOrder'] <= 3).sum()
        
        return features
    
    def create_circuit_features(self, circuit_id):
        """Create comprehensive features for a specific circuit"""
        circuit_races = self.races_df[self.races_df['circuitId'] == circuit_id]
        
        if circuit_races.empty:
            return None
            
        features = {}
        features['circuit_experience'] = len(circuit_races)
        features['circuit_years_active'] = circuit_races['year'].max() - circuit_races['year'].min() + 1
        features['circuit_frequency'] = features['circuit_experience'] / features['circuit_years_active']
        
        # Recent circuit performance (last 5 years)
        recent_year = self.races_df['year'].max()
        recent_circuit_races = circuit_races[circuit_races['year'] >= recent_year - 5]
        if not recent_circuit_races.empty:
            features['recent_circuit_races'] = len(recent_circuit_races)
        
        return features
    
    def create_combined_features(self, driver_id, constructor_id, circuit_id, races_back=20):
        """Combine driver, team, and circuit features with interaction terms"""
        driver_features = self.create_comprehensive_driver_features(driver_id, races_back)
        team_features = self.create_comprehensive_team_features(constructor_id, races_back)
        circuit_features = self.create_circuit_features(circuit_id)
        
        if driver_features is None or team_features is None:
            return None
            
        # Combine all features
        combined_features = {**driver_features, **team_features}
        
        if circuit_features:
            combined_features.update(circuit_features)
        
        # Add interaction features
        combined_features['driver_team_experience_ratio'] = driver_features['driver_experience'] / team_features['team_experience']
        combined_features['driver_team_performance_gap'] = driver_features['avg_position'] - team_features['team_avg_position']
        combined_features['driver_team_points_gap'] = driver_features['avg_points'] - team_features['team_avg_points']
        
        # Driver-team synergy features
        combined_features['driver_team_qualifying_sync'] = abs(driver_features.get('avg_qualifying_position', 0) - team_features.get('team_avg_qualifying', 0))
        combined_features['driver_team_consistency_sync'] = abs(driver_features.get('position_std', 0) - team_features.get('team_position_std', 0))
        
        # Circuit-specific features
        combined_features['driver_circuit_experience'] = driver_features.get('driver_experience', 0) * circuit_features.get('circuit_frequency', 0)
        combined_features['team_circuit_experience'] = team_features.get('team_experience', 0) * circuit_features.get('circuit_frequency', 0)
        
        return combined_features
    
    def prepare_qualifying_data(self):
        """Prepare training data for qualifying prediction"""
        print("Preparing qualifying training data...")
        
        # Get recent results (last 10 years for better relevance)
        recent_year = self.races_df['year'].max()
        recent_races = self.races_df[self.races_df['year'] >= recent_year - 10]
        
        training_data = []
        
        for _, race in recent_races.iterrows():
            race_qualifying = self.qualifying_df[self.qualifying_df['raceId'] == race['raceId']]
            
            for _, qualifying in race_qualifying.iterrows():
                features = self.create_combined_features(
                    qualifying['driverId'], 
                    qualifying['constructorId'], 
                    race['circuitId']
                )
                
                if features and pd.notna(qualifying['position']):
                    features['target'] = qualifying['position']
                    features['raceId'] = race['raceId']
                    features['driverId'] = qualifying['driverId']
                    features['constructorId'] = qualifying['constructorId']
                    training_data.append(features)
        
        df = pd.DataFrame(training_data)
        print(f"Qualifying training data prepared: {len(df)} samples")
        return df
    
    def prepare_race_data(self):
        """Prepare training data for race position prediction"""
        print("Preparing race training data...")
        
        # Get recent results (last 10 years for better relevance)
        recent_year = self.races_df['year'].max()
        recent_races = self.races_df[self.races_df['year'] >= recent_year - 10]
        
        training_data = []
        
        for _, race in recent_races.iterrows():
            race_results = self.results_df[self.results_df['raceId'] == race['raceId']]
            
            for _, result in race_results.iterrows():
                # Get qualifying position for this driver in this race
                qualifying_pos = self.qualifying_df[
                    (self.qualifying_df['raceId'] == race['raceId']) & 
                    (self.qualifying_df['driverId'] == result['driverId'])
                ]
                
                if not qualifying_pos.empty:
                    qualifying_position = qualifying_pos.iloc[0]['position']
                else:
                    qualifying_position = 20  # Default if no qualifying data
                
                features = self.create_combined_features(
                    result['driverId'], 
                    result['constructorId'], 
                    race['circuitId']
                )
                
                if features and pd.notna(result['positionOrder']):
                    # Add qualifying position as a feature for race prediction
                    features['qualifying_position'] = qualifying_position
                    features['target'] = result['positionOrder']
                    features['raceId'] = race['raceId']
                    features['driverId'] = result['driverId']
                    features['constructorId'] = result['constructorId']
                    training_data.append(features)
        
        df = pd.DataFrame(training_data)
        print(f"Race training data prepared: {len(df)} samples")
        return df
    
    def train_qualifying_model(self):
        """Train sophisticated qualifying prediction model with GridSearchCV"""
        print("Training qualifying prediction model with GridSearchCV...")
        
        # Prepare training data
        df = self.prepare_qualifying_data()
        
        if df.empty:
            print("No qualifying training data available!")
            return
        
        # Remove rows with missing values
        df = df.dropna()
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col not in ['target', 'raceId', 'driverId', 'constructorId']]
        X = df[feature_columns]
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.qualifying_scaler.fit_transform(X_train)
        X_test_scaled = self.qualifying_scaler.transform(X_test)
        
        # Define models for GridSearchCV
        models = {
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7, 9],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'num_leaves': [31, 63, 127]
                }
            },
            'Neural Network': {
                'model': MLPRegressor(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50, 25), (100, 50), (100, 50, 25), (200, 100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate_init': [0.001, 0.01]
                }
            }
        }
        
        best_model = None
        best_score = -np.inf
        best_model_name = None
        
        # Train models with GridSearchCV
        for name, model_info in models.items():
            print(f"Training {name} with GridSearchCV...")
            
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # Evaluate on test set
            y_pred = grid_search.predict(X_test_scaled)
            score = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.qualifying_performance[name] = {
                'r2': score,
                'mae': mae,
                'rmse': rmse,
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_
            }
            
            print(f"{name}: R² = {score:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}")
            print(f"Best params: {grid_search.best_params_}")
            
            if score > best_score:
                best_score = score
                best_model = grid_search.best_estimator_
                best_model_name = name
        
        self.qualifying_model = best_model
        self.qualifying_features = feature_columns
        
        print(f"\nBest qualifying model: {best_model_name} with R² = {best_score:.3f}")
        
        # Feature importance (for tree-based models)
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 15 most important features for qualifying:")
            print(feature_importance.head(15))
    
    def train_race_model(self):
        """Train sophisticated race position prediction model with GridSearchCV"""
        print("Training race position prediction model with GridSearchCV...")
        
        # Prepare training data
        df = self.prepare_race_data()
        
        if df.empty:
            print("No race training data available!")
            return
        
        # Remove rows with missing values
        df = df.dropna()
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col not in ['target', 'raceId', 'driverId', 'constructorId']]
        X = df[feature_columns]
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.race_scaler.fit_transform(X_train)
        X_test_scaled = self.race_scaler.transform(X_test)
        
        # Define models for GridSearchCV
        models = {
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7, 9],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'num_leaves': [31, 63, 127]
                }
            },
            'Neural Network': {
                'model': MLPRegressor(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50, 25), (100, 50), (100, 50, 25), (200, 100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate_init': [0.001, 0.01]
                }
            }
        }
        
        best_model = None
        best_score = -np.inf
        best_model_name = None
        
        # Train models with GridSearchCV
        for name, model_info in models.items():
            print(f"Training {name} with GridSearchCV...")
            
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # Evaluate on test set
            y_pred = grid_search.predict(X_test_scaled)
            score = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.race_performance[name] = {
                'r2': score,
                'mae': mae,
                'rmse': rmse,
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_
            }
            
            print(f"{name}: R² = {score:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}")
            print(f"Best params: {grid_search.best_params_}")
            
            if score > best_score:
                best_score = score
                best_model = grid_search.best_estimator_
                best_model_name = name
        
        self.race_model = best_model
        self.race_features = feature_columns
        
        print(f"\nBest race model: {best_model_name} with R² = {best_score:.3f}")
        
        # Feature importance (for tree-based models)
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 15 most important features for race position:")
            print(feature_importance.head(15))
    
    def predict_driver_performance(self, driver_id, constructor_id, circuit_id, races_back=15):
        """Predict performance for a specific driver-team-circuit combination"""
        if self.qualifying_model is None or self.race_model is None:
            print("Models not trained yet. Please train the models first.")
            return None
        
        # Create features
        features = self.create_combined_features(driver_id, constructor_id, circuit_id, races_back)
        
        if features is None:
            print(f"Insufficient data for driver {driver_id}, team {constructor_id}, circuit {circuit_id}")
            return None
        
        try:
            # Predict qualifying position
            qualifying_features = [features.get(col, 0) for col in self.qualifying_features]
            qualifying_features_scaled = self.qualifying_scaler.transform([qualifying_features])
            predicted_qualifying = self.qualifying_model.predict(qualifying_features_scaled)[0]
            predicted_qualifying = max(1, min(20, round(predicted_qualifying)))
            
            # Add qualifying position to features for race prediction
            features['qualifying_position'] = predicted_qualifying
            
            # Predict race position
            race_features = [features.get(col, 0) for col in self.race_features]
            race_features_scaled = self.race_scaler.transform([race_features])
            predicted_race = self.race_model.predict(race_features_scaled)[0]
            predicted_race = max(1, min(20, round(predicted_race)))
            
            # Calculate points
            points = self.position_to_points(predicted_race)
            
            # Calculate confidence based on model performance
            qualifying_confidence = self.qualifying_performance.get('XGBoost', {}).get('r2', 0.5)
            race_confidence = self.race_performance.get('XGBoost', {}).get('r2', 0.5)
            overall_confidence = (qualifying_confidence + race_confidence) / 2
            
            # Calculate uncertainty (simplified)
            uncertainty = 1 - overall_confidence
            
            # Create prediction intervals (simplified)
            position_std = features.get('position_std', 3.0)  # Use driver's position standard deviation
            prediction_intervals = {
                'position_lower': max(1, predicted_race - position_std),
                'position_upper': min(20, predicted_race + position_std),
                'points_lower': self.position_to_points(min(20, predicted_race + position_std)),
                'points_upper': self.position_to_points(max(1, predicted_race - position_std))
            }
            
            return {
                'predicted_position': predicted_race,
                'predicted_qualifying': predicted_qualifying,
                'predicted_points': points,
                'confidence': overall_confidence,
                'uncertainty': uncertainty,
                'prediction_intervals': prediction_intervals,
                'features_used': len(features)
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def predict_complete_race(self, circuit_id):
        """Predict qualifying and race positions for all 2024 drivers at a specific circuit, limited to top 20 for each, with unique positions."""
        if self.qualifying_model is None or self.race_model is None:
            print("Models not trained yet. Please train the models first.")
            return None
        
        print(f"Predicting complete race results for circuit {circuit_id}...")
        
        # Get circuit name
        circuit_name = self.races_df[self.races_df['circuitId'] == circuit_id]['name'].iloc[0]
        
        predictions = []
        
        # Predict for each 2024 driver
        for _, driver in self.drivers_2024.iterrows():
            driver_id = driver['driverId']
            driver_name = f"{driver['forename']} {driver['surname']}"
            
            # Get driver's current team (2024 team)
            driver_2024_results = self.results_df[
                (self.results_df['driverId'] == driver_id) & 
                (self.results_df['raceId'].isin(self.races_df[self.races_df['year'] == 2024]['raceId']))
            ]
            
            if not driver_2024_results.empty:
                # Use the most recent team
                constructor_id = driver_2024_results.iloc[-1]['constructorId']
                team_name = self.constructors_df[self.constructors_df['constructorId'] == constructor_id]['name'].iloc[0]
                
                # Create features
                features = self.create_combined_features(driver_id, constructor_id, circuit_id)
                
                if features:
                    # Predict qualifying position
                    qualifying_features = [features.get(col, 0) for col in self.qualifying_features]
                    qualifying_features_scaled = self.qualifying_scaler.transform([qualifying_features])
                    predicted_qualifying = self.qualifying_model.predict(qualifying_features_scaled)[0]
                    predicted_qualifying = max(1, min(20, round(predicted_qualifying)))
                    
                    # Add qualifying position to features for race prediction
                    features['qualifying_position'] = predicted_qualifying
                    
                    # Predict race position
                    race_features = [features.get(col, 0) for col in self.race_features]
                    race_features_scaled = self.race_scaler.transform([race_features])
                    predicted_race = self.race_model.predict(race_features_scaled)[0]
                    predicted_race = max(1, min(20, round(predicted_race)))
                    
                    # Calculate points
                    points = self.position_to_points(predicted_race)
                    
                    predictions.append({
                        'driver_id': driver_id,
                        'driver_name': driver_name,
                        'team_name': team_name,
                        'predicted_qualifying': predicted_qualifying,
                        'predicted_race': predicted_race,
                        'predicted_points': points
                    })
        # Sort and select top 20 for qualifying and race
        qual_sorted = sorted(predictions, key=lambda x: (x['predicted_qualifying'], -x['predicted_points'], x['driver_name']))[:20]
        race_sorted = sorted(predictions, key=lambda x: (x['predicted_race'], -x['predicted_points'], x['driver_name']))[:20]
        # Assign unique positions (1-20) for display
        for i, pred in enumerate(qual_sorted, 1):
            pred['qualifying_display'] = i
        for i, pred in enumerate(race_sorted, 1):
            pred['race_display'] = i
        # For backward compatibility, keep 'predictions' as top 20 by race
        return {
            'circuit_name': circuit_name,
            'circuit_id': circuit_id,
            'predictions': race_sorted,
            'qualifying_order': qual_sorted,
            'race_order': race_sorted
        }
    
    def position_to_points(self, position):
        """Convert position to points (current F1 points system)"""
        points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        position_int = int(round(position))
        if position_int < 1:
            position_int = 1
        elif position_int > 20:
            position_int = 20
        return points_system.get(position_int, 0)
    
    def get_available_circuits(self):
        """Get list of available circuits"""
        return self.races_df[['circuitId', 'name']].drop_duplicates().sort_values('name')
    
    def get_available_drivers(self):
        """Get list of available drivers"""
        return self.drivers_df[['driverId', 'forename', 'surname']].copy()
    
    def get_available_teams(self):
        """Get list of available teams/constructors"""
        return self.constructors_df[['constructorId', 'name']].copy()

# Example usage
if __name__ == "__main__":
    predictor = AdvancedRacePredictor()
    predictor.load_data()
    
    print("\nTraining qualifying model...")
    predictor.train_qualifying_model()
    
    print("\nTraining race model...")
    predictor.train_race_model()
    
    print("\n" + "="*60)
    print("ADVANCED RACE PREDICTOR")
    print("="*60)
    
    # Example: Predict complete race for Monaco
    monaco_circuit = predictor.races_df[predictor.races_df['name'].str.contains('Monaco', case=False)]['circuitId'].iloc[0]
    
    print(f"\nPredicting complete race for Monaco (Circuit ID: {monaco_circuit})...")
    race_prediction = predictor.predict_complete_race(monaco_circuit)
    
    if race_prediction:
        print(f"\nComplete Race Prediction: {race_prediction['circuit_name']}")
        print("="*60)
        print(f"{'Pos':<3} {'Driver':<20} {'Team':<15} {'Qual':<4} {'Race':<4} {'Points':<6}")
        print("-" * 60)
        
        for i, pred in enumerate(race_prediction['predictions'], 1):
            print(f"{i:<3} {pred['driver_name']:<20} {pred['team_name']:<15} {pred['predicted_qualifying']:<4} {pred['predicted_race']:<4} {pred['predicted_points']:<6}") 
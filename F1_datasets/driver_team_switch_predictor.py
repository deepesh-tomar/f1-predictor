import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DriverTeamSwitchPredictor:
    def __init__(self):
        self.results_df = None
        self.drivers_df = None
        self.constructors_df = None
        self.races_df = None
        self.qualifying_df = None
        self.lap_times_df = None
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.model_performance = {}
        self.prediction_intervals = {}
        
    def load_data(self):
        """Load all relevant F1 datasets"""
        print("Loading F1 datasets...")
        self.results_df = pd.read_csv('results.csv')
        self.drivers_df = pd.read_csv('drivers.csv')
        self.constructors_df = pd.read_csv('constructors.csv')
        self.races_df = pd.read_csv('races.csv')
        self.qualifying_df = pd.read_csv('qualifying.csv')
        
        # Load a sample of lap_times due to large size
        self.lap_times_df = pd.read_csv('lap_times.csv', nrows=100000)
        print("Data loaded successfully!")
        
    def create_driver_features(self, driver_id, races_back=15):
        """Create comprehensive features for a specific driver based on recent performance"""
        driver_results = self.results_df[self.results_df['driverId'] == driver_id].copy()
        
        if driver_results.empty:
            return None
            
        # Merge with races to get year and circuit info
        driver_results = driver_results.merge(self.races_df[['raceId', 'year', 'circuitId', 'name']], 
                                            on='raceId', how='left')
        
        # Sort by year and raceId to get chronological order
        driver_results = driver_results.sort_values(['year', 'raceId'])
        
        # Calculate comprehensive rolling averages for recent performance
        features = {}
        
        # Recent performance metrics (last N races)
        recent_results = driver_results.tail(races_back)
        
        if not recent_results.empty:
            # Basic performance metrics
            features['avg_position'] = recent_results['positionOrder'].mean()
            features['avg_points'] = recent_results['points'].mean()
            features['avg_laps'] = recent_results['laps'].mean()
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
            if len(recent_results) >= 5:
                recent_5 = recent_results.tail(5)
                older_5 = recent_results.head(5)
                features['position_trend'] = older_5['positionOrder'].mean() - recent_5['positionOrder'].mean()
                features['points_trend'] = recent_5['points'].mean() - older_5['points'].mean()
            
            # Qualifying performance
            driver_qualifying = self.qualifying_df[self.qualifying_df['driverId'] == driver_id].tail(races_back)
            if not driver_qualifying.empty:
                features['avg_qualifying_position'] = driver_qualifying['position'].mean()
                features['qualifying_consistency'] = driver_qualifying['position'].std()
                features['best_qualifying'] = driver_qualifying['position'].min()
                features['qualifying_to_race_gap'] = features['avg_position'] - features['avg_qualifying_position']
            
            # Lap time consistency
            driver_laps = self.lap_times_df[self.lap_times_df['driverId'] == driver_id].tail(2000)
            if not driver_laps.empty:
                features['avg_lap_time'] = driver_laps['milliseconds'].mean()
                features['lap_time_consistency'] = driver_laps['milliseconds'].std()
                features['fastest_lap_rate'] = (driver_laps['position'] == 1).sum() / len(driver_laps)
        
        # Driver experience and longevity
        features['driver_experience'] = len(driver_results)
        features['years_active'] = driver_results['year'].max() - driver_results['year'].min() + 1
        features['races_per_year'] = features['driver_experience'] / features['years_active']
        
        # Season performance
        current_year = driver_results['year'].max()
        current_season = driver_results[driver_results['year'] == current_year]
        if not current_season.empty:
            features['current_season_avg_position'] = current_season['positionOrder'].mean()
            features['current_season_points'] = current_season['points'].sum()
            features['current_season_races'] = len(current_season)
        
        return features
    
    def create_team_features(self, constructor_id, races_back=15):
        """Create comprehensive features for a specific team based on recent performance"""
        team_results = self.results_df[self.results_df['constructorId'] == constructor_id].copy()
        
        if team_results.empty:
            return None
            
        # Merge with races
        team_results = team_results.merge(self.races_df[['raceId', 'year', 'circuitId', 'name']], 
                                        on='raceId', how='left')
        
        # Sort chronologically
        team_results = team_results.sort_values(['year', 'raceId'])
        
        # Recent team performance
        recent_team_results = team_results.tail(races_back * 2)  # More data for teams (2 drivers)
        
        features = {}
        
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
            if len(recent_team_results) >= 10:
                recent_10 = recent_team_results.tail(10)
                older_10 = recent_team_results.head(10)
                features['team_position_trend'] = older_10['positionOrder'].mean() - recent_10['positionOrder'].mean()
                features['team_points_trend'] = recent_10['points'].mean() - older_10['points'].mean()
            
            # Team qualifying performance
            team_qualifying = self.qualifying_df[self.qualifying_df['constructorId'] == constructor_id].tail(races_back * 2)
            if not team_qualifying.empty:
                features['team_avg_qualifying'] = team_qualifying['position'].mean()
                features['team_qualifying_consistency'] = team_qualifying['position'].std()
                features['team_best_qualifying'] = team_qualifying['position'].min()
                features['team_qualifying_to_race_gap'] = features['team_avg_position'] - features['team_avg_qualifying']
        
        # Team experience and history
        features['team_experience'] = len(team_results)
        features['team_years_active'] = team_results['year'].max() - team_results['year'].min() + 1
        features['team_races_per_year'] = features['team_experience'] / features['team_years_active']
        
        # Current season team performance
        current_year = team_results['year'].max()
        current_season = team_results[team_results['year'] == current_year]
        if not current_season.empty:
            features['team_current_season_avg_position'] = current_season['positionOrder'].mean()
            features['team_current_season_points'] = current_season['points'].sum()
            features['team_current_season_races'] = len(current_season)
        
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
    
    def create_combined_features(self, driver_id, constructor_id, circuit_id, races_back=15):
        """Combine driver, team, and circuit features with interaction terms"""
        driver_features = self.create_driver_features(driver_id, races_back)
        team_features = self.create_team_features(constructor_id, races_back)
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
        
        return combined_features
    
    def prepare_training_data(self, target_variable='positionOrder'):
        """Prepare comprehensive training data for the model"""
        print("Preparing training data...")
        
        # Get recent results (last 8 years for better relevance)
        recent_year = self.races_df['year'].max()
        recent_races = self.races_df[self.races_df['year'] >= recent_year - 8]
        
        training_data = []
        
        for _, race in recent_races.iterrows():
            race_results = self.results_df[self.results_df['raceId'] == race['raceId']]
            
            for _, result in race_results.iterrows():
                # Only include valid F1 grid positions (1-20)
                if pd.notna(result[target_variable]) and 1 <= result[target_variable] <= 20:
                    features = self.create_combined_features(
                        result['driverId'], 
                        result['constructorId'], 
                        race['circuitId']
                    )
                    
                    if features:
                        features['target'] = result[target_variable]
                        features['raceId'] = race['raceId']
                        features['driverId'] = result['driverId']
                        features['constructorId'] = result['constructorId']
                        training_data.append(features)
        
        df = pd.DataFrame(training_data)
        print(f"Training data prepared: {len(df)} samples (positions 1-20 only)")
        return df
    
    def train_model(self, target_variable='positionOrder'):
        """Train sophisticated ensemble model with hyperparameter tuning"""
        print("Training sophisticated ensemble model...")
        
        # Prepare training data
        df = self.prepare_training_data(target_variable)
        
        if df.empty:
            print("No training data available!")
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
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define sophisticated models with hyperparameter tuning
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=5,
                random_state=42
            ),
            'SVR': SVR(
                kernel='rbf',
                C=10,
                gamma='scale'
            ),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            ),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1)
        }
        
        # Train individual models and store performance
        trained_models = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.model_performance[name] = {
                'r2': score,
                'mae': mae,
                'rmse': rmse,
                'model': model
            }
            
            print(f"{name}: R² = {score:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}")
            trained_models[name] = model
        
        # Create ensemble model
        best_models = ['Random Forest', 'Gradient Boosting', 'Neural Network']
        ensemble_models = [(name, trained_models[name]) for name in best_models]
        
        self.model = VotingRegressor(estimators=ensemble_models)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate ensemble
        y_pred_ensemble = self.model.predict(X_test_scaled)
        ensemble_score = r2_score(y_test, y_pred_ensemble)
        ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
        
        print(f"\nEnsemble Model: R² = {ensemble_score:.3f}, MAE = {ensemble_mae:.3f}, RMSE = {ensemble_rmse:.3f}")
        
        self.feature_columns = feature_columns
        
        # Calculate prediction intervals
        self.calculate_prediction_intervals(X_test_scaled, y_test, y_pred_ensemble)
        
        # Feature importance (for Random Forest)
        rf_model = trained_models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 most important features:")
        print(feature_importance.head(15))
    
    def calculate_prediction_intervals(self, X_test, y_test, y_pred):
        """Calculate prediction intervals for uncertainty quantification"""
        residuals = y_test - y_pred
        residual_std = np.std(residuals)
        
        # Calculate confidence intervals
        self.prediction_intervals = {
            'residual_std': residual_std,
            'confidence_68': 1.0 * residual_std,  # 68% confidence interval
            'confidence_95': 1.96 * residual_std,  # 95% confidence interval
            'confidence_99': 2.58 * residual_std   # 99% confidence interval
        }
    
    def predict_driver_performance(self, driver_id, constructor_id, circuit_id, races_back=15):
        """Predict driver performance with realistic confidence intervals"""
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return None
        
        # Create features for the prediction scenario
        features = self.create_combined_features(driver_id, constructor_id, circuit_id, races_back)
        
        if features is None:
            print("Could not create features for the given scenario.")
            return None
        
        # Prepare feature vector
        feature_vector = [features.get(col, 0) for col in self.feature_columns]
        feature_vector_scaled = self.scaler.transform([feature_vector])
        
        # Make prediction
        predicted_position = self.model.predict(feature_vector_scaled)[0]
        
        # Ensure prediction is within valid F1 grid range (1-20)
        predicted_position = max(1, min(20, predicted_position))
        
        # Calculate prediction intervals
        confidence_68 = self.prediction_intervals['confidence_68']
        confidence_95 = self.prediction_intervals['confidence_95']
        
        lower_68 = max(1, predicted_position - confidence_68)
        upper_68 = min(20, predicted_position + confidence_68)
        lower_95 = max(1, predicted_position - confidence_95)
        upper_95 = min(20, predicted_position + confidence_95)
        
        # Calculate realistic confidence based on data quality and prediction uncertainty
        confidence_score = self.calculate_realistic_confidence(features, confidence_68)
        
        # Get driver and constructor names
        driver_name = self.drivers_df[self.drivers_df['driverId'] == driver_id]['forename'].iloc[0] + " " + \
                     self.drivers_df[self.drivers_df['driverId'] == driver_id]['surname'].iloc[0]
        constructor_name = self.constructors_df[self.constructors_df['constructorId'] == constructor_id]['name'].iloc[0]
        circuit_name = self.races_df[self.races_df['circuitId'] == circuit_id]['name'].iloc[0]
        
        return {
            'driver': driver_name,
            'team': constructor_name,
            'circuit': circuit_name,
            'predicted_position': round(predicted_position, 1),
            'predicted_points': self.position_to_points(predicted_position),
            'confidence': confidence_score,
            'prediction_intervals': {
                '68_percent': (round(lower_68, 1), round(upper_68, 1)),
                '95_percent': (round(lower_95, 1), round(upper_95, 1))
            },
            'uncertainty': round(confidence_68, 1)
        }
    
    def position_to_points(self, position):
        """Convert position to points (current F1 points system)"""
        points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        position_int = int(round(position))
        # Ensure position is within valid F1 grid range (1-20)
        if position_int < 1:
            position_int = 1
        elif position_int > 20:
            position_int = 20
        return points_system.get(position_int, 0)
    
    def calculate_realistic_confidence(self, features, uncertainty):
        """Calculate realistic confidence based on data quality and prediction uncertainty"""
        # Base confidence factors
        confidence_factors = []
        
        # Driver data quality
        driver_experience = features.get('driver_experience', 0)
        if driver_experience > 50:
            confidence_factors.append(0.9)
        elif driver_experience > 20:
            confidence_factors.append(0.7)
        elif driver_experience > 10:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.3)
        
        # Team data quality
        team_experience = features.get('team_experience', 0)
        if team_experience > 100:
            confidence_factors.append(0.9)
        elif team_experience > 50:
            confidence_factors.append(0.7)
        elif team_experience > 20:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.3)
        
        # Performance consistency
        driver_consistency = features.get('position_std', 10)
        if driver_consistency < 3:
            confidence_factors.append(0.9)
        elif driver_consistency < 5:
            confidence_factors.append(0.7)
        elif driver_consistency < 8:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.3)
        
        # Team consistency
        team_consistency = features.get('team_position_std', 10)
        if team_consistency < 3:
            confidence_factors.append(0.9)
        elif team_consistency < 5:
            confidence_factors.append(0.7)
        elif team_consistency < 8:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.3)
        
        # Uncertainty penalty
        uncertainty_penalty = max(0, 1 - (uncertainty / 5))  # Penalty based on prediction uncertainty
        
        # Calculate final confidence
        base_confidence = np.mean(confidence_factors)
        final_confidence = base_confidence * uncertainty_penalty
        
        return max(0.1, min(0.95, final_confidence))  # Ensure confidence is between 10% and 95%
    
    def analyze_team_switch_scenario(self, driver_id, old_constructor_id, new_constructor_id, circuit_id):
        """Analyze the impact of team switch on driver performance for 2025"""
        print("Analyzing team switch scenario...")
        
        # Get driver and team names
        driver_name = self.drivers_df[self.drivers_df['driverId'] == driver_id]['forename'].iloc[0] + " " + \
                     self.drivers_df[self.drivers_df['driverId'] == driver_id]['surname'].iloc[0]
        old_team = self.constructors_df[self.constructors_df['constructorId'] == old_constructor_id]['name'].iloc[0]
        new_team = self.constructors_df[self.constructors_df['constructorId'] == new_constructor_id]['name'].iloc[0]
        
        # Predict performance with old team (2024 scenario)
        old_prediction = self.predict_driver_performance(driver_id, old_constructor_id, circuit_id)
        
        # Predict performance with new team (2025 scenario)
        new_prediction = self.predict_driver_performance(driver_id, new_constructor_id, circuit_id)
        
        if old_prediction and new_prediction:
            print(f"\n=== Team Switch Analysis: {driver_name} ===")
            print(f"2024 Team: {old_team}")
            print(f"2025 Team: {new_team}")
            print(f"Circuit: {new_prediction['circuit']}")
            print(f"\nPredicted Performance:")
            print(f"  2024 (with {old_team}): Position {old_prediction['predicted_position']:.1f} ({old_prediction['predicted_points']} points)")
            print(f"     Confidence: {old_prediction['confidence']:.1%}")
            print(f"     Range (68%): {old_prediction['prediction_intervals']['68_percent'][0]}-{old_prediction['prediction_intervals']['68_percent'][1]}")
            print(f"  2025 (with {new_team}): Position {new_prediction['predicted_position']:.1f} ({new_prediction['predicted_points']} points)")
            print(f"     Confidence: {new_prediction['confidence']:.1%}")
            print(f"     Range (68%): {new_prediction['prediction_intervals']['68_percent'][0]}-{new_prediction['prediction_intervals']['68_percent'][1]}")
            
            position_change = old_prediction['predicted_position'] - new_prediction['predicted_position']
            points_change = new_prediction['predicted_points'] - old_prediction['predicted_points']
            
            print(f"\nImpact of Team Switch:")
            print(f"  Position Change: {position_change:+.1f} (positive = improvement)")
            print(f"  Points Change: {points_change:+.1f} points")
            
            if position_change > 0:
                print(f"  Prediction: IMPROVEMENT with {new_team}")
            elif position_change < 0:
                print(f"  Prediction: DECLINE with {new_team}")
            else:
                print(f"  Prediction: SIMILAR performance")
            
            print(f"  Overall Confidence: {min(old_prediction['confidence'], new_prediction['confidence']):.1%}")
    
    def get_available_drivers(self):
        """Get list of available drivers"""
        return self.drivers_df[['driverId', 'forename', 'surname', 'nationality']].sort_values('surname')
    
    def get_available_teams(self):
        """Get list of available teams"""
        return self.constructors_df[['constructorId', 'name', 'nationality']].sort_values('name')
    
    def get_available_circuits(self):
        """Get list of available circuits"""
        return self.races_df[['circuitId', 'name']].drop_duplicates().sort_values('name')

# Example usage and demonstration
if __name__ == "__main__":
    predictor = DriverTeamSwitchPredictor()
    predictor.load_data()
    predictor.train_model()
    
    print("\n" + "="*50)
    print("DRIVER TEAM SWITCH PREDICTOR")
    print("="*50)
    
    # Example: Predict Lewis Hamilton's performance if he switched to Ferrari
    # (Note: You'll need to find the correct IDs from the data)
    
    print("\nAvailable drivers:")
    print(predictor.get_available_drivers().head(10))
    
    print("\nAvailable teams:")
    print(predictor.get_available_teams().head(10))
    
    print("\nAvailable circuits:")
    print(predictor.get_available_circuits().head(10))
    
    # Example prediction (you can modify these IDs based on your data)
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    
    # Find some example IDs (you'll need to adjust these based on your data)
    try:
        # Example: Driver ID 1 (Lewis Hamilton) switching from constructor 1 to constructor 6 (Ferrari)
        # at circuit 1 (Australian GP)
        example_prediction = predictor.predict_driver_performance(1, 6, 1)
        if example_prediction:
            print(f"Driver: {example_prediction['driver']}")
            print(f"Team: {example_prediction['team']}")
            print(f"Circuit: {example_prediction['circuit']}")
            print(f"Predicted Position: {example_prediction['predicted_position']}")
            print(f"Predicted Points: {example_prediction['predicted_points']}")
            print(f"Confidence: {example_prediction['confidence']:.1%}")
            print(f"Prediction Range (68%): {example_prediction['prediction_intervals']['68_percent'][0]}-{example_prediction['prediction_intervals']['68_percent'][1]}")
    except Exception as e:
        print(f"Example prediction failed: {e}")
        print("Please check the available IDs in your dataset and adjust accordingly.") 
#!/usr/bin/env python3
"""
Interactive F1 Race Predictor
Combines 2025 Season Prediction and Complete Race Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from driver_team_switch_predictor import DriverTeamSwitchPredictor
from advanced_race_predictor import AdvancedRacePredictor
import warnings
warnings.filterwarnings('ignore')

class InteractiveRacePredictor:
    def __init__(self):
        self.season_predictor = DriverTeamSwitchPredictor()
        self.race_predictor = AdvancedRacePredictor(force_retrain=False)
        self.season_predictor.load_data()
        self.race_predictor.load_data()
        
    def display_menu(self):
        """Display the main menu"""
        print("\n" + "="*80)
        print("🏎️  F1 RACE PREDICTION SYSTEM 🏎️")
        print("="*80)
        print("1. 🏆 2025 Season Race Prediction")
        print("   - Predict driver performance when switching teams")
        print("   - Compare performance across different teams")
        print("   - Analyze team switch impact")
        print()
        print("2. 🏁 Complete Race Prediction")
        print("   - Predict qualifying and race positions for all 2024 drivers")
        print("   - Deep learning models with GridSearchCV")
        print("   - Comprehensive feature engineering")
        print()
        print("3. 📊 Model Performance Analysis")
        print("   - View model performance metrics")
        print("   - Feature importance analysis")
        print("   - Model comparison")
        print()
        print("4. 🔍 Data Exploration")
        print("   - Explore available drivers, teams, and circuits")
        print("   - View historical performance data")
        print()
        print("5. 🎯 Quick Predictions")
        print("   - Fast predictions for popular scenarios")
        print()
        print("0. 🚪 Exit")
        print("="*80)
        
    def season_prediction_menu(self):
        """Menu for 2025 season predictions"""
        while True:
            print("\n" + "="*60)
            print("🏆 2025 SEASON RACE PREDICTION")
            print("="*60)
            print("1. Predict driver performance with team switch")
            print("2. Compare driver across multiple teams")
            print("3. Analyze team switch impact")
            print("4. View available drivers and teams")
            print("5. Back to main menu")
            print("="*60)
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                self.predict_team_switch()
            elif choice == '2':
                self.compare_teams()
            elif choice == '3':
                self.analyze_team_switch()
            elif choice == '4':
                self.view_season_data()
            elif choice == '5':
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def complete_race_menu(self):
        """Menu for complete race predictions"""
        while True:
            print("\n" + "="*60)
            print("🏁 COMPLETE RACE PREDICTION")
            print("="*60)
            print("1. Load or train advanced models (required for predictions)")
            print("2. Predict complete race for any circuit")
            print("3. Predict for popular circuits")
            print("4. View available circuits")
            print("5. Force retrain models (overwrite saved models)")
            print("6. Back to main menu")
            print("="*60)
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                self.load_or_train_models()
            elif choice == '2':
                self.predict_complete_race()
            elif choice == '3':
                self.predict_popular_circuits()
            elif choice == '4':
                self.view_circuits()
            elif choice == '5':
                self.force_retrain_models()
            elif choice == '6':
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def predict_team_switch(self):
        """Predict driver performance with team switch"""
        print("\n🏆 PREDICT TEAM SWITCH PERFORMANCE")
        print("-" * 40)
        
        # Get available drivers
        drivers = self.season_predictor.get_available_drivers()
        print("\nAvailable drivers:")
        for i, driver in enumerate(drivers.head(20), 1):
            print(f"{i:2d}. {driver['name']}")
        
        try:
            driver_idx = int(input("\nSelect driver (enter number): ")) - 1
            if driver_idx < 0 or driver_idx >= len(drivers):
                print("❌ Invalid driver selection.")
                return
            driver = drivers.iloc[driver_idx]
        except ValueError:
            print("❌ Invalid input.")
            return
        
        # Get available teams
        teams = self.season_predictor.get_available_teams()
        print(f"\nAvailable teams for {driver['name']}:")
        for i, team in enumerate(teams, 1):
            print(f"{i:2d}. {team['name']}")
        
        try:
            team_idx = int(input("\nSelect team (enter number): ")) - 1
            if team_idx < 0 or team_idx >= len(teams):
                print("❌ Invalid team selection.")
                return
            team = teams.iloc[team_idx]
        except ValueError:
            print("❌ Invalid input.")
            return
        
        # Get current team
        current_team = self.season_predictor.get_driver_current_team(driver['driverId'])
        
        print(f"\n🎯 Predicting performance for {driver['name']}")
        print(f"Current team: {current_team['name']}")
        print(f"New team: {team['name']}")
        
        # Make prediction
        prediction = self.season_predictor.predict_team_switch_performance(
            driver['driverId'], team['constructorId']
        )
        
        if prediction:
            print(f"\n📊 PREDICTION RESULTS:")
            print(f"Predicted average position: {prediction['predicted_position']:.1f}")
            print(f"Predicted average points: {prediction['predicted_points']:.1f}")
            print(f"Predicted podium rate: {prediction['predicted_podium_rate']:.1%}")
            print(f"Predicted points finish rate: {prediction['predicted_points_finish_rate']:.1%}")
            
            # Compare with current performance
            current_perf = self.season_predictor.get_driver_current_performance(driver['driverId'])
            if current_perf:
                print(f"\n📈 COMPARISON WITH CURRENT PERFORMANCE:")
                print(f"Current avg position: {current_perf['avg_position']:.1f}")
                print(f"Current avg points: {current_perf['avg_points']:.1f}")
                print(f"Current podium rate: {current_perf['podium_rate']:.1%}")
                
                position_change = current_perf['avg_position'] - prediction['predicted_position']
                points_change = prediction['predicted_points'] - current_perf['avg_points']
                
                print(f"\n🔄 PREDICTED CHANGE:")
                if position_change > 0:
                    print(f"Position improvement: +{position_change:.1f} positions")
                else:
                    print(f"Position decline: {position_change:.1f} positions")
                
                if points_change > 0:
                    print(f"Points improvement: +{points_change:.1f} points")
                else:
                    print(f"Points decline: {points_change:.1f} points")
    
    def compare_teams(self):
        """Compare driver performance across multiple teams"""
        print("\n🏆 COMPARE TEAMS FOR DRIVER")
        print("-" * 40)
        
        # Get available drivers
        drivers = self.season_predictor.get_available_drivers()
        print("\nAvailable drivers:")
        for i, driver in enumerate(drivers.head(20), 1):
            print(f"{i:2d}. {driver['name']}")
        
        try:
            driver_idx = int(input("\nSelect driver (enter number): ")) - 1
            if driver_idx < 0 or driver_idx >= len(drivers):
                print("❌ Invalid driver selection.")
                return
            driver = drivers.iloc[driver_idx]
        except ValueError:
            print("❌ Invalid input.")
            return
        
        # Get available teams
        teams = self.season_predictor.get_available_teams()
        
        print(f"\n🎯 Comparing {driver['name']} across all teams...")
        print("-" * 80)
        print(f"{'Team':<20} {'Avg Position':<12} {'Avg Points':<10} {'Podium Rate':<12} {'Points Rate':<12}")
        print("-" * 80)
        
        comparisons = []
        
        for team in teams:
            prediction = self.season_predictor.predict_team_switch_performance(
                driver['driverId'], team['constructorId']
            )
            
            if prediction:
                comparisons.append({
                    'team': team['name'],
                    'position': prediction['predicted_position'],
                    'points': prediction['predicted_points'],
                    'podium_rate': prediction['predicted_podium_rate'],
                    'points_rate': prediction['predicted_points_finish_rate']
                })
                
                print(f"{team['name']:<20} {prediction['predicted_position']:<12.1f} "
                      f"{prediction['predicted_points']:<10.1f} "
                      f"{prediction['predicted_podium_rate']:<12.1%} "
                      f"{prediction['predicted_points_finish_rate']:<12.1%}")
        
        # Find best team
        if comparisons:
            best_team = min(comparisons, key=lambda x: x['position'])
            print(f"\n🏆 BEST TEAM FOR {driver['name']}: {best_team['team']}")
            print(f"   Average position: {best_team['position']:.1f}")
            print(f"   Average points: {best_team['points']:.1f}")
    
    def analyze_team_switch(self):
        """Analyze team switch impact"""
        print("\n🏆 TEAM SWITCH IMPACT ANALYSIS")
        print("-" * 40)
        
        # Get available drivers
        drivers = self.season_predictor.get_available_drivers()
        print("\nAvailable drivers:")
        for i, driver in enumerate(drivers.head(20), 1):
            print(f"{i:2d}. {driver['name']}")
        
        try:
            driver_idx = int(input("\nSelect driver (enter number): ")) - 1
            if driver_idx < 0 or driver_idx >= len(drivers):
                print("❌ Invalid driver selection.")
                return
            driver = drivers.iloc[driver_idx]
        except ValueError:
            print("❌ Invalid input.")
            return
        
        # Get current performance
        current_perf = self.season_predictor.get_driver_current_performance(driver['driverId'])
        if not current_perf:
            print("❌ No current performance data available.")
            return
        
        print(f"\n📊 CURRENT PERFORMANCE FOR {driver['name']}:")
        print(f"Average position: {current_perf['avg_position']:.1f}")
        print(f"Average points: {current_perf['avg_points']:.1f}")
        print(f"Podium rate: {current_perf['podium_rate']:.1%}")
        print(f"Points finish rate: {current_perf['points_finish_rate']:.1%}")
        
        # Get available teams
        teams = self.season_predictor.get_available_teams()
        
        print(f"\n🔄 TEAM SWITCH IMPACT ANALYSIS:")
        print("-" * 80)
        print(f"{'Team':<20} {'Pos Change':<12} {'Points Change':<12} {'Impact':<15}")
        print("-" * 80)
        
        impacts = []
        
        for team in teams:
            prediction = self.season_predictor.predict_team_switch_performance(
                driver['driverId'], team['constructorId']
            )
            
            if prediction:
                pos_change = current_perf['avg_position'] - prediction['predicted_position']
                points_change = prediction['predicted_points'] - current_perf['avg_points']
                
                if pos_change > 0 and points_change > 0:
                    impact = "🟢 Positive"
                elif pos_change < 0 and points_change < 0:
                    impact = "🔴 Negative"
                else:
                    impact = "🟡 Mixed"
                
                impacts.append({
                    'team': team['name'],
                    'pos_change': pos_change,
                    'points_change': points_change,
                    'impact': impact
                })
                
                print(f"{team['name']:<20} {pos_change:+<12.1f} {points_change:+<12.1f} {impact:<15}")
        
        # Summary
        positive_switches = [imp for imp in impacts if imp['impact'] == "🟢 Positive"]
        negative_switches = [imp for imp in impacts if imp['impact'] == "🔴 Negative"]
        
        print(f"\n📈 SUMMARY:")
        print(f"Positive team switches: {len(positive_switches)}")
        print(f"Negative team switches: {len(negative_switches)}")
        print(f"Mixed impact switches: {len(impacts) - len(positive_switches) - len(negative_switches)}")
        
        if positive_switches:
            best_switch = max(positive_switches, key=lambda x: x['points_change'])
            print(f"\n🏆 BEST TEAM SWITCH: {best_switch['team']}")
            print(f"   Position improvement: +{best_switch['pos_change']:.1f}")
            print(f"   Points improvement: +{best_switch['points_change']:.1f}")
    
    def view_season_data(self):
        """View available drivers and teams"""
        print("\n📊 AVAILABLE DATA")
        print("-" * 40)
        
        drivers = self.season_predictor.get_available_drivers()
        teams = self.season_predictor.get_available_teams()
        
        print(f"📈 Drivers available: {len(drivers)}")
        print(f"🏎️  Teams available: {len(teams)}")
        
        print(f"\n👥 TOP 20 DRIVERS:")
        for i, driver in enumerate(drivers.head(20), 1):
            current_team = self.season_predictor.get_driver_current_team(driver['driverId'])
            team_name = current_team['name'] if current_team else "Unknown"
            print(f"{i:2d}. {driver['name']:<25} ({team_name})")
        
        print(f"\n🏎️  ALL TEAMS:")
        for i, team in enumerate(teams, 1):
            print(f"{i:2d}. {team['name']}")
    
    def load_or_train_models(self):
        print("\n🔄 Checking for pre-trained models...")
        self.race_predictor = AdvancedRacePredictor(force_retrain=False)
        self.race_predictor.load_data()
        print("\n✅ Models are ready for instant predictions!")
    
    def predict_complete_race(self):
        """Predict complete race for any circuit"""
        if self.race_predictor.qualifying_model is None or self.race_predictor.race_model is None:
            print("❌ Models not trained. Please train the models first (Option 1).")
            return
        
        print("\n🏁 COMPLETE RACE PREDICTION")
        print("-" * 40)
        
        # Get available circuits
        circuits = self.race_predictor.get_available_circuits()
        print("\nAvailable circuits:")
        for i, circuit in enumerate(circuits.head(30), 1):
            print(f"{i:2d}. {circuit['name']}")
        
        try:
            circuit_idx = int(input("\nSelect circuit (enter number): ")) - 1
            if circuit_idx < 0 or circuit_idx >= len(circuits):
                print("❌ Invalid circuit selection.")
                return
            circuit = circuits.iloc[circuit_idx]
        except ValueError:
            print("❌ Invalid input.")
            return
        
        print(f"\n🎯 Predicting complete race for {circuit['name']}...")
        
        try:
            race_prediction = self.race_predictor.predict_complete_race(circuit['circuitId'])
            
            if race_prediction:
                print(f"\n🏁 COMPLETE RACE PREDICTION: {race_prediction['circuit_name']}")
                print("="*80)
                print(f"{'Pos':<3} {'Driver':<20} {'Team':<15} {'Qual':<4} {'Race':<4} {'Points':<6}")
                print("-" * 80)
                
                for i, pred in enumerate(race_prediction['predictions'], 1):
                    print(f"{i:<3} {pred['driver_name']:<20} {pred['team_name']:<15} "
                          f"{pred['predicted_qualifying']:<4} {pred['predicted_race']:<4} {pred['predicted_points']:<6}")
                
                # Summary statistics
                total_points = sum(pred['predicted_points'] for pred in race_prediction['predictions'])
                print(f"\n📊 SUMMARY:")
                print(f"Total points distributed: {total_points}")
                print(f"Average qualifying position: {np.mean([p['predicted_qualifying'] for p in race_prediction['predictions']]):.1f}")
                print(f"Average race position: {np.mean([p['predicted_race'] for p in race_prediction['predictions']]):.1f}")
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
    
    def predict_popular_circuits(self):
        """Predict for popular circuits"""
        if self.race_predictor.qualifying_model is None or self.race_predictor.race_model is None:
            print("❌ Models not trained. Please train the models first (Option 1).")
            return
        
        print("\n🏁 POPULAR CIRCUITS PREDICTION")
        print("-" * 40)
        
        popular_circuits = [
            "Monaco Grand Prix",
            "British Grand Prix", 
            "Italian Grand Prix",
            "Belgian Grand Prix",
            "Japanese Grand Prix",
            "Brazilian Grand Prix"
        ]
        
        print("Popular circuits available:")
        for i, circuit_name in enumerate(popular_circuits, 1):
            print(f"{i}. {circuit_name}")
        
        try:
            choice = int(input("\nSelect circuit (1-6): ")) - 1
            if choice < 0 or choice >= len(popular_circuits):
                print("❌ Invalid choice.")
                return
            
            circuit_name = popular_circuits[choice]
            
            # Find circuit ID
            circuit = self.race_predictor.races_df[
                self.race_predictor.races_df['name'].str.contains(circuit_name.split()[0], case=False)
            ]
            
            if circuit.empty:
                print(f"❌ Circuit '{circuit_name}' not found.")
                return
            
            circuit_id = circuit.iloc[0]['circuitId']
            
            print(f"\n🎯 Predicting for {circuit_name}...")
            race_prediction = self.race_predictor.predict_complete_race(circuit_id)
            
            if race_prediction:
                print(f"\n🏁 {circuit_name.upper()} PREDICTION")
                print("="*80)
                print(f"{'Pos':<3} {'Driver':<20} {'Team':<15} {'Qual':<4} {'Race':<4} {'Points':<6}")
                print("-" * 80)
                
                for i, pred in enumerate(race_prediction['predictions'], 1):
                    print(f"{i:<3} {pred['driver_name']:<20} {pred['team_name']:<15} "
                          f"{pred['predicted_qualifying']:<4} {pred['predicted_race']:<4} {pred['predicted_points']:<6}")
        
        except ValueError:
            print("❌ Invalid input.")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def view_circuits(self):
        """View available circuits"""
        print("\n🏁 AVAILABLE CIRCUITS")
        print("-" * 40)
        
        circuits = self.race_predictor.get_available_circuits()
        print(f"Total circuits available: {len(circuits)}")
        
        print(f"\n🏎️  CIRCUITS:")
        for i, circuit in enumerate(circuits, 1):
            print(f"{i:3d}. {circuit['name']}")
    
    def model_performance_menu(self):
        """Menu for model performance analysis"""
        print("\n📊 MODEL PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        if not self.race_predictor.qualifying_performance:
            print("❌ Models not trained yet. Please train the models first.")
            return
        
        print("🏁 QUALIFYING MODEL PERFORMANCE:")
        print("-" * 50)
        for model_name, perf in self.race_predictor.qualifying_performance.items():
            print(f"{model_name}:")
            print(f"  R² Score: {perf['r2']:.3f}")
            print(f"  MAE: {perf['mae']:.3f}")
            print(f"  RMSE: {perf['rmse']:.3f}")
            print()
        
        print("🏁 RACE MODEL PERFORMANCE:")
        print("-" * 50)
        for model_name, perf in self.race_predictor.race_performance.items():
            print(f"{model_name}:")
            print(f"  R² Score: {perf['r2']:.3f}")
            print(f"  MAE: {perf['mae']:.3f}")
            print(f"  RMSE: {perf['rmse']:.3f}")
            print()
        
        # Find best models
        best_qualifying = max(self.race_predictor.qualifying_performance.items(), 
                            key=lambda x: x[1]['r2'])
        best_race = max(self.race_predictor.race_performance.items(), 
                       key=lambda x: x[1]['r2'])
        
        print("🏆 BEST MODELS:")
        print(f"Qualifying: {best_qualifying[0]} (R² = {best_qualifying[1]['r2']:.3f})")
        print(f"Race: {best_race[0]} (R² = {best_race[1]['r2']:.3f})")
    
    def data_exploration_menu(self):
        """Menu for data exploration"""
        print("\n🔍 DATA EXPLORATION")
        print("-" * 40)
        
        print("📊 DATASET OVERVIEW:")
        print(f"Results: {len(self.race_predictor.results_df):,} records")
        print(f"Drivers: {len(self.race_predictor.drivers_df):,} drivers")
        print(f"Teams: {len(self.race_predictor.constructors_df):,} teams")
        print(f"Races: {len(self.race_predictor.races_df):,} races")
        print(f"Qualifying: {len(self.race_predictor.qualifying_df):,} records")
        print(f"Lap Times: {len(self.race_predictor.lap_times_df):,} records (sample)")
        
        print(f"\n🏁 2024 DRIVERS ({len(self.race_predictor.drivers_2024)}):")
        for i, driver in enumerate(self.race_predictor.drivers_2024.head(10), 1):
            print(f"{i:2d}. {driver['forename']} {driver['surname']}")
        
        if len(self.race_predictor.drivers_2024) > 10:
            print(f"... and {len(self.race_predictor.drivers_2024) - 10} more")
        
        print(f"\n🏎️  AVAILABLE CIRCUITS ({len(self.race_predictor.get_available_circuits())}):")
        circuits = self.race_predictor.get_available_circuits()
        for i, circuit in enumerate(circuits.head(10), 1):
            print(f"{i:2d}. {circuit['name']}")
        
        if len(circuits) > 10:
            print(f"... and {len(circuits) - 10} more")
    
    def quick_predictions_menu(self):
        """Menu for quick predictions"""
        print("\n🎯 QUICK PREDICTIONS")
        print("-" * 40)
        
        print("1. Max Verstappen at Red Bull vs Mercedes")
        print("2. Lewis Hamilton at Mercedes vs Ferrari")
        print("3. Charles Leclerc at Ferrari vs Red Bull")
        print("4. Lando Norris at McLaren vs Red Bull")
        print("5. Monaco Grand Prix prediction")
        print("6. British Grand Prix prediction")
        print("7. Back to main menu")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            self.quick_verstappen_comparison()
        elif choice == '2':
            self.quick_hamilton_comparison()
        elif choice == '3':
            self.quick_leclerc_comparison()
        elif choice == '4':
            self.quick_norris_comparison()
        elif choice == '5':
            self.quick_monaco_prediction()
        elif choice == '6':
            self.quick_british_prediction()
        elif choice == '7':
            return
        else:
            print("❌ Invalid choice.")
    
    def quick_verstappen_comparison(self):
        """Quick comparison for Max Verstappen"""
        print("\n🏆 MAX VERSTAPPEN - RED BULL vs MERCEDES")
        print("-" * 50)
        
        # Find Verstappen
        verstappen = self.season_predictor.drivers_df[
            self.season_predictor.drivers_df['surname'].str.contains('Verstappen', case=False)
        ]
        
        if verstappen.empty:
            print("❌ Max Verstappen not found in database.")
            return
        
        driver_id = verstappen.iloc[0]['driverId']
        
        # Find Red Bull and Mercedes
        red_bull = self.season_predictor.constructors_df[
            self.season_predictor.constructors_df['name'].str.contains('Red Bull', case=False)
        ]
        mercedes = self.season_predictor.constructors_df[
            self.season_predictor.constructors_df['name'].str.contains('Mercedes', case=False)
        ]
        
        if red_bull.empty or mercedes.empty:
            print("❌ Teams not found.")
            return
        
        red_bull_id = red_bull.iloc[0]['constructorId']
        mercedes_id = mercedes.iloc[0]['constructorId']
        
        # Current performance
        current_perf = self.season_predictor.get_driver_current_performance(driver_id)
        
        # Predictions
        red_bull_pred = self.season_predictor.predict_team_switch_performance(driver_id, red_bull_id)
        mercedes_pred = self.season_predictor.predict_team_switch_performance(driver_id, mercedes_id)
        
        print(f"📊 CURRENT PERFORMANCE:")
        if current_perf:
            print(f"Average position: {current_perf['avg_position']:.1f}")
            print(f"Average points: {current_perf['avg_points']:.1f}")
            print(f"Podium rate: {current_perf['podium_rate']:.1%}")
        
        print(f"\n🏎️  RED BULL PREDICTION:")
        if red_bull_pred:
            print(f"Average position: {red_bull_pred['predicted_position']:.1f}")
            print(f"Average points: {red_bull_pred['predicted_points']:.1f}")
            print(f"Podium rate: {red_bull_pred['predicted_podium_rate']:.1%}")
        
        print(f"\n🏎️  MERCEDES PREDICTION:")
        if mercedes_pred:
            print(f"Average position: {mercedes_pred['predicted_position']:.1f}")
            print(f"Average points: {mercedes_pred['predicted_points']:.1f}")
            print(f"Podium rate: {mercedes_pred['predicted_podium_rate']:.1%}")
    
    def quick_hamilton_comparison(self):
        """Quick comparison for Lewis Hamilton"""
        print("\n🏆 LEWIS HAMILTON - MERCEDES vs FERRARI")
        print("-" * 50)
        
        # Find Hamilton
        hamilton = self.season_predictor.drivers_df[
            self.season_predictor.drivers_df['surname'].str.contains('Hamilton', case=False)
        ]
        
        if hamilton.empty:
            print("❌ Lewis Hamilton not found in database.")
            return
        
        driver_id = hamilton.iloc[0]['driverId']
        
        # Find Mercedes and Ferrari
        mercedes = self.season_predictor.constructors_df[
            self.season_predictor.constructors_df['name'].str.contains('Mercedes', case=False)
        ]
        ferrari = self.season_predictor.constructors_df[
            self.season_predictor.constructors_df['name'].str.contains('Ferrari', case=False)
        ]
        
        if mercedes.empty or ferrari.empty:
            print("❌ Teams not found.")
            return
        
        mercedes_id = mercedes.iloc[0]['constructorId']
        ferrari_id = ferrari.iloc[0]['constructorId']
        
        # Predictions
        mercedes_pred = self.season_predictor.predict_team_switch_performance(driver_id, mercedes_id)
        ferrari_pred = self.season_predictor.predict_team_switch_performance(driver_id, ferrari_id)
        
        print(f"🏎️  MERCEDES PREDICTION:")
        if mercedes_pred:
            print(f"Average position: {mercedes_pred['predicted_position']:.1f}")
            print(f"Average points: {mercedes_pred['predicted_points']:.1f}")
            print(f"Podium rate: {mercedes_pred['predicted_podium_rate']:.1%}")
        
        print(f"\n🏎️  FERRARI PREDICTION:")
        if ferrari_pred:
            print(f"Average position: {ferrari_pred['predicted_position']:.1f}")
            print(f"Average points: {ferrari_pred['predicted_points']:.1f}")
            print(f"Podium rate: {ferrari_pred['predicted_podium_rate']:.1%}")
    
    def quick_leclerc_comparison(self):
        """Quick comparison for Charles Leclerc"""
        print("\n🏆 CHARLES LECLERC - FERRARI vs RED BULL")
        print("-" * 50)
        
        # Find Leclerc
        leclerc = self.season_predictor.drivers_df[
            self.season_predictor.drivers_df['surname'].str.contains('Leclerc', case=False)
        ]
        
        if leclerc.empty:
            print("❌ Charles Leclerc not found in database.")
            return
        
        driver_id = leclerc.iloc[0]['driverId']
        
        # Find Ferrari and Red Bull
        ferrari = self.season_predictor.constructors_df[
            self.season_predictor.constructors_df['name'].str.contains('Ferrari', case=False)
        ]
        red_bull = self.season_predictor.constructors_df[
            self.season_predictor.constructors_df['name'].str.contains('Red Bull', case=False)
        ]
        
        if ferrari.empty or red_bull.empty:
            print("❌ Teams not found.")
            return
        
        ferrari_id = ferrari.iloc[0]['constructorId']
        red_bull_id = red_bull.iloc[0]['constructorId']
        
        # Predictions
        ferrari_pred = self.season_predictor.predict_team_switch_performance(driver_id, ferrari_id)
        red_bull_pred = self.season_predictor.predict_team_switch_performance(driver_id, red_bull_id)
        
        print(f"🏎️  FERRARI PREDICTION:")
        if ferrari_pred:
            print(f"Average position: {ferrari_pred['predicted_position']:.1f}")
            print(f"Average points: {ferrari_pred['predicted_points']:.1f}")
            print(f"Podium rate: {ferrari_pred['predicted_podium_rate']:.1%}")
        
        print(f"\n🏎️  RED BULL PREDICTION:")
        if red_bull_pred:
            print(f"Average position: {red_bull_pred['predicted_position']:.1f}")
            print(f"Average points: {red_bull_pred['predicted_points']:.1f}")
            print(f"Podium rate: {red_bull_pred['predicted_podium_rate']:.1%}")
    
    def quick_norris_comparison(self):
        """Quick comparison for Lando Norris"""
        print("\n🏆 LANDO NORRIS - MCLAREN vs RED BULL")
        print("-" * 50)
        
        # Find Norris
        norris = self.season_predictor.drivers_df[
            self.season_predictor.drivers_df['surname'].str.contains('Norris', case=False)
        ]
        
        if norris.empty:
            print("❌ Lando Norris not found in database.")
            return
        
        driver_id = norris.iloc[0]['driverId']
        
        # Find McLaren and Red Bull
        mclaren = self.season_predictor.constructors_df[
            self.season_predictor.constructors_df['name'].str.contains('McLaren', case=False)
        ]
        red_bull = self.season_predictor.constructors_df[
            self.season_predictor.constructors_df['name'].str.contains('Red Bull', case=False)
        ]
        
        if mclaren.empty or red_bull.empty:
            print("❌ Teams not found.")
            return
        
        mclaren_id = mclaren.iloc[0]['constructorId']
        red_bull_id = red_bull.iloc[0]['constructorId']
        
        # Predictions
        mclaren_pred = self.season_predictor.predict_team_switch_performance(driver_id, mclaren_id)
        red_bull_pred = self.season_predictor.predict_team_switch_performance(driver_id, red_bull_id)
        
        print(f"🏎️  MCLAREN PREDICTION:")
        if mclaren_pred:
            print(f"Average position: {mclaren_pred['predicted_position']:.1f}")
            print(f"Average points: {mclaren_pred['predicted_points']:.1f}")
            print(f"Podium rate: {mclaren_pred['predicted_podium_rate']:.1%}")
        
        print(f"\n🏎️  RED BULL PREDICTION:")
        if red_bull_pred:
            print(f"Average position: {red_bull_pred['predicted_position']:.1f}")
            print(f"Average points: {red_bull_pred['predicted_points']:.1f}")
            print(f"Podium rate: {red_bull_pred['predicted_podium_rate']:.1%}")
    
    def quick_monaco_prediction(self):
        """Quick Monaco Grand Prix prediction"""
        if self.race_predictor.qualifying_model is None or self.race_predictor.race_model is None:
            print("❌ Models not trained. Please train the models first.")
            return
        
        print("\n🏁 MONACO GRAND PRIX PREDICTION")
        print("-" * 50)
        
        # Find Monaco circuit
        monaco = self.race_predictor.races_df[
            self.race_predictor.races_df['name'].str.contains('Monaco', case=False)
        ]
        
        if monaco.empty:
            print("❌ Monaco Grand Prix not found.")
            return
        
        circuit_id = monaco.iloc[0]['circuitId']
        
        print("🎯 Predicting Monaco Grand Prix...")
        race_prediction = self.race_predictor.predict_complete_race(circuit_id)
        
        if race_prediction:
            print(f"\n🏁 MONACO GRAND PRIX PREDICTION")
            print("="*80)
            print(f"{'Pos':<3} {'Driver':<20} {'Team':<15} {'Qual':<4} {'Race':<4} {'Points':<6}")
            print("-" * 80)
            
            for i, pred in enumerate(race_prediction['predictions'], 1):
                print(f"{i:<3} {pred['driver_name']:<20} {pred['team_name']:<15} "
                      f"{pred['predicted_qualifying']:<4} {pred['predicted_race']:<4} {pred['predicted_points']:<6}")
    
    def quick_british_prediction(self):
        """Quick British Grand Prix prediction"""
        if self.race_predictor.qualifying_model is None or self.race_predictor.race_model is None:
            print("❌ Models not trained. Please train the models first.")
            return
        
        print("\n🏁 BRITISH GRAND PRIX PREDICTION")
        print("-" * 50)
        
        # Find British Grand Prix
        british = self.race_predictor.races_df[
            self.race_predictor.races_df['name'].str.contains('British', case=False)
        ]
        
        if british.empty:
            print("❌ British Grand Prix not found.")
            return
        
        circuit_id = british.iloc[0]['circuitId']
        
        print("🎯 Predicting British Grand Prix...")
        race_prediction = self.race_predictor.predict_complete_race(circuit_id)
        
        if race_prediction:
            print(f"\n🏁 BRITISH GRAND PRIX PREDICTION")
            print("="*80)
            print(f"{'Pos':<3} {'Driver':<20} {'Team':<15} {'Qual':<4} {'Race':<4} {'Points':<6}")
            print("-" * 80)
            
            for i, pred in enumerate(race_prediction['predictions'], 1):
                print(f"{i:<3} {pred['driver_name']:<20} {pred['team_name']:<15} "
                      f"{pred['predicted_qualifying']:<4} {pred['predicted_race']:<4} {pred['predicted_points']:<6}")
    
    def force_retrain_models(self):
        print("\n⚡ Forcing retrain of all models. This will overwrite saved models and may take several minutes.")
        confirm = input("Are you sure you want to retrain? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Retrain cancelled.")
            return
        self.race_predictor = AdvancedRacePredictor(force_retrain=True)
        self.race_predictor.load_data()
        print("\n✅ Models retrained and saved!")
    
    def run(self):
        """Run the interactive predictor"""
        print("🏎️  Welcome to the F1 Race Prediction System! 🏎️")
        print("Loading data and initializing models...")
        
        while True:
            self.display_menu()
            choice = input("\nEnter your choice (0-5): ").strip()
            
            if choice == '0':
                print("👋 Thank you for using the F1 Race Prediction System!")
                break
            elif choice == '1':
                self.season_prediction_menu()
            elif choice == '2':
                self.complete_race_menu()
            elif choice == '3':
                self.model_performance_menu()
            elif choice == '4':
                self.data_exploration_menu()
            elif choice == '5':
                self.quick_predictions_menu()
            else:
                print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    predictor = InteractiveRacePredictor()
    predictor.run() 
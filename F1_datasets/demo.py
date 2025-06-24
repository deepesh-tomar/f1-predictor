#!/usr/bin/env python3
"""
Quick Demo of the F1 Driver Team Switch Predictor
This script demonstrates the core functionality without user interaction.
"""

from driver_team_switch_predictor import DriverTeamSwitchPredictor
import time

def main():
    print("🏎️  F1 DRIVER TEAM SWITCH PREDICTOR - DEMO")
    print("=" * 60)
    
    # Initialize predictor
    print("🚀 Initializing predictor...")
    predictor = DriverTeamSwitchPredictor()
    
    try:
        # Load data
        print("📊 Loading F1 datasets...")
        predictor.load_data()
        
        # Train model
        print("🤖 Training prediction model...")
        predictor.train_model()
        print("✅ Model trained successfully!")
        
        # Show available data
        print(f"\n📋 Dataset Overview:")
        print(f"  Drivers: {len(predictor.get_available_drivers())}")
        print(f"  Teams: {len(predictor.get_available_teams())}")
        print(f"  Circuits: {len(predictor.get_available_circuits())}")
        
        # Example predictions
        print("\n" + "="*60)
        print("🎯 EXAMPLE PREDICTIONS")
        print("="*60)
        
        # Get some example IDs from the data
        drivers = predictor.get_available_drivers()
        teams = predictor.get_available_teams()
        circuits = predictor.get_available_circuits()
        
        # Find some interesting examples
        print("\n🔍 Finding example drivers and teams...")
        
        # Look for well-known drivers (by name)
        hamilton = drivers[drivers['surname'].str.contains('Hamilton', case=False, na=False)]
        alonso = drivers[drivers['surname'].str.contains('Alonso', case=False, na=False)]
        vettel = drivers[drivers['surname'].str.contains('Vettel', case=False, na=False)]
        
        # Look for well-known teams
        ferrari = teams[teams['name'].str.contains('Ferrari', case=False, na=False)]
        mclaren = teams[teams['name'].str.contains('McLaren', case=False, na=False)]
        mercedes = teams[teams['name'].str.contains('Mercedes', case=False, na=False)]
        
        # Look for popular circuits
        australia = circuits[circuits['name'].str.contains('Australian', case=False, na=False)]
        monaco = circuits[circuits['name'].str.contains('Monaco', case=False, na=False)]
        silverstone = circuits[circuits['name'].str.contains('British', case=False, na=False)]
        
        # Make some example predictions
        examples = []
        
        if not hamilton.empty and not ferrari.empty and not australia.empty:
            examples.append({
                'driver_id': hamilton.iloc[0]['driverId'],
                'driver_name': f"{hamilton.iloc[0]['forename']} {hamilton.iloc[0]['surname']}",
                'team_id': ferrari.iloc[0]['constructorId'],
                'team_name': ferrari.iloc[0]['name'],
                'circuit_id': australia.iloc[0]['circuitId'],
                'circuit_name': australia.iloc[0]['name'],
                'description': "Hamilton to Ferrari at Australian GP"
            })
        
        if not alonso.empty and not mclaren.empty and not monaco.empty:
            examples.append({
                'driver_id': alonso.iloc[0]['driverId'],
                'driver_name': f"{alonso.iloc[0]['forename']} {alonso.iloc[0]['surname']}",
                'team_id': mclaren.iloc[0]['constructorId'],
                'team_name': mclaren.iloc[0]['name'],
                'circuit_id': monaco.iloc[0]['circuitId'],
                'circuit_name': monaco.iloc[0]['name'],
                'description': "Alonso to McLaren at Monaco GP"
            })
        
        if not vettel.empty and not mercedes.empty and not silverstone.empty:
            examples.append({
                'driver_id': vettel.iloc[0]['driverId'],
                'driver_name': f"{vettel.iloc[0]['forename']} {vettel.iloc[0]['surname']}",
                'team_id': mercedes.iloc[0]['constructorId'],
                'team_name': mercedes.iloc[0]['name'],
                'circuit_id': silverstone.iloc[0]['circuitId'],
                'circuit_name': silverstone.iloc[0]['name'],
                'description': "Vettel to Mercedes at British GP"
            })
        
        # If no specific examples found, use first available
        if not examples:
            print("⚠️  Using generic examples...")
            examples = [
                {
                    'driver_id': drivers.iloc[0]['driverId'],
                    'driver_name': f"{drivers.iloc[0]['forename']} {drivers.iloc[0]['surname']}",
                    'team_id': teams.iloc[0]['constructorId'],
                    'team_name': teams.iloc[0]['name'],
                    'circuit_id': circuits.iloc[0]['circuitId'],
                    'circuit_name': circuits.iloc[0]['name'],
                    'description': "Generic example"
                }
            ]
        
        # Make predictions
        for i, example in enumerate(examples, 1):
            print(f"\n📊 Example {i}: {example['description']}")
            print("-" * 50)
            
            try:
                prediction = predictor.predict_driver_performance(
                    example['driver_id'],
                    example['team_id'],
                    example['circuit_id']
                )
                
                if prediction:
                    print(f"Driver: {prediction['driver']}")
                    print(f"New Team: {prediction['new_team']}")
                    print(f"Circuit: {prediction['circuit']}")
                    print(f"Predicted Position: {prediction['predicted_position']}")
                    print(f"Predicted Points: {prediction['predicted_points']}")
                    print(f"Confidence: {prediction['confidence']:.1%}")
                    
                    # Interpret result
                    position = prediction['predicted_position']
                    if position <= 3:
                        result = "🏆 PODIUM FINISH!"
                    elif position <= 10:
                        result = "📊 POINTS FINISH"
                    elif position <= 15:
                        result = "🏁 MIDFIELD FINISH"
                    else:
                        result = "💪 BACK OF THE GRID"
                    
                    print(f"Expected Result: {result}")
                else:
                    print("❌ Could not make prediction (insufficient data)")
                    
            except Exception as e:
                print(f"❌ Error in prediction: {e}")
            
            time.sleep(1)  # Brief pause between predictions
        
        # Show team switch analysis example
        if len(examples) >= 2:
            print("\n" + "="*60)
            print("🔄 TEAM SWITCH ANALYSIS EXAMPLE")
            print("="*60)
            
            try:
                # Use first two examples to simulate a team switch
                old_team_id = teams.iloc[0]['constructorId'] if len(teams) > 0 else examples[0]['team_id']
                new_team_id = teams.iloc[1]['constructorId'] if len(teams) > 1 else examples[1]['team_id']
                driver_id = examples[0]['driver_id']
                circuit_id = examples[0]['circuit_id']
                
                predictor.analyze_team_switch_scenario(
                    driver_id, old_team_id, new_team_id, circuit_id
                )
                
            except Exception as e:
                print(f"❌ Error in team switch analysis: {e}")
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED!")
        print("="*60)
        print("\nTo use the full interactive version, run:")
        print("python interactive_predictor.py")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Make sure all CSV files are in the current directory.")

if __name__ == "__main__":
    main() 
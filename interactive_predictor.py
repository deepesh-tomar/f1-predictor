#!/usr/bin/env python3
"""
Interactive F1 Driver Team Switch Predictor
This script allows you to predict how a driver will perform when switching teams.
"""

from driver_team_switch_predictor import DriverTeamSwitchPredictor
import pandas as pd

def display_menu():
    """Display the main menu"""
    print("\n" + "="*60)
    print("🏎️  F1 DRIVER TEAM SWITCH PREDICTOR 🏎️")
    print("="*60)
    print("1. View available drivers")
    print("2. View available teams")
    print("3. View available circuits")
    print("4. Predict driver performance with new team")
    print("5. Analyze team switch impact")
    print("6. Exit")
    print("="*60)

def display_drivers(predictor):
    """Display available drivers"""
    drivers = predictor.get_available_drivers()
    print(f"\n📋 Available Drivers ({len(drivers)} total):")
    print("-" * 60)
    print(f"{'ID':<5} {'Name':<30} {'Nationality':<15}")
    print("-" * 60)
    
    for _, driver in drivers.iterrows():
        name = f"{driver['forename']} {driver['surname']}"
        print(f"{driver['driverId']:<5} {name:<30} {driver['nationality']:<15}")

def display_teams(predictor):
    """Display available teams"""
    teams = predictor.get_available_teams()
    print(f"\n🏁 Available Teams ({len(teams)} total):")
    print("-" * 60)
    print(f"{'ID':<5} {'Team Name':<30} {'Nationality':<15}")
    print("-" * 60)
    
    for _, team in teams.iterrows():
        print(f"{team['constructorId']:<5} {team['name']:<30} {team['nationality']:<15}")

def display_circuits(predictor):
    """Display available circuits"""
    circuits = predictor.get_available_circuits()
    print(f"\n🏟️  Available Circuits ({len(circuits)} total):")
    print("-" * 60)
    print(f"{'ID':<5} {'Circuit Name':<50}")
    print("-" * 60)
    
    for _, circuit in circuits.iterrows():
        print(f"{circuit['circuitId']:<5} {circuit['name']:<50}")

def get_user_selection(prompt, options_df, id_col, name_col):
    """Get user selection from a list of options"""
    while True:
        try:
            choice = input(prompt)
            if choice.lower() == 'q':
                return None
            
            choice_id = int(choice)
            if choice_id in options_df[id_col].values:
                selected = options_df[options_df[id_col] == choice_id].iloc[0]
                print(f"✅ Selected: {selected[name_col]}")
                return choice_id
            else:
                print(f"❌ Invalid {id_col}. Please try again or enter 'q' to quit.")
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit.")

def predict_performance(predictor):
    """Predict driver performance with new team"""
    print("\n🎯 PREDICT DRIVER PERFORMANCE WITH NEW TEAM")
    print("-" * 50)
    
    # Get driver selection
    drivers = predictor.get_available_drivers()
    print("\nSelect a driver:")
    for _, driver in drivers.head(20).iterrows():
        name = f"{driver['forename']} {driver['surname']}"
        print(f"{driver['driverId']}: {name} ({driver['nationality']})")
    print("... (showing first 20, enter ID to select)")
    
    driver_id = get_user_selection("\nEnter driver ID (or 'q' to quit): ", drivers, 'driverId', 'forename')
    if driver_id is None:
        return
    
    # Get new team selection
    teams = predictor.get_available_teams()
    print("\nSelect the new team:")
    for _, team in teams.head(20).iterrows():
        print(f"{team['constructorId']}: {team['name']} ({team['nationality']})")
    print("... (showing first 20, enter ID to select)")
    
    new_team_id = get_user_selection("\nEnter new team ID (or 'q' to quit): ", teams, 'constructorId', 'name')
    if new_team_id is None:
        return
    
    # Get circuit selection
    circuits = predictor.get_available_circuits()
    print("\nSelect a circuit:")
    for _, circuit in circuits.head(20).iterrows():
        print(f"{circuit['circuitId']}: {circuit['name']}")
    print("... (showing first 20, enter ID to select)")
    
    circuit_id = get_user_selection("\nEnter circuit ID (or 'q' to quit): ", circuits, 'circuitId', 'name')
    if circuit_id is None:
        return
    
    # Make prediction
    print("\n🔮 Making prediction...")
    try:
        prediction = predictor.predict_driver_performance(driver_id, new_team_id, circuit_id)
        
        if prediction:
            print("\n" + "="*50)
            print("🎯 PREDICTION RESULTS")
            print("="*50)
            print(f"Driver: {prediction['driver']}")
            print(f"New Team: {prediction['new_team']}")
            print(f"Circuit: {prediction['circuit']}")
            print(f"Predicted Position: {prediction['predicted_position']}")
            print(f"Predicted Points: {prediction['predicted_points']}")
            print(f"Confidence: {prediction['confidence']:.1%}")
            
            # Interpret the prediction
            position = prediction['predicted_position']
            if position <= 3:
                print(f"🏆 Expected Result: PODIUM FINISH!")
            elif position <= 10:
                print(f"📊 Expected Result: POINTS FINISH")
            elif position <= 15:
                print(f"🏁 Expected Result: MIDFIELD FINISH")
            else:
                print(f"💪 Expected Result: BACK OF THE GRID")
                
        else:
            print("❌ Could not make prediction. Insufficient data.")
            
    except Exception as e:
        print(f"❌ Error making prediction: {e}")

def analyze_team_switch(predictor):
    """Analyze the impact of team switch"""
    print("\n🔄 ANALYZE TEAM SWITCH IMPACT")
    print("-" * 50)
    
    # Get driver selection
    drivers = predictor.get_available_drivers()
    print("\nSelect a driver:")
    for _, driver in drivers.head(20).iterrows():
        name = f"{driver['forename']} {driver['surname']}"
        print(f"{driver['driverId']}: {name} ({driver['nationality']})")
    print("... (showing first 20, enter ID to select)")
    
    driver_id = get_user_selection("\nEnter driver ID (or 'q' to quit): ", drivers, 'driverId', 'forename')
    if driver_id is None:
        return
    
    # Get old team selection
    teams = predictor.get_available_teams()
    print("\nSelect the OLD team:")
    for _, team in teams.head(20).iterrows():
        print(f"{team['constructorId']}: {team['name']} ({team['nationality']})")
    print("... (showing first 20, enter ID to select)")
    
    old_team_id = get_user_selection("\nEnter old team ID (or 'q' to quit): ", teams, 'constructorId', 'name')
    if old_team_id is None:
        return
    
    # Get new team selection
    print("\nSelect the NEW team:")
    for _, team in teams.head(20).iterrows():
        print(f"{team['constructorId']}: {team['name']} ({team['nationality']})")
    print("... (showing first 20, enter ID to select)")
    
    new_team_id = get_user_selection("\nEnter new team ID (or 'q' to quit): ", teams, 'constructorId', 'name')
    if new_team_id is None:
        return
    
    # Get circuit selection
    circuits = predictor.get_available_circuits()
    print("\nSelect a circuit:")
    for _, circuit in circuits.head(20).iterrows():
        print(f"{circuit['circuitId']}: {circuit['name']}")
    print("... (showing first 20, enter ID to select)")
    
    circuit_id = get_user_selection("\nEnter circuit ID (or 'q' to quit): ", circuits, 'circuitId', 'name')
    if circuit_id is None:
        return
    
    # Analyze the switch
    print("\n🔮 Analyzing team switch impact...")
    try:
        predictor.analyze_team_switch_scenario(driver_id, old_team_id, new_team_id, circuit_id)
    except Exception as e:
        print(f"❌ Error analyzing team switch: {e}")

def main():
    """Main interactive function"""
    print("🚀 Initializing F1 Driver Team Switch Predictor...")
    
    # Initialize predictor
    predictor = DriverTeamSwitchPredictor()
    
    try:
        print("📊 Loading F1 datasets...")
        predictor.load_data()
        print("🤖 Training prediction model...")
        predictor.train_model()
        print("✅ Ready to make predictions!")
        
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        print("Make sure all CSV files are in the current directory.")
        return
    
    # Main interaction loop
    while True:
        display_menu()
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                display_drivers(predictor)
            elif choice == '2':
                display_teams(predictor)
            elif choice == '3':
                display_circuits(predictor)
            elif choice == '4':
                predict_performance(predictor)
            elif choice == '5':
                analyze_team_switch(predictor)
            elif choice == '6':
                print("\n👋 Thanks for using the F1 Driver Team Switch Predictor!")
                break
            else:
                print("❌ Invalid choice. Please enter a number between 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the F1 Driver Team Switch Predictor!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main() 
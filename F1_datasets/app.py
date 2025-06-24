#!/usr/bin/env python3
"""
F1 Driver Team Switch Predictor - Web Application
A Flask web app with a form interface for predicting driver performance when switching teams.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect
from advanced_race_predictor import AdvancedRacePredictor
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'f1_predictor_secret_key_2025'

# Global predictor instance
predictor = None

def initialize_predictor():
    """Initialize the predictor if not already done"""
    global predictor
    if predictor is None:
        try:
            print("🏎️ Starting F1 Driver Team Switch Predictor Web App...")
            print("📊 Initializing predictor...")
            predictor = AdvancedRacePredictor(force_retrain=False)
            predictor.load_data()
            print("✅ Predictor initialized successfully!")
            return True
        except Exception as e:
            print(f"Error initializing predictor: {e}")
            return False
    return True

def search_drivers(query):
    """Search drivers by name"""
    if not query or not predictor:
        return []
    
    # Use predictor's driver data
    drivers = predictor.drivers_df
    # Search in both forename and surname
    matches = drivers[
        drivers['forename'].str.contains(query, case=False, na=False) |
        drivers['surname'].str.contains(query, case=False, na=False)
    ]
    results = []
    for _, driver in matches.head(10).iterrows():
        results.append({
            'id': driver['driverId'],
            'name': f"{driver['forename']} {driver['surname']}"
        })
    return results

def search_teams(query):
    """Search teams by name"""
    if not query or not predictor:
        return []
    
    # Use predictor's constructor data
    constructors = predictor.constructors_df
    matches = constructors[constructors['name'].str.contains(query, case=False, na=False)]
    results = []
    for _, team in matches.head(10).iterrows():
        results.append({
            'id': team['constructorId'],
            'name': team['name']
        })
    return results

def search_circuits(query):
    """Search circuits by name"""
    if not query or not predictor:
        return []
    
    # Use predictor's circuit data
    circuits = predictor.get_available_circuits()
    matches = circuits[circuits['name'].str.contains(query, case=False, na=False)]
    results = []
    for _, circuit in matches.head(10).iterrows():
        results.append({
            'id': circuit['circuitId'],
            'name': circuit['name']
        })
    return results

def get_driver_name(driver_id):
    """Get driver name from ID"""
    try:
        if predictor:
            driver = predictor.drivers_df[predictor.drivers_df['driverId'] == int(driver_id)]
            if not driver.empty:
                return f"{driver.iloc[0]['forename']} {driver.iloc[0]['surname']}"
        return f"Driver {driver_id}"
    except:
        return f"Driver {driver_id}"

def get_team_name(team_id):
    """Get team name from ID"""
    try:
        if predictor:
            team = predictor.constructors_df[predictor.constructors_df['constructorId'] == int(team_id)]
            if not team.empty:
                return team.iloc[0]['name']
        return f"Team {team_id}"
    except:
        return f"Team {team_id}"

def get_circuit_name(circuit_id):
    """Get circuit name from ID"""
    try:
        if predictor:
            circuit = predictor.races_df[predictor.races_df['circuitId'] == int(circuit_id)]
            if not circuit.empty:
                return circuit.iloc[0]['name']
        return f"Circuit {circuit_id}"
    except:
        return f"Circuit {circuit_id}"

@app.route('/')
def index():
    """Main page with the prediction form"""
    return render_template('index.html')

@app.route('/search_drivers')
def search_drivers_route():
    """API endpoint for driver search"""
    query = request.args.get('q', '')
    if not initialize_predictor():
        return jsonify({'error': 'Predictor not initialized'})
    
    results = search_drivers(query)
    return jsonify(results)

@app.route('/search_teams')
def search_teams_route():
    """API endpoint for team search"""
    query = request.args.get('q', '')
    if not initialize_predictor():
        return jsonify({'error': 'Predictor not initialized'})
    
    results = search_teams(query)
    return jsonify(results)

@app.route('/search_circuits')
def search_circuits_route():
    """API endpoint for circuit search"""
    query = request.args.get('q', '')
    if not initialize_predictor():
        return jsonify({'error': 'Predictor not initialized'})
    
    results = search_circuits(query)
    return jsonify(results)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Initialize predictor if not already done
        if not initialize_predictor():
            flash('Error: Predictor not initialized. Please check if all data files are present.', 'error')
            return redirect('/')
        
        # Get form data
        driver_id = request.form.get('driver_id')
        team_id = request.form.get('team_id')
        circuit_id = request.form.get('circuit_id')
        
        if not all([driver_id, team_id, circuit_id]):
            flash('Please select all required fields', 'error')
            return redirect('/')
        
        # Get names for display
        driver_name = get_driver_name(driver_id)
        team_name = get_team_name(team_id)
        circuit_name = get_circuit_name(circuit_id)
        
        # Make prediction using the new model
        prediction = predictor.predict_driver_performance(
            driver_id=int(driver_id),
            constructor_id=int(team_id),
            circuit_id=int(circuit_id)
        )
        
        # Check if prediction was successful
        if not prediction:
            flash('Prediction failed: insufficient data or model error. Please try a different combination.', 'error')
            return redirect('/')
        
        # Determine result category and emoji
        position = prediction['predicted_position']
        if position <= 3:
            result_category = "Podium Finish"
            result_emoji = "🏆"
        elif position <= 10:
            result_category = "Points Finish"
            result_emoji = "🎯"
        elif position <= 15:
            result_category = "Midfield Finish"
            result_emoji = "📊"
        else:
            result_category = "Back of the Grid"
            result_emoji = "🔧"
        
        # Prepare result data
        result = {
            'driver': driver_name,
            'team': team_name,
            'circuit': circuit_name,
            'predicted_position': f"{position:.1f}",
            'predicted_qualifying': f"{prediction['predicted_qualifying']:.1f}",
            'predicted_points': prediction['predicted_points'],
            'confidence': prediction['confidence'],
            'result_category': result_category,
            'result_emoji': result_emoji,
            'prediction_intervals': prediction['prediction_intervals'],
            'uncertainty': prediction['uncertainty'],
            'features_used': prediction.get('features_used', 0)
        }
        
        return render_template('prediction_result.html', prediction=result)
        
    except Exception as e:
        flash(f'Error making prediction: {str(e)}', 'error')
        return redirect('/')

@app.route('/predict_circuit_order', methods=['POST'])
def predict_circuit_order():
    """Handle circuit start and finish order prediction"""
    if not initialize_predictor():
        flash('Error: Predictor not initialized. Please check if all data files are present.', 'error')
        return redirect('/')
    
    try:
        # Get form data
        circuit_id = request.form.get('circuit_id')
        
        if not circuit_id:
            flash('Please select a circuit', 'error')
            return redirect('/')
        
        # Get circuit name for display
        circuit_name = get_circuit_name(circuit_id)
        
        # Make complete race prediction for the circuit
        race_prediction = predictor.predict_complete_race(int(circuit_id))
        
        if not race_prediction:
            flash('Could not predict circuit order. Insufficient data.', 'error')
            return redirect('/')
        
        return render_template('circuit_order_result.html', prediction=race_prediction)
        
    except Exception as e:
        flash(f'Error predicting circuit order: {str(e)}', 'error')
        return redirect('/')

@app.route('/stats')
def stats():
    """Show dataset statistics"""
    if not initialize_predictor():
        flash('Error: Predictor not initialized.', 'error')
        return render_template('index.html')
    
    try:
        drivers_count = len(predictor.get_available_drivers())
        teams_count = len(predictor.get_available_teams())
        circuits_count = len(predictor.get_available_circuits())
        
        stats_data = {
            'drivers': drivers_count,
            'teams': teams_count,
            'circuits': circuits_count,
            'years': '1950-2024',
            'total_races': len(predictor.races_df),
            'total_results': len(predictor.results_df)
        }
        
        return render_template('stats.html', stats=stats_data)
    except Exception as e:
        flash(f'Error loading statistics: {str(e)}', 'error')
        return render_template('index.html')

@app.route('/qual_race_results', methods=['POST'])
def qual_race_results():
    if not initialize_predictor():
        flash('Error: Predictor not initialized. Please check if all data files are present.', 'error')
        return redirect('/')
    try:
        circuit_id = request.form.get('circuit_id')
        if not circuit_id:
            flash('Please select a circuit', 'error')
            return redirect('/')
        circuit_name = get_circuit_name(circuit_id)
        race_prediction = predictor.predict_complete_race(int(circuit_id))
        if not race_prediction:
            flash('Could not predict results. Insufficient data.', 'error')
            return redirect('/')
        return render_template('qual_race_results.html',
                               circuit_name=race_prediction['circuit_name'],
                               qualifying_order=race_prediction['qualifying_order'],
                               race_order=race_prediction['race_order'])
    except Exception as e:
        flash(f'Error predicting qualification & race results: {str(e)}', 'error')
        return redirect('/')

if __name__ == '__main__':
    print("🏎️ Starting F1 Driver Team Switch Predictor Web App...")
    print("📊 Initializing predictor...")
    
    if initialize_predictor():
        print("✅ Predictor initialized successfully!")
        print("🌐 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:5001")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("❌ Failed to initialize predictor. Please check your data files.") 
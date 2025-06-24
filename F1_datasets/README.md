# 🏎️ F1 Driver Team Switch Predictor

A sophisticated machine learning web application that predicts F1 driver performance when switching teams, using advanced ensemble models and comprehensive feature engineering.

## 🎯 What This Project Does

This predictor analyzes:
- **Driver's historical performance** (recent form, consistency, experience)
- **Team's capabilities** (recent performance, reliability, qualifying pace)
- **Circuit characteristics** (track experience, historical performance)
- **Combined factors** to predict race position and points

## 🚀 Features

### **Advanced Machine Learning Models**
- **Dual Model Architecture**: Separate models for qualifying and race position prediction
- **Ensemble Learning**: Combines XGBoost, LightGBM, Random Forest, and Neural Networks
- **GridSearchCV Optimization**: Automatic hyperparameter tuning for best performance
- **Feature Engineering**: 50+ features including driver history, team performance, and circuit characteristics

### **Web Application**
- **Modern UI**: Beautiful, responsive design with F1-themed styling
- **Real-time Search**: Interactive search for drivers, teams, and circuits
- **Two Prediction Modes**:
  - **Single Race Prediction**: Predict performance for a specific driver-team-circuit combination
  - **Team Switch Analysis**: Compare performance before and after team changes
- **Statistics Dashboard**: Comprehensive dataset insights and analytics

### **Key Capabilities**
- **Performance Prediction**: Get predicted race position, qualifying position, and points
- **Confidence Scoring**: Understand prediction reliability with confidence intervals
- **Visual Results**: Emojis and color-coded impact indicators
- **Mobile-Friendly**: Responsive design that works on all devices

## 📊 Model Architecture

### **Feature Engineering**
- **Driver Features**: Recent performance, qualifying consistency, experience, trends
- **Team Features**: Historical performance, current season data, qualifying capabilities
- **Circuit Features**: Track characteristics, frequency, recent performance
- **Interaction Features**: Driver-team synergy, performance gaps, circuit experience

### **Model Performance**
- **Qualifying Model**: R² scores up to 0.35+ with MAE ~3.7 positions
- **Race Model**: R² scores up to 0.35+ with MAE ~3.7 positions
- **Ensemble Approach**: Combines multiple models for robust predictions

## 🛠️ Installation & Setup

### **Prerequisites**
```bash
# Required Python packages
pip install pandas numpy scikit-learn xgboost lightgbm flask matplotlib seaborn joblib
```

### **Data Files**
Ensure you have the following CSV files in your project directory:
- `drivers.csv` - Driver information
- `constructors.csv` - Team/constructor information
- `races.csv` - Race and circuit information
- `results.csv` - Race results
- `qualifying.csv` - Qualifying results
- `lap_times.csv` - Lap time data

### **Running the Application**
```bash
# Start the web application
python app.py

# Access the application
# Open your browser and go to: http://localhost:5001
```

## 🎯 Usage Examples

### **Single Race Prediction**
1. Select a driver (e.g., "Lewis Hamilton")
2. Choose a team (e.g., "Ferrari")
3. Pick a circuit (e.g., "Monaco")
4. Get predictions for race position, qualifying position, and points

### **Team Switch Analysis**
1. Select a driver
2. Choose their current team
3. Select their new team
4. Pick a circuit
5. Compare performance changes and impact

### **Example Scenarios**
- **Hamilton to Ferrari**: Predict how Lewis Hamilton would perform at Ferrari
- **Alonso to McLaren**: Analyze Fernando Alonso's potential at McLaren
- **Circuit-Specific Analysis**: See how drivers perform at specific tracks

## 📈 Model Training

### **Automatic Training**
The system automatically trains models on first run:
```python
predictor = AdvancedRacePredictor(force_retrain=False)
predictor.load_data()  # Automatically loads or trains models
```

### **Manual Training**
```python
# Force retrain models
predictor = AdvancedRacePredictor(force_retrain=True)
predictor.load_data()

# Train specific models
predictor.train_qualifying_model()
predictor.train_race_model()
```

### **Model Persistence**
- Models are automatically saved to disk for fast startup
- Pre-trained models are loaded on subsequent runs
- Model files: `qualifying_model.joblib`, `race_model.joblib`

## 🔧 Technical Details

### **File Structure**
```
F1_datasets/
├── app.py                          # Flask web application
├── advanced_race_predictor.py      # Main predictor class
├── templates/                      # HTML templates
│   ├── index.html                 # Main page
│   ├── prediction_result.html     # Single prediction results
│   ├── team_switch_result.html    # Team switch analysis results
│   └── stats.html                 # Statistics page
├── *.csv                          # F1 dataset files
└── *.joblib                       # Saved model files
```

### **API Endpoints**
- `GET /` - Main prediction interface
- `GET /stats` - Dataset statistics
- `POST /predict` - Single race prediction
- `POST /team_switch_analysis` - Team switch analysis
- `GET /search_drivers` - Driver search API
- `GET /search_teams` - Team search API
- `GET /search_circuits` - Circuit search API

### **Prediction Output**
```python
{
    'predicted_position': 5.2,
    'predicted_qualifying': 4.8,
    'predicted_points': 10.0,
    'confidence': 0.75,
    'uncertainty': 0.25,
    'prediction_intervals': {
        'position_lower': 3.2,
        'position_upper': 7.2,
        'points_lower': 6.0,
        'points_upper': 15.0
    },
    'features_used': 67
}
```

## 🎨 UI Features

### **Design Elements**
- **F1 Theme**: Red and dark color scheme matching F1 branding
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Interactive Elements**: Hover effects, animations, and visual feedback
- **Modern Cards**: Glassmorphism design with backdrop blur effects

### **Visual Indicators**
- **Position Badges**: Color-coded race positions
- **Points Badges**: Golden points indicators
- **Confidence Bars**: Visual confidence representation
- **Impact Emojis**: 📈 Improvement, 📉 Decline, ➡️ Similar

## 📊 Dataset Statistics

The system includes comprehensive F1 data:
- **861 Drivers** from 1950-2024
- **212 Teams/Constructors** across F1 history
- **92 Circuits** worldwide
- **1,125 Races** with complete results
- **Comprehensive Results** including qualifying and lap times

## 🔮 Future Enhancements

### **Planned Features**
- **Weather Integration**: Include weather conditions in predictions
- **Car Development**: Track car development over seasons
- **Driver Form**: Real-time driver performance trends
- **Advanced Analytics**: More detailed performance breakdowns
- **API Access**: RESTful API for external integrations

### **Model Improvements**
- **Deep Learning**: Neural network enhancements
- **Time Series**: Better trend analysis
- **Ensemble Methods**: More sophisticated model combinations
- **Feature Selection**: Advanced feature importance analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Feature enhancements
- Model improvements
- UI/UX updates
- Documentation improvements

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **F1 Data**: Historical F1 data for model training
- **Scikit-learn**: Machine learning framework
- **Flask**: Web framework
- **Bootstrap**: UI framework
- **F1 Community**: Inspiration and feedback

---

**Built with machine learning and F1 passion! 🏎️**

*Predict the future of F1 driver performance with advanced AI technology.* 
# 🔥 Calorie Burn Prediction ML

A Machine Learning web app that predicts calories burned during exercise based on body metrics and workout statistics using **KNN Regression** with **99.52% accuracy**.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://calorie-burn-prediction-ml-za3q9j9qmxgczpgzqgztkv.streamlit.app)

## 📊 Dataset Features

| Feature | Description |
|---|---|
| Gender | Male / Female |
| Age | Age in years |
| Height | Height in cm |
| Weight | Weight in kg |
| Duration | Exercise duration (minutes) |
| Heart_Rate | Heart rate (bpm) |
| Body_Temp | Body temperature (°C) |
| **Calories** | **Target: Calories burned** |

## 🧠 ML Pipeline

1. **Data Preprocessing** - Z-Score outlier removal, clipping
2. **Feature Engineering** - Gender label encoding, MinMaxScaler normalization
3. **Model Selection** - Cross-validated KNN with optimal K search (1–20)
4. **Training** - KNeighborsRegressor with best K
5. **Deployment** - Streamlit web app

## 🛠️ Setup & Run Locally

`ash
# Clone the repo
git clone https://github.com/oletiVenkatasai/Calorie-Burn-Prediction-ML.git
cd Calorie-Burn-Prediction-ML

# Install dependencies
pip install -r requirements.txt

# Train the model (generates model.pkl and scaler.pkl)
python train_model.py

# Run the app
streamlit run app.py
`

## 📁 Project Structure

`
Calorie-Burn-Prediction-ML/
├── calories (3).csv       # Dataset
├── DWDM_PROJECT_KNN.ipynb # Research notebook
├── train_model.py         # Model training script
├── app.py                 # Streamlit web app
├── model.pkl              # Trained KNN model
├── scaler.pkl             # Fitted MinMaxScaler
├── requirements.txt       # Python dependencies
└── README.md
`

## 🤖 Tech Stack

- **Python** 3.x
- **Scikit-Learn** - KNN Regression
- **Pandas / NumPy** - Data processing
- **Streamlit** - Web app deployment
- **SciPy** - Statistical outlier detection
